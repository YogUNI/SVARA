"""Train Robust Bilingual SLU Model on RTX 4060 GPU with Online Data Augmentation.

Step 3 of Robust SLU Pipeline:
- Train Dataset: 3,906 balanced clips (50% FSC native English + 50% acoustic-augmented Indonesian).
- Online Audio Augmentation: Random gain [-6dB, +6dB], speed perturbation [0.9x, 1.1x], SpecAugment masking.
- Mixed Precision: PyTorch AMP FP16 on NVIDIA GeForce RTX 4060.
- Dual Benchmark Evaluation: Evaluates on Unseen Indonesian Test Set and English FSC Test Set.
- Output Checkpoint: reports/runs/robust_bilingual_w2v2/best.ckpt
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from svara.data.collate import FSCDataset, collate_fn_pad
from svara.models.wav2vec_slu import Wav2VecSLU
from svara.train.losses import MultiHeadSLULoss, compute_batch_accuracies
from svara.data.augment import apply_random_gain, apply_speed_perturbation

class RobustBilingualDataset(FSCDataset):
    """FSCDataset with online real-time acoustic augmentations during training."""
    
    def __init__(self, *args, augment: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = augment
        
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if self.augment:
            wav = item["waveform"].numpy()
            
            # 1. Random Gain Scaling (-6 dB to +6 dB)
            if np.random.rand() < 0.6:
                wav = apply_random_gain(wav, min_gain_db=-6.0, max_gain_db=6.0)
                
            # 2. Speed Perturbation (0.9x to 1.1x)
            if np.random.rand() < 0.4:
                wav = apply_speed_perturbation(wav, speed_factors=(0.92, 1.0, 1.08))
                
            # 3. Ambient noise floor injection (simulating laptop mic hiss)
            if np.random.rand() < 0.4:
                noise = np.random.randn(len(wav)).astype(np.float32) * np.random.uniform(0.002, 0.015)
                wav = wav + noise
                
            item["waveform"] = torch.from_numpy(wav.astype(np.float32))
            item["length"] = len(wav)
        return item

def train_robust_bilingual():
    parser = argparse.ArgumentParser(description="Train Robust Bilingual SLU Model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr_encoder", type=float, default=2e-5, help="Learning rate for wav2vec2 encoder")
    parser.add_argument("--lr_heads", type=float, default=3e-4, help="Learning rate for heads")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Vocabularies
    with open("configs/intent_map.json", "r", encoding="utf-8") as f:
        intent_map = json.load(f)
    slot_vocab = intent_map["slot_vocab"]

    # 2. Datasets
    train_csv = "data/bilingual_dataset/splits/train_bilingual.csv"
    id_test_csv = "data/bilingual_dataset/splits/test_indonesian.csv"
    en_test_csv = "data/bilingual_dataset/splits/test_english.csv"

    train_dataset = RobustBilingualDataset(train_csv, augment=True, is_training=True)
    # Validation uses Indonesian Unseen Test Set for model selection
    valid_dataset = FSCDataset("data/synthetic_indonesian/splits/valid.csv", dataset_root="data/synthetic_indonesian")
    id_test_dataset = FSCDataset(id_test_csv)
    en_test_dataset = FSCDataset(en_test_csv)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_pad)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_pad)
    id_test_loader = DataLoader(id_test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_pad)
    en_test_loader = DataLoader(en_test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_pad)

    print(f"Train samples (Bilingual Balanced): {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test Indonesia Unseen: {len(id_test_dataset)} | Test English FSC: {len(en_test_dataset)}")

    # 3. Model
    model = Wav2VecSLU(
        pretrained_model_name_or_path="facebook/wav2vec2-base",
        n_action=len(slot_vocab["action"]),
        n_object=len(slot_vocab["object"]),
        n_location=len(slot_vocab["location"]),
        dropout=0.1,
    ).to(device)

    # Transfer from best mature checkpoint
    base_ckpt = "reports/runs/bilingual_indonesian_w2v2/best.ckpt"
    if os.path.exists(base_ckpt):
        print(f"Transferring weights from: {base_ckpt}")
        ckpt = torch.load(base_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("Pre-trained weights transferred successfully!")

    # 4. Optimizer & Schedulers
    encoder_params = [p for p in model.enc.parameters() if p.requires_grad]
    head_params = list(model.heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder, "weight_decay": 0.01},
        {"params": head_params, "lr": args.lr_heads, "weight_decay": 0.01},
    ])

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = MultiHeadSLULoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    out_dir = "reports/runs/robust_bilingual_w2v2"
    os.makedirs(out_dir, exist_ok=True)
    best_exact_match = 0.0
    best_ckpt_path = os.path.join(out_dir, "best.ckpt")

    print("\n" + "="*50)
    print(f"Memulai Training STEP 3: Robust Bilingual SLU ({args.epochs} Epochs)...")
    print("="*50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        train_exact_hits = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for batch in pbar:
            waveforms = batch["waveform"].to(device)
            lengths = batch["lengths"].to(device)
            targets = {
                "action_id": batch["action_id"].to(device),
                "object_id": batch["object_id"].to(device),
                "location_id": batch["location_id"].to(device),
                "action": batch["action_id"].to(device),
                "object": batch["object_id"].to(device),
                "location": batch["location_id"].to(device),
            }

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(waveforms, lengths)
                loss, _ = criterion(logits, targets)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            total_loss += loss.item() * len(lengths)
            accs = compute_batch_accuracies(logits, targets)
            train_exact_hits += int(round(accs["acc_exact_match"] * len(lengths)))
            total_train += len(lengths)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "em": f"{accs['acc_exact_match']*100:.1f}%"})

        train_loss = total_loss / total_train
        train_em = train_exact_hits / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_exact_hits = 0
        total_val = 0

        with torch.no_grad():
            for batch in valid_loader:
                waveforms = batch["waveform"].to(device)
                lengths = batch["lengths"].to(device)
                targets = {
                    "action_id": batch["action_id"].to(device),
                    "object_id": batch["object_id"].to(device),
                    "location_id": batch["location_id"].to(device),
                }

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(waveforms, lengths)
                    loss, _ = criterion(logits, targets)

                val_loss += loss.item() * len(lengths)
                accs = compute_batch_accuracies(logits, targets)
                val_exact_hits += int(round(accs["acc_exact_match"] * len(lengths)))
                total_val += len(lengths)

        val_loss /= total_val
        val_em = val_exact_hits / total_val

        print(f"-> Epoch {epoch:02d} Summary | Train Loss: {train_loss:.4f} | Train EM: {train_em*100:.2f}% | Val Loss: {val_loss:.4f} | Val EM: {val_em*100:.2f}%")

        if val_em >= best_exact_match:
            best_exact_match = val_em
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_exact_match": val_em,
                "val_loss": val_loss,
            }, best_ckpt_path)
            print(f"   [SAVED] Best checkpoint updated! Val EM: {val_em*100:.2f}%")

    print("\n" + "="*50)
    print(f"Training Selesai! Model tersimpan di: {best_ckpt_path}")
    print("="*50)

    # 5. Dual Evaluation (Indonesian Unseen + English FSC)
    print("\nMenjalankan Evaluasi Ilmiah pada Test Set Bahasa Indonesia (155 sampel tak terlihat)...")
    best_ckpt_data = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt_data["model_state_dict"])
    model.eval()

    def eval_loader(loader):
        hits, act_hits, obj_hits, loc_hits, total = 0, 0, 0, 0, 0
        with torch.no_grad():
            for b in loader:
                w = b["waveform"].to(device)
                l = b["lengths"].to(device)
                t = {
                    "action_id": b["action_id"].to(device),
                    "object_id": b["object_id"].to(device),
                    "location_id": b["location_id"].to(device),
                }
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(w, l)
                accs = compute_batch_accuracies(logits, t)
                hits += int(round(accs["acc_exact_match"] * len(l)))
                act_hits += int(round(accs["acc_action"] * len(l)))
                obj_hits += int(round(accs["acc_object"] * len(l)))
                loc_hits += int(round(accs["acc_location"] * len(l)))
                total += len(l)
        return {
            "exact_match": hits / total,
            "action_acc": act_hits / total,
            "object_acc": obj_hits / total,
            "location_acc": loc_hits / total,
            "total_samples": total,
        }

    id_results = eval_loader(id_test_loader)
    print("\n[HASIL EVALUASI BAHASA INDONESIA UNSEEN]")
    print(f"- Exact-Match Accuracy: {id_results['exact_match']*100:.2f}%")
    print(f"- Action Accuracy:      {id_results['action_acc']*100:.2f}%")
    print(f"- Object Accuracy:      {id_results['object_acc']*100:.2f}%")
    print(f"- Location Accuracy:    {id_results['location_acc']*100:.2f}%")

    print("\nMenjalankan Evaluasi Ilmiah pada Test Set Bahasa Inggris (3.793 sampel native FSC)...")
    en_results = eval_loader(en_test_loader)
    print("\n[HASIL EVALUASI BAHASA INGGRIS FSC]")
    print(f"- Exact-Match Accuracy: {en_results['exact_match']*100:.2f}%")
    print(f"- Action Accuracy:      {en_results['action_acc']*100:.2f}%")
    print(f"- Object Accuracy:      {en_results['object_acc']*100:.2f}%")
    print(f"- Location Accuracy:    {en_results['location_acc']*100:.2f}%")

    combined_metrics = {
        "indonesian": id_results,
        "english": en_results,
    }
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2)

    # Save run config
    run_cfg = {
        "model": {
            "type": "wav2vec2",
            "pretrained_model": "facebook/wav2vec2-base",
            "dropout": 0.1,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dataset": "bilingual_balanced_robust",
            "augmentations": ["gain_scaling", "speed_perturbation", "noise_injection"],
        }
    }
    import yaml
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(run_cfg, f)

if __name__ == "__main__":
    train_robust_bilingual()
