"""Fine-tune SVARA wav2vec2 model to become Bilingual (English + Indonesian).

Takes the best FSC English checkpoint, continues fine-tuning on Indonesian speech commands
using RTX 4060 GPU with mixed precision (fp16), and evaluates exact-match accuracy.
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
from transformers import get_linear_schedule_with_warmup

from svara.data.collate import FSCDataset, collate_fn_pad
from svara.models.wav2vec_slu import Wav2VecSLU
from svara.train.losses import MultiHeadSLULoss, compute_batch_accuracies

def train_indonesian():
    parser = argparse.ArgumentParser(description="Fine-tune SVARA wav2vec2 model for Indonesian commands")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr_encoder", type=float, default=2e-5, help="Learning rate for wav2vec2 encoder")
    parser.add_argument("--lr_heads", type=float, default=3e-4, help="Learning rate for classification heads")
    parser.add_argument("--base_checkpoint", type=str, default="reports/runs/20261009-1953_wav2vec2_A_s42/best.ckpt",
                        help="Path to pre-trained English checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Vocabularies
    with open("configs/intent_map.json", "r", encoding="utf-8") as f:
        intent_map = json.load(f)
    slot_vocab = intent_map["slot_vocab"]
    n_action = len(slot_vocab["action"])
    n_object = len(slot_vocab["object"])
    n_location = len(slot_vocab["location"])

    # 2. Datasets & DataLoaders
    indonesian_root = "data/synthetic_indonesian"
    train_dataset = FSCDataset(
        split_csv_path="data/synthetic_indonesian/splits/train.csv",
        dataset_root=indonesian_root,
        max_audio_seconds=5.0,
        is_training=True,
    )
    valid_dataset = FSCDataset(
        split_csv_path="data/synthetic_indonesian/splits/valid.csv",
        dataset_root=indonesian_root,
        max_audio_seconds=5.0,
        is_training=False,
    )
    test_dataset = FSCDataset(
        split_csv_path="data/synthetic_indonesian/splits/test.csv",
        dataset_root=indonesian_root,
        max_audio_seconds=5.0,
        is_training=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_pad)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_pad)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_pad)

    # 3. Model Initialization & Load Base Checkpoint
    model = Wav2VecSLU(
        pretrained_model_name_or_path="facebook/wav2vec2-base",
        n_action=n_action,
        n_object=n_object,
        n_location=n_location,
        n_joint=None,
        dropout=0.1,
    ).to(device)

    if os.path.exists(args.base_checkpoint):
        print(f"Memuat bobot pre-trained dari: {args.base_checkpoint}")
        ckpt = torch.load(args.base_checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("Bobot pre-trained berhasil dimuat!")
    else:
        print(f"Peringatan: Checkpoint {args.base_checkpoint} tidak ditemukan, melatih dari base model.")

    # 4. Optimizer & Schedulers
    encoder_params = [p for p in model.enc.parameters() if p.requires_grad]
    head_params = list(model.heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoder, "weight_decay": 0.01},
        {"params": head_params, "lr": args.lr_heads, "weight_decay": 0.01},
    ])

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion = MultiHeadSLULoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # 5. Training Loop
    out_dir = "reports/runs/bilingual_indonesian_w2v2"
    os.makedirs(out_dir, exist_ok=True)
    best_exact_match = 0.0
    best_ckpt_path = os.path.join(out_dir, "best.ckpt")

    print("\n" + "="*50)
    print(f"Memulai Fine-Tuning Bilingual SVARA ({args.epochs} Epochs)...")
    print("="*50)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        train_exact_hits = 0
        total_train = 0

        for batch in train_loader:
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

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Train EM: {train_em*100:.2f}% | Val Loss: {val_loss:.4f} | Val EM: {val_em*100:.2f}%")

        if val_em > best_exact_match:
            best_exact_match = val_em
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_exact_match": val_em,
                "val_loss": val_loss,
            }, best_ckpt_path)

    print("\n" + "="*50)
    print(f"Training Selesai! Model terbaik tersimpan di: {best_ckpt_path}")
    print(f"Akurasi Validasi Tertinggi: {best_exact_match*100:.2f}%")
    print("="*50)

    # 6. Evaluasi pada Test Set Bahasa Indonesia
    print("\nMenjalankan Evaluasi pada Test Set Bahasa Indonesia (155 sampel tak terlihat)...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_exact_hits = 0
    test_act_hits = 0
    test_obj_hits = 0
    test_loc_hits = 0
    total_test = 0

    with torch.no_grad():
        for batch in test_loader:
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

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(waveforms, lengths)

            accs = compute_batch_accuracies(logits, targets)
            test_exact_hits += int(round(accs["acc_exact_match"] * len(lengths)))
            test_act_hits += int(round(accs["acc_action"] * len(lengths)))
            test_obj_hits += int(round(accs["acc_object"] * len(lengths)))
            test_loc_hits += int(round(accs["acc_location"] * len(lengths)))
            total_test += len(lengths)

    test_results = {
        "test_exact_match": test_exact_hits / total_test,
        "test_action_acc": test_act_hits / total_test,
        "test_object_acc": test_obj_hits / total_test,
        "test_location_acc": test_loc_hits / total_test,
        "total_samples": total_test,
    }

    print("\n[HASIL EVALUASI MODEL BAHASA INDONESIA]")
    print(f"- Exact-Match Accuracy: {test_results['test_exact_match']*100:.2f}%")
    print(f"- Action Accuracy:      {test_results['test_action_acc']*100:.2f}%")
    print(f"- Object Accuracy:      {test_results['test_object_acc']*100:.2f}%")
    print(f"- Location Accuracy:    {test_results['test_location_acc']*100:.2f}%")

    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

if __name__ == "__main__":
    train_indonesian()
