"""Hardened Bilingual SLU Training Script on GPU/CPU with Real-World Voice Hardening.

Task: Sub-step 14.1 / P14-01
Reference: docs/14 §3 FASE 14, docs/11 V-11..V-18

Features:
- Backbone: facebook/wav2vec2-base (Frozen temporal CNN feature encoder, trainable 12-layer Transformer)
- Dual-Level SpecAugment: Waveform STFT and model-level feature masking
- Acoustic Anchor Injection: Real human recordings from data/own_recordings/metadata.csv
- Pitch & Prosody Hardening: Supports pitch-shifted data and expanded colloquial corpus
- Mixed Precision: PyTorch AMP FP16 when CUDA is available
- Output: models/svara_hardened.pt and reports/models/hardened_train_history.json
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.svara.data.audio import preprocess_audio
from src.svara.data.augment import apply_random_gain, apply_speed_perturbation, apply_spec_augment_waveform
from src.svara.data.collate import collate_fn_pad
from src.svara.models.wav2vec_slu import Wav2VecSLU
from src.svara.train.losses import MultiHeadSLULoss, compute_batch_accuracies


class HardenedBilingualDataset(Dataset):
    """Bilingual SLU Dataset integrating Synthetic Pitch variations and Real Human Acoustic Anchors."""

    def __init__(
        self,
        manifest_path: str,
        own_recordings_path: Optional[str] = "data/own_recordings/metadata.csv",
        intent_map_path: str = "configs/intent_map.json",
        augment: bool = True,
        is_training: bool = True,
        max_samples: Optional[int] = None,
        anchor_multiplier: int = 5,
    ):
        self.augment = augment
        self.is_training = is_training

        with open(intent_map_path, "r", encoding="utf-8") as f:
            imap_data = json.load(f)
        self.slot_vocab = imap_data["slot_vocab"]
        self.action_to_idx = {a: i for i, a in enumerate(self.slot_vocab["action"])}
        self.object_to_idx = {o: i for i, o in enumerate(self.slot_vocab["object"])}
        self.location_to_idx = {l: i for i, l in enumerate(self.slot_vocab["location"])}

        # Load main manifest
        self.records: List[Dict] = []
        if os.path.exists(manifest_path):
            df_main = pd.read_csv(manifest_path)
            col_path = "audio_path" if "audio_path" in df_main.columns else "path"
            col_intent = "intent_id" if "intent_id" in df_main.columns else None

            paths = df_main[col_path].astype(str).tolist()
            actions = df_main["action"].astype(str).tolist() if "action" in df_main.columns else [""] * len(paths)
            objects = df_main["object"].astype(str).tolist() if "object" in df_main.columns else [""] * len(paths)
            locations = df_main["location"].astype(str).tolist() if "location" in df_main.columns else [""] * len(paths)
            intents = df_main[col_intent].fillna(-1).astype(int).tolist() if col_intent else [-1] * len(paths)

            self.records = [
                {
                    "path": p,
                    "action": a,
                    "object": o,
                    "location": l,
                    "intent_id": i,
                    "is_anchor": False,
                }
                for p, a, o, l, i in zip(paths, actions, objects, locations, intents)
            ]

        # Inject Real Human Voice Acoustic Anchors if present
        if own_recordings_path and os.path.exists(own_recordings_path):
            df_own = pd.read_csv(own_recordings_path)
            print(f"[Dataset] Found {len(df_own)} real human recordings in {own_recordings_path}!", flush=True)
            col_path = "audio_path" if "audio_path" in df_own.columns else "path"
            col_intent = "intent_id" if "intent_id" in df_own.columns else None

            paths = df_own[col_path].astype(str).tolist()
            actions = df_own["action"].astype(str).tolist() if "action" in df_own.columns else [""] * len(paths)
            objects = df_own["object"].astype(str).tolist() if "object" in df_own.columns else [""] * len(paths)
            locations = df_own["location"].astype(str).tolist() if "location" in df_own.columns else [""] * len(paths)
            intents = df_own[col_intent].fillna(-1).astype(int).tolist() if col_intent else [-1] * len(paths)

            anchor_records = [
                {
                    "path": p,
                    "action": a,
                    "object": o,
                    "location": l,
                    "intent_id": i,
                    "is_anchor": True,
                }
                for p, a, o, l, i in zip(paths, actions, objects, locations, intents)
            ]
            for _ in range(anchor_multiplier):
                self.records.extend(anchor_records)
            print(f"[Dataset] Injected {len(anchor_records) * anchor_multiplier} oversampled human anchor instances.", flush=True)

        if max_samples and max_samples < len(self.records):
            n_anchors = (
                len(anchor_records) * anchor_multiplier
                if (own_recordings_path and os.path.exists(own_recordings_path))
                else 0
            )
            if n_anchors > 0:
                main_recs = self.records[:-n_anchors]
                anchor_recs = self.records[-n_anchors:]
                n_main_keep = max(0, max_samples - n_anchors)
                self.records = main_recs[:n_main_keep] + anchor_recs
            else:
                self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        audio_path = rec["path"]

        # Resolve relative to project root if needed
        if not os.path.isabs(audio_path) and not os.path.exists(audio_path):
            cand = os.path.join(str(PROJECT_ROOT), audio_path)
            if os.path.exists(cand):
                audio_path = cand

        if os.path.exists(audio_path):
            try:
                data, sr = sf.read(audio_path, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                wav, orig_len, sr = preprocess_audio(
                    data,
                    sample_rate=sr,
                    target_sample_rate=16000,
                    max_audio_seconds=5.0,
                    normalize=True,
                    is_training=self.is_training,
                )
            except Exception:
                # Fallback to dummy silence on decode error
                wav = np.zeros(16000, dtype=np.float32)
        else:
            # Placeholder for offline tests
            wav = np.zeros(16000, dtype=np.float32)

        # Online Acoustic Augmentations during Training
        if self.augment and self.is_training and len(wav) > 1600:
            # Random gain (-6 dB to +6 dB)
            if np.random.rand() < 0.5:
                wav = apply_random_gain(wav, -6.0, 6.0)
            # Speed perturbation (0.9x to 1.1x)
            if np.random.rand() < 0.3:
                wav = apply_speed_perturbation(wav, (0.92, 1.0, 1.08))
            # Waveform-level SpecAugment (Complex STFT time/freq masking)
            if np.random.rand() < 0.4:
                wav_t = torch.from_numpy(wav).unsqueeze(0)
                wav_aug = apply_spec_augment_waveform(wav_t, freq_mask_param=20, time_mask_param=25)
                wav = wav_aug.squeeze(0).cpu().numpy()

        act_idx = self.action_to_idx.get(rec["action"], 0)
        obj_idx = self.object_to_idx.get(rec["object"], 0)
        loc_idx = self.location_to_idx.get(rec["location"], 0)
        joint_idx = rec["intent_id"] if rec["intent_id"] >= 0 else 0

        return {
            "waveform": torch.from_numpy(wav.astype(np.float32)),
            "length": torch.tensor(len(wav), dtype=torch.long),
            "lengths": torch.tensor(len(wav), dtype=torch.long),
            "action": torch.tensor(act_idx, dtype=torch.long),
            "object": torch.tensor(obj_idx, dtype=torch.long),
            "location": torch.tensor(loc_idx, dtype=torch.long),
            "joint": torch.tensor(joint_idx, dtype=torch.long),
            "action_id": torch.tensor(act_idx, dtype=torch.long),
            "object_id": torch.tensor(obj_idx, dtype=torch.long),
            "location_id": torch.tensor(loc_idx, dtype=torch.long),
            "intent_id": torch.tensor(joint_idx, dtype=torch.long),
        }


def train_hardened_bilingual(
    epochs: int = 10,
    batch_size: int = 16,
    lr_encoder: float = 3e-5,
    lr_heads: float = 1e-3,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.05,
    device_str: Optional[str] = None,
    dry_run: bool = False,
    max_train_samples: Optional[int] = None,
):
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Intent Vocabulary
    intent_map_path = str(PROJECT_ROOT / "configs" / "intent_map.json")
    with open(intent_map_path, "r", encoding="utf-8") as f:
        imap = json.load(f)
    slot_vocab = imap["slot_vocab"]

    n_act = len(slot_vocab["action"])
    n_obj = len(slot_vocab["object"])
    n_loc = len(slot_vocab["location"])

    # 2. Setup Datasets & DataLoaders
    # Priority manifests
    manifest_candidates = [
        str(PROJECT_ROOT / "data" / "bilingual_massive" / "splits" / "train.csv"),
        str(PROJECT_ROOT / "data" / "synthetic_indonesian_massive" / "metadata_pitch.csv"),
        str(PROJECT_ROOT / "data" / "bilingual_dataset" / "splits" / "train_bilingual.csv"),
    ]
    train_manifest = manifest_candidates[0]
    for c in manifest_candidates:
        if os.path.exists(c):
            train_manifest = c
            break

    print(f"[Train] Using Training Manifest: {train_manifest}")
    own_path = str(PROJECT_ROOT / "data" / "own_recordings" / "metadata.csv")

    train_dataset = HardenedBilingualDataset(
        manifest_path=train_manifest,
        own_recordings_path=own_path if os.path.exists(own_path) else None,
        intent_map_path=intent_map_path,
        augment=True,
        is_training=True,
        max_samples=64 if dry_run else max_train_samples,
    )

    valid_manifest = str(PROJECT_ROOT / "data" / "bilingual_massive" / "splits" / "valid.csv")
    if not os.path.exists(valid_manifest):
        valid_manifest = train_manifest

    valid_dataset = HardenedBilingualDataset(
        manifest_path=valid_manifest,
        own_recordings_path=None,
        intent_map_path=intent_map_path,
        augment=False,
        is_training=False,
        max_samples=32 if dry_run else 300,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_fn_pad,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=min(batch_size, len(valid_dataset)),
        shuffle=False,
        collate_fn=collate_fn_pad,
    )

    print(f"[Train] Total Training Samples: {len(train_dataset)} | Validation Samples: {len(valid_dataset)}")

    # 3. Model Architecture with SpecAugment
    print("[Train] Initializing Wav2VecSLU with Dual-Level SpecAugment...", flush=True)
    model = Wav2VecSLU(
        pretrained_model_name_or_path="facebook/wav2vec2-base",
        n_action=n_act,
        n_object=n_obj,
        n_location=n_loc,
        n_joint=31,
        dropout=0.1,
        freeze_feature_encoder=True,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=20,
        mask_feature_prob=0.05,
        mask_feature_length=20,
    )

    # Warm-start from pre-trained bilingual checkpoint if present
    ckpt_candidates = [
        str(PROJECT_ROOT / "reports" / "models" / "best_slu_model.pt"),
        str(PROJECT_ROOT / "models" / "svara_hardened.pt"),
    ]
    for cpath in ckpt_candidates:
        if os.path.exists(cpath):
            try:
                ckpt = torch.load(cpath, map_location=device)
                state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
                model.load_state_dict(state_dict, strict=False)
                print(f"[Train] Successfully warm-started from pre-trained weights: {cpath}", flush=True)
                break
            except Exception as e:
                print(f"[Train] Could not load checkpoint {cpath}: {e}", flush=True)

    model.to(device)

    # 4. Decoupled Optimizer & Cosine Scheduler
    encoder_params = [p for n, p in model.named_parameters() if "enc" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "enc" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_heads, "weight_decay": weight_decay},
        ]
    )

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = MultiHeadSLULoss(label_smoothing=label_smoothing, joint_weight=0.2)
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and torch.cuda.is_available()) else None

    # 5. Training Loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_action_acc": [],
        "val_object_acc": [],
        "val_location_acc": [],
    }

    best_val_acc = 0.0
    os.makedirs(str(PROJECT_ROOT / "models"), exist_ok=True)
    os.makedirs(str(PROJECT_ROOT / "reports" / "models"), exist_ok=True)
    best_ckpt_path = str(PROJECT_ROOT / "models" / "svara_hardened.pt")

    print("=" * 70)
    print("STARTING HARDENED BILINGUAL SLU RETRAINING")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Encoder LR: {lr_encoder} | Heads LR: {lr_heads}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        train_exact_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            waveforms = batch["waveform"].to(device)
            raw_lengths = batch.get("lengths", batch.get("length"))
            lengths = raw_lengths.to(device)
            act_t = batch.get("action", batch.get("action_id")).to(device)
            obj_t = batch.get("object", batch.get("object_id")).to(device)
            loc_t = batch.get("location", batch.get("location_id")).to(device)
            joint_t = batch.get("joint", batch.get("intent_id", torch.zeros_like(act_t))).to(device)
            targets = {
                "action": act_t,
                "object": obj_t,
                "location": loc_t,
                "joint": joint_t,
                "action_id": act_t,
                "object_id": obj_t,
                "location_id": loc_t,
                "intent_id": joint_t,
            }

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(waveforms, lengths)
                    loss, _ = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(waveforms, lengths)
                loss, _ = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            bs = waveforms.size(0)
            total_train_loss += loss.item() * bs
            accs = compute_batch_accuracies(outputs, targets)
            train_exact_correct += int(accs["exact_match"] * bs)
            train_total += bs

        train_loss = total_train_loss / max(train_total, 1)
        train_acc = train_exact_correct / max(train_total, 1)

        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        val_exact_correct = 0
        val_act_correct = 0
        val_obj_correct = 0
        val_loc_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in valid_loader:
                waveforms = batch["waveform"].to(device)
                raw_lengths = batch.get("lengths", batch.get("length"))
                lengths = raw_lengths.to(device)
                act_t = batch.get("action", batch.get("action_id")).to(device)
                obj_t = batch.get("object", batch.get("object_id")).to(device)
                loc_t = batch.get("location", batch.get("location_id")).to(device)
                joint_t = batch.get("joint", batch.get("intent_id", torch.zeros_like(act_t))).to(device)
                targets = {
                    "action": act_t,
                    "object": obj_t,
                    "location": loc_t,
                    "joint": joint_t,
                    "action_id": act_t,
                    "object_id": obj_t,
                    "location_id": loc_t,
                    "intent_id": joint_t,
                }

                outputs = model(waveforms, lengths)
                loss, _ = criterion(outputs, targets)

                bs = waveforms.size(0)
                total_val_loss += loss.item() * bs
                accs = compute_batch_accuracies(outputs, targets)
                val_exact_correct += int(accs["exact_match"] * bs)
                val_act_correct += int(accs["action"] * bs)
                val_obj_correct += int(accs["object"] * bs)
                val_loc_correct += int(accs["location"] * bs)
                val_total += bs

        val_loss = total_val_loss / max(val_total, 1)
        val_acc = val_exact_correct / max(val_total, 1)
        val_act = val_act_correct / max(val_total, 1)
        val_obj = val_obj_correct / max(val_total, 1)
        val_loc = val_loc_correct / max(val_total, 1)
        elapsed = time.time() - t0

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_action_acc"].append(round(val_act, 4))
        history["val_object_acc"].append(round(val_obj, 4))
        history["val_location_acc"].append(round(val_loc, 4))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Exact: {val_acc*100:.2f}% (Act: {val_act*100:.1f}%, Obj: {val_obj*100:.1f}%, Loc: {val_loc*100:.1f}%)",
            flush=True,
        )

        if val_acc > best_val_acc or (epoch == 1 and val_acc >= best_val_acc):
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                best_ckpt_path,
            )
            print(f"  -> Best model checkpoint saved to: {best_ckpt_path}")

    # Save training history
    history_path = str(PROJECT_ROOT / "reports" / "models" / "hardened_train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Train] Training history written to: {history_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Retrain Hardened Bilingual SLU Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr_encoder", type=float, default=3e-5, help="Learning rate for wav2vec2 encoder")
    parser.add_argument("--lr_heads", type=float, default=1e-3, help="Learning rate for heads")
    parser.add_argument("--device", type=str, default=None, help="Target device (cuda/cpu)")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 quick epoch on small batch to test pipeline")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum training samples")
    args = parser.parse_args()

    train_hardened_bilingual(
        epochs=1 if args.dry_run else args.epochs,
        batch_size=args.batch_size,
        lr_encoder=args.lr_encoder,
        lr_heads=args.lr_heads,
        device_str=args.device,
        dry_run=args.dry_run,
        max_train_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

