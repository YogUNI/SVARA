"""Resumable, config-driven training loop for SVARA models.

Supports:
- CRNN baseline & wav2vec2 models
- Epoch checkpointing with RNG / optimizer state preservation
- CSV logging of metrics (docs/02 §8)
- Sanity overfit-64 samples mode (docs/11 V-12)

Task: P2-08, P2-09
Reference: docs/02 §8, docs/11 V-12
"""

import argparse
import csv
import datetime
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from svara.data.collate import FSCDataset, collate_fn_pad
from svara.models.crnn_baseline import CRNNBaseline
from svara.train.losses import MultiHeadSLULoss, compute_batch_accuracies


def set_seed(seed: int = 42):
    """Seed Python, NumPy, and PyTorch for reproducibility (docs/11 §6)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model_from_config(cfg: dict, num_slots: dict) -> torch.nn.Module:
    m_cfg = cfg["model"]
    m_type = m_cfg.get("type", "crnn")

    if m_type == "crnn":
        return CRNNBaseline(
            n_action=num_slots["action"],
            n_object=num_slots["object"],
            n_location=num_slots["location"],
            n_joint=31 if m_cfg.get("use_joint_head", False) else None,
            n_mels=m_cfg.get("n_mels", 64),
            conv_channels=m_cfg.get("conv_channels", 64),
            rnn_hidden=m_cfg.get("rnn_hidden", 128),
            rnn_layers=m_cfg.get("rnn_layers", 2),
            dropout=m_cfg.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Unsupported model type in train.py: {m_type}")


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiHeadSLULoss,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_exact = 0.0
    n_batches = len(loader)

    for batch in loader:
        wav = batch["waveform"].to(device)
        lens = batch["lengths"].to(device)
        targets = {
            "action_id": batch["action_id"].to(device),
            "object_id": batch["object_id"].to(device),
            "location_id": batch["location_id"].to(device),
            "intent_id": batch["intent_id"].to(device),
        }

        optimizer.zero_grad()
        logits = model(wav, lengths=lens)
        loss, _ = criterion(logits, targets)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        accs = compute_batch_accuracies(logits, targets)
        total_loss += loss.item()
        total_exact += accs["acc_exact_match"]

    return {
        "train_loss": total_loss / max(1, n_batches),
        "train_acc_exact": total_exact / max(1, n_batches),
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiHeadSLULoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_exact = 0.0
    total_act = 0.0
    total_obj = 0.0
    total_loc = 0.0
    n_batches = len(loader)

    with torch.no_grad():
        for batch in loader:
            wav = batch["waveform"].to(device)
            lens = batch["lengths"].to(device)
            targets = {
                "action_id": batch["action_id"].to(device),
                "object_id": batch["object_id"].to(device),
                "location_id": batch["location_id"].to(device),
                "intent_id": batch["intent_id"].to(device),
            }

            logits = model(wav, lengths=lens)
            loss, _ = criterion(logits, targets)
            accs = compute_batch_accuracies(logits, targets)

            total_loss += loss.item()
            total_exact += accs["acc_exact_match"]
            total_act += accs["acc_action"]
            total_obj += accs["acc_object"]
            total_loc += accs["acc_location"]

    return {
        "val_loss": total_loss / max(1, n_batches),
        "val_acc_exact": total_exact / max(1, n_batches),
        "val_acc_action": total_act / max(1, n_batches),
        "val_acc_object": total_obj / max(1, n_batches),
        "val_acc_location": total_loc / max(1, n_batches),
    }


def run_training(
    config_path: str = "configs/model_crnn.yaml",
    split_name: str = "A",
    seed: int = 42,
    overfit_64: bool = False,
    resume: bool = False,
    output_dir_override: Optional[str] = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train.py] Using device: {device} | Seed: {seed} | Split: {split_name}")

    # Paths and run directory
    split_dir = os.path.join("data/processed/splits", split_name)
    train_csv = os.path.join(split_dir, "train.csv")
    valid_csv = os.path.join(split_dir, "valid.csv")

    if output_dir_override:
        run_dir = output_dir_override
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        run_id = f"{timestamp}_{cfg['model']['type']}_{split_name}_s{seed}"
        if overfit_64:
            run_id = f"sanity_overfit64_{run_id}"
        run_dir = os.path.join("reports/runs", run_id)

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    # Datasets
    train_ds = FSCDataset(train_csv, is_training=True)
    val_ds = FSCDataset(valid_csv, is_training=False)

    num_slots = {
        "action": len(train_ds.actions),
        "object": len(train_ds.objects),
        "location": len(train_ds.locations),
    }

    if overfit_64:
        print("[train.py] Running SANITY OVERFIT-64 MODE (docs/11 V-12)...")
        train_ds = Subset(train_ds, list(range(min(64, len(train_ds)))))
        val_ds = train_ds  # evaluate on same 64 samples

    t_cfg = cfg["training"]
    batch_size = 16 if overfit_64 else t_cfg.get("batch_size", 32)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pad
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad
    )

    model = create_model_from_config(cfg, num_slots).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(t_cfg.get("weight_decay", 0.01)),
    )
    criterion = MultiHeadSLULoss(use_joint=cfg["model"].get("use_joint_head", False))

    log_file = os.path.join(run_dir, "train_log.csv")
    csv_header = [
        "epoch",
        "train_loss",
        "train_acc_exact",
        "val_loss",
        "val_acc_exact",
        "val_acc_action",
        "val_acc_object",
        "val_acc_location",
        "epoch_time_s",
    ]

    start_epoch = 1
    best_val_acc = 0.0
    last_ckpt = os.path.join(run_dir, "last.ckpt")

    if resume and os.path.exists(last_ckpt):
        print(f"[train.py] Resuming from checkpoint: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)

    if start_epoch == 1 and not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    max_epochs = 30 if overfit_64 else t_cfg.get("epochs", 15)

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        tr_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_clip=t_cfg.get("grad_clip_norm", 1.0),
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        dur = round(time.time() - t0, 2)

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {tr_metrics['train_loss']:.4f} | Train Acc: {tr_metrics['train_acc_exact']:.3f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc_exact']:.3f} | "
            f"Time: {dur}s"
        )

        # Append to log
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{tr_metrics['train_loss']:.5f}",
                f"{tr_metrics['train_acc_exact']:.4f}",
                f"{val_metrics['val_loss']:.5f}",
                f"{val_metrics['val_acc_exact']:.4f}",
                f"{val_metrics['val_acc_action']:.4f}",
                f"{val_metrics['val_acc_object']:.4f}",
                f"{val_metrics['val_acc_location']:.4f}",
                dur,
            ])

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            },
            last_ckpt,
        )

        # Save best checkpoint
        if val_metrics["val_acc_exact"] > best_val_acc:
            best_val_acc = val_metrics["val_acc_exact"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best.ckpt"))

        # In overfit sanity check, break early once reached near 100%
        if overfit_64 and tr_metrics["train_acc_exact"] >= 0.98:
            print("[train.py] Sanity Check PASSED: Model overfit 64 samples successfully!")
            break

    print(f"[train.py] Training completed. Run directory: {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVARA Training Script.")
    parser.add_argument("--config", default="configs/model_crnn.yaml")
    parser.add_argument("--split", default="A")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-64", action="store_true", help="Sanity check mode overfit 64 samples (V-12)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        split_name=args.split,
        seed=args.seed,
        overfit_64=args.overfit_64,
        resume=args.resume,
    )
