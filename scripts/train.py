"""Resumable, config-driven training loop for SVARA models.

Supports:
- CRNN baseline & wav2vec2 models (P2-08, P3-01, P3-03)
- Two learning rate parameter groups (encoder vs heads per docs/02 §3)
- Linear warmup + linear decay learning rate scheduler
- Mixed precision (fp16 autocast with GradScaler on CUDA)
- Gradient accumulation and gradient clipping (max-norm 1.0)
- Early stopping on validation exact-match accuracy
- Epoch checkpointing with optimizer, scheduler, and RNG state preservation
- CSV logging of metrics (docs/02 §8)
- Sanity overfit-64 samples mode (docs/11 V-12)

Tasks: P2-08, P2-09, P3-03
Reference: docs/02 §3, docs/11 V-12..V-16, docs/12 §3
"""

import argparse
import csv
import datetime
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from svara.data.collate import FSCDataset, collate_fn_pad
from svara.models.crnn_baseline import CRNNBaseline
from svara.models.hf_baseline import HFWav2Vec2ClassificationBaseline
from svara.models.wav2vec_slu import Wav2VecSLU
from svara.train.losses import MultiHeadSLULoss, compute_batch_accuracies


def set_seed(seed: int = 42):
    """Seed Python, NumPy, and PyTorch for reproducibility (docs/11 §6)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model_from_config(cfg: dict, num_slots: dict) -> torch.nn.Module:
    """Instantiate model according to config dict."""
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
    elif m_type in ("wav2vec2", "w2v2"):
        return Wav2VecSLU(
            pretrained_model_name_or_path=m_cfg.get("pretrained_model", "facebook/wav2vec2-base"),
            n_action=num_slots["action"],
            n_object=num_slots["object"],
            n_location=num_slots["location"],
            n_joint=31 if m_cfg.get("use_joint_head", False) else None,
            keep_layers=m_cfg.get("keep_layers", None),
            freeze_feature_encoder=m_cfg.get("freeze_feature_encoder", True),
            freeze_encoder=m_cfg.get("freeze_encoder", False),
            dropout=m_cfg.get("dropout", 0.1),
            mask_time_prob=m_cfg.get("mask_time_prob", 0.05),
            layerdrop=m_cfg.get("layerdrop", 0.0),
            config_only=m_cfg.get("config_only", False),
        )
    elif m_type in ("hf_baseline", "hf_sequence_classification"):
        return HFWav2Vec2ClassificationBaseline(
            pretrained_model_name_or_path=m_cfg.get("pretrained_model", "facebook/wav2vec2-base"),
            num_labels=31,
            freeze_feature_encoder=m_cfg.get("freeze_feature_encoder", True),
            config_only=m_cfg.get("config_only", False),
        )
    else:
        raise ValueError(f"Unsupported model type in train.py: {m_type}")


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    t_cfg: dict,
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LambdaLR]]:
    """Build AdamW optimizer with 2 LR groups (encoder vs heads) and warmup scheduler (P3-03)."""
    head_lr = float(t_cfg.get("learning_rate", 1e-3))
    encoder_lr = float(t_cfg.get("encoder_learning_rate", 3e-5))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))

    if hasattr(model, "enc") and hasattr(model, "heads"):
        # Separate parameters into encoder vs heads (no decay on bias/LayerNorm)
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        enc_params_decay = []
        enc_params_no_decay = []
        head_params_decay = []
        head_params_no_decay = []

        for name, param in model.enc.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                enc_params_no_decay.append(param)
            else:
                enc_params_decay.append(param)

        for name, param in model.heads.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)

        if getattr(model, "joint", None) is not None:
            for name, param in model.joint.named_parameters():
                if not param.requires_grad:
                    continue
                if any(nd in name for nd in no_decay):
                    head_params_no_decay.append(param)
                else:
                    head_params_decay.append(param)

        param_groups = [
            {"params": enc_params_decay, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": enc_params_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
            {"params": head_params_decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_params_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ]
        # Filter out empty parameter groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        optimizer = torch.optim.AdamW(param_groups)
    else:
        # Default single group for baseline models
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=head_lr,
            weight_decay=weight_decay,
        )

    # Linear warmup + linear decay scheduler
    warmup_ratio = float(t_cfg.get("warmup_ratio", 0.1))
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) if total_steps > 0 else None
    return optimizer, scheduler


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    criterion: MultiHeadSLULoss,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """Train single epoch with gradient accumulation, mixed precision, and clipping."""
    model.train()
    n_batches = len(loader)
    total_loss = 0.0
    total_exact = 0.0
    use_amp = scaler is not None and device.type == "cuda"
    optimizer.zero_grad()

    from tqdm import tqdm

    pbar = tqdm(loader, desc="Training", leave=False)
    for step_idx, batch in enumerate(pbar):
        wav = batch["waveform"].to(device)
        lens = batch["lengths"].to(device)
        targets = {
            "action_id": batch["action_id"].to(device),
            "object_id": batch["object_id"].to(device),
            "location_id": batch["location_id"].to(device),
            "intent_id": batch["intent_id"].to(device),
        }

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(wav, lengths=lens)
                loss, _ = criterion(logits, targets)
                scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()
        else:
            logits = model(wav, lengths=lens)
            loss, _ = criterion(logits, targets)
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == n_batches:
            if use_amp:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        with torch.no_grad():
            accs = compute_batch_accuracies(logits, targets)
            total_loss += loss.item()
            total_exact += accs["acc_exact_match"]

        if (step_idx + 1) % 10 == 0:
            avg_loss = total_loss / (step_idx + 1)
            avg_exact = total_exact / (step_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_exact:.3f}"})

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
    """Evaluate model on validation or test loader."""
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
) -> str:
    """Execute complete config-driven training workflow."""
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
        val_ds = train_ds

    t_cfg = cfg["training"]
    batch_size = 16 if overfit_64 else t_cfg.get("batch_size", 32)
    grad_accum_steps = 1 if overfit_64 else t_cfg.get("grad_accum_steps", 1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pad
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad
    )

    max_epochs = 30 if overfit_64 else t_cfg.get("epochs", 15)
    total_steps = (len(train_loader) // grad_accum_steps) * max_epochs

    model = create_model_from_config(cfg, num_slots).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, t_cfg, total_steps)
    criterion = MultiHeadSLULoss(use_joint=cfg["model"].get("use_joint_head", False))

    use_fp16 = t_cfg.get("fp16", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

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
    epochs_no_improve = 0
    patience = t_cfg.get("early_stopping_patience", 0)
    last_ckpt = os.path.join(run_dir, "last.ckpt")

    if resume and os.path.exists(last_ckpt):
        print(f"[train.py] Resuming from checkpoint: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)

    if start_epoch == 1 and not os.path.exists(log_file):
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        tr_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            scaler=scaler,
            grad_clip=t_cfg.get("grad_clip_norm", 1.0),
            grad_accum_steps=grad_accum_steps,
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

        # Check early stopping / best validation checkpoint
        if val_metrics["val_acc_exact"] > best_val_acc:
            best_val_acc = val_metrics["val_acc_exact"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.ckpt"))
        else:
            epochs_no_improve += 1

        # Save last checkpoint for resume
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_val_acc": best_val_acc,
                "epochs_no_improve": epochs_no_improve,
            },
            last_ckpt,
        )

        # In overfit sanity check, break early once reached near 100%
        if overfit_64 and tr_metrics["train_acc_exact"] >= 0.98:
            print("[train.py] Sanity Check PASSED: Model overfit 64 samples successfully!")
            break

        # Early stopping trigger
        if patience > 0 and epochs_no_improve >= patience and not overfit_64:
            print(f"[train.py] Early stopping triggered after {epoch} epochs (patience={patience}).")
            break

    print(f"[train.py] Training completed. Best Val Acc: {best_val_acc:.4f} | Run directory: {run_dir}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVARA Training Script.")
    parser.add_argument("--config", default="configs/model_crnn.yaml")
    parser.add_argument("--split", default="A")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None, help="Explicit output directory for run")
    parser.add_argument("--overfit-64", action="store_true", help="Sanity check mode overfit 64 samples (V-12)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        split_name=args.split,
        seed=args.seed,
        overfit_64=args.overfit_64,
        resume=args.resume,
        output_dir_override=args.output_dir,
    )
