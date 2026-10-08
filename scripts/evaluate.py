"""Model evaluation script writing metrics.json, predictions.csv, and confusion CSVs.

Supports:
- Evaluating any trained checkpoint (CRNN or Wav2VecSLU)
- Calculating exact-match accuracy, per-slot accuracy, and macro-F1
- 95% bootstrap confidence interval (1,000 resamples)
- Exporting full predictions CSV and confusion matrices

Tasks: P3-05
Reference: docs/01 §5, docs/02 §8, docs/11 V-21..V-24
"""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from svara.data.collate import FSCDataset, collate_fn_pad
from svara.eval.metrics import (
    compute_bootstrap_ci,
    compute_confusion_matrices,
    compute_exact_match,
    compute_metrics,
)
from svara.models.crnn_baseline import CRNNBaseline
from svara.models.wav2vec_slu import Wav2VecSLU


def load_model_from_run(run_dir: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load intent map for slot counts
    with open("configs/intent_map.json", "r", encoding="utf-8") as f:
        imap = json.load(f)

    num_slots = {
        "action": len(imap["slot_vocab"]["action"]),
        "object": len(imap["slot_vocab"]["object"]),
        "location": len(imap["slot_vocab"]["location"]),
    }

    m_cfg = cfg["model"]
    m_type = m_cfg.get("type", "crnn")

    if m_type == "crnn":
        model = CRNNBaseline(
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
        model = Wav2VecSLU(
            pretrained_model_name_or_path=m_cfg.get("pretrained_model", "facebook/wav2vec2-base"),
            n_action=num_slots["action"],
            n_object=num_slots["object"],
            n_location=num_slots["location"],
            n_joint=31 if m_cfg.get("use_joint_head", False) else None,
            keep_layers=m_cfg.get("keep_layers", None),
            freeze_feature_encoder=m_cfg.get("freeze_feature_encoder", True),
            freeze_encoder=m_cfg.get("freeze_encoder", False),
            config_only=m_cfg.get("config_only", False),
        )
    else:
        raise ValueError(f"Unknown model type: {m_type}")

    # Checkpoint path
    best_ckpt = os.path.join(run_dir, "best.ckpt")
    last_ckpt = os.path.join(run_dir, "last.ckpt")
    ckpt_path = best_ckpt if os.path.exists(best_ckpt) else last_ckpt

    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        print(f"[evaluate.py] Loaded weights from {ckpt_path}")
    else:
        print(f"[evaluate.py] WARNING: No checkpoint found in {run_dir}, using initialized weights.")

    model = model.to(device)
    model.eval()
    return model, cfg


def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    records = []

    with torch.no_grad():
        for batch in loader:
            wav = batch["waveform"].to(device)
            lens = batch["lengths"].to(device)
            logits = model(wav, lengths=lens)

            pred_act = torch.argmax(logits["action"], dim=-1).cpu().numpy()
            pred_obj = torch.argmax(logits["object"], dim=-1).cpu().numpy()
            pred_loc = torch.argmax(logits["location"], dim=-1).cpu().numpy()

            true_act = batch["action_id"].numpy()
            true_obj = batch["object_id"].numpy()
            true_loc = batch["location_id"].numpy()
            true_int = batch["intent_id"].numpy()

            for i in range(len(wav)):
                records.append({
                    "path": batch["path"][i],
                    "speakerId": batch["speakerId"][i],
                    "transcript": batch["transcript"][i],
                    "target_action": int(true_act[i]),
                    "pred_action": int(pred_act[i]),
                    "target_object": int(true_obj[i]),
                    "pred_object": int(pred_obj[i]),
                    "target_location": int(true_loc[i]),
                    "pred_location": int(pred_loc[i]),
                    "target_intent_id": int(true_int[i]),
                    "correct_exact": bool(
                        pred_act[i] == true_act[i]
                        and pred_obj[i] == true_obj[i]
                        and pred_loc[i] == true_loc[i]
                    ),
                })

    return pd.DataFrame(records)


def run_evaluation(run_dir: str, split_name: str = "A", set_type: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate.py] Evaluating run {run_dir} on Split {split_name} ({set_type})...")

    model, cfg = load_model_from_run(run_dir, device)

    split_csv = os.path.join("data/processed/splits", split_name, f"{set_type}.csv")
    ds = FSCDataset(split_csv, is_training=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn_pad)

    df_preds = evaluate_dataset(model, loader, device)

    # Compute metrics
    metrics = compute_metrics(
        df_preds["pred_action"].values,
        df_preds["target_action"].values,
        df_preds["pred_object"].values,
        df_preds["target_object"].values,
        df_preds["pred_location"].values,
        df_preds["target_location"].values,
    )

    # 95% bootstrap CI
    mean_acc, ci_low, ci_high = compute_bootstrap_ci(df_preds["correct_exact"].values, n_resamples=1000)
    metrics["exact_match_ci95"] = [ci_low, ci_high]
    metrics["split"] = split_name
    metrics["set"] = set_type

    # Save predictions
    preds_path = os.path.join(run_dir, f"predictions_{split_name}_{set_type}.csv")
    df_preds.to_csv(preds_path, index=False)

    # Save metrics JSON
    metrics_path = os.path.join(run_dir, f"metrics_{split_name}_{set_type}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrices
    cms = compute_confusion_matrices(
        df_preds["pred_action"].values,
        df_preds["target_action"].values,
        df_preds["pred_object"].values,
        df_preds["target_object"].values,
        df_preds["pred_location"].values,
        df_preds["target_location"].values,
    )
    cm_path = os.path.join(run_dir, f"confusion_{split_name}_{set_type}.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cms, f, indent=2)

    print(f"[evaluate.py] Done! Exact-match: {metrics['exact_match_acc']:.4f} "
          f"(95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    print(f"[evaluate.py] Saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SVARA model run.")
    parser.add_argument("--run", required=True, help="Path to run directory in reports/runs/")
    parser.add_argument("--split", default="A", help="Split name (A, B1, B2, B3, C)")
    parser.add_argument("--set", default="test", choices=["valid", "test"])
    args = parser.parse_args()

    run_evaluation(run_dir=args.run, split_name=args.split, set_type=args.set)
