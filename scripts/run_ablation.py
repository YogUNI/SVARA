"""Ablation queue runner for SVARA experiments.

Executes and coordinates the full ablation and benchmark matrix (M0, E1-E10):
- Checks existing outputs (`metrics_*.json`, `best.ckpt`) to safely resume or skip finished runs.
- Supports dry-run mode to inspect the execution plan before launching heavy jobs.
- Sequential and controlled execution suitable for Google Colab GPU sessions.
- Generates reproducible run directories under `reports/runs/<run_id>`.

Tasks: P4-04
Reference: docs/02 §5, docs/10 P4-04, docs/11 V-30..V-33
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional


# Official experiment definitions mapping ID to config, split, and description
EXPERIMENT_REGISTRY: Dict[str, dict] = {
    "M0": {
        "config": "configs/model_w2v2_base.yaml",
        "split": "B1",
        "description": "Main Model: wav2vec2-base full fine-tune (CNN frozen), 3 heads, SpecAugment",
    },
    "E1": {
        "config": "configs/model_crnn.yaml",
        "split": "B1",
        "description": "CRNN from scratch (RQ2 Baseline)",
    },
    "E2": {
        "config": "configs/model_w2v2_e2_frozen.yaml",
        "split": "B1",
        "description": "wav2vec2 frozen encoder, heads only (Linear probe)",
    },
    "E3": {
        "config": "configs/model_w2v2_e3_joint.yaml",
        "split": "B1",
        "description": "wav2vec2 with single joint 31-class head (RQ3)",
    },
    "E4": {
        "config": "configs/model_w2v2_e4_960h.yaml",
        "split": "B1",
        "description": "wav2vec2-base-960h initialization (pretraining check)",
    },
    "E5_8": {
        "config": "configs/model_w2v2_e5_layers8.yaml",
        "split": "B1",
        "description": "wav2vec2 truncated to 8 transformer layers (RQ4)",
    },
    "E5_6": {
        "config": "configs/model_w2v2_e5_layers6.yaml",
        "split": "B1",
        "description": "wav2vec2 truncated to 6 transformer layers (RQ4)",
    },
    "E5_4": {
        "config": "configs/model_w2v2_e5_layers4.yaml",
        "split": "B1",
        "description": "wav2vec2 truncated to 4 transformer layers (RQ4)",
    },
    "E7": {
        "config": "configs/model_w2v2_e7_no_specaug.yaml",
        "split": "B1",
        "description": "wav2vec2 without SpecAugment",
    },
    "E8": {
        "config": "configs/model_w2v2_base.yaml",
        "split": "A",
        "description": "wav2vec2 on Split A (original standard split, RQ1)",
    },
}


def is_run_completed(run_dir: str, split: str) -> bool:
    """Check if a run directory has completed training and test evaluation."""
    if not os.path.isdir(run_dir):
        return False
    # A run is complete if test metrics exist and best.ckpt exists
    test_metrics = os.path.join(run_dir, f"metrics_{split}_test.json")
    best_ckpt = os.path.join(run_dir, "best.ckpt")
    return os.path.isfile(test_metrics) and os.path.isfile(best_ckpt)


def build_run_dir(exp_id: str, split: str, seed: int, runs_root: str = "reports/runs") -> str:
    """Create deterministic run directory path for ablation queue."""
    run_name = f"{exp_id}_{split}_s{seed}"
    return os.path.join(runs_root, run_name)


def run_ablation_queue(
    experiment_ids: List[str],
    seed: int = 42,
    split_override: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
    runs_root: str = "reports/runs",
) -> List[Dict[str, str]]:
    """Execute queue of experiments sequentially."""
    plan = []

    print(f"\n=======================================================")
    print(f" SVARA Ablation Queue Runner")
    print(f" Mode: {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f" Seed: {seed} | Overwrite/Force: {force}")
    print(f"=======================================================\n")

    for exp_id in experiment_ids:
        if exp_id not in EXPERIMENT_REGISTRY:
            print(f"[ERROR] Unknown experiment ID: '{exp_id}'. Available: {list(EXPERIMENT_REGISTRY.keys())}")
            continue

        exp_info = EXPERIMENT_REGISTRY[exp_id]
        split = split_override if split_override else exp_info["split"]
        config_path = exp_info["config"]
        run_dir = build_run_dir(exp_id, split, seed, runs_root)
        completed = is_run_completed(run_dir, split)

        status = "COMPLETED (SKIP)" if (completed and not force) else "PENDING"
        item = {
            "exp_id": exp_id,
            "description": exp_info["description"],
            "config": config_path,
            "split": split,
            "seed": str(seed),
            "run_dir": run_dir,
            "status": status,
        }
        plan.append(item)

        print(f"[{item['status']}] {exp_id}: {exp_info['description']}")
        print(f"  -> Config: {config_path}")
        print(f"  -> Split:  {split} | Seed: {seed}")
        print(f"  -> Target: {run_dir}\n")

    if dry_run:
        print("[DRY RUN COMPLETE] No processes launched.")
        return plan

    # Execute pending items
    for item in plan:
        if item["status"] == "COMPLETED (SKIP)":
            print(f"Skipping already completed experiment: {item['exp_id']} ({item['run_dir']})")
            continue

        print(f"\n=======================================================")
        print(f" LAUNCHING: {item['exp_id']} ({item['description']})")
        print(f" Run Directory: {item['run_dir']}")
        print(f"=======================================================\n")

        # 1. Run training
        train_cmd = [
            sys.executable,
            "scripts/train.py",
            "--config",
            item["config"],
            "--split",
            item["split"],
            "--seed",
            item["seed"],
            "--output-dir",
            item["run_dir"],
        ]

        print(f"[CMD] {' '.join(train_cmd)}")
        ret = subprocess.run(train_cmd)
        if ret.returncode != 0:
            print(f"[ERROR] Training failed for {item['exp_id']} with code {ret.returncode}")
            sys.exit(ret.returncode)

        # 2. Run validation evaluation
        eval_val_cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--run",
            item["run_dir"],
            "--split",
            item["split"],
            "--set",
            "valid",
        ]
        print(f"[CMD] {' '.join(eval_val_cmd)}")
        ret_val = subprocess.run(eval_val_cmd)
        if ret_val.returncode != 0:
            print(f"[ERROR] Validation evaluation failed for {item['exp_id']}")
            sys.exit(ret_val.returncode)

        # 3. Run test evaluation
        eval_test_cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--run",
            item["run_dir"],
            "--split",
            item["split"],
            "--set",
            "test",
        ]
        print(f"[CMD] {' '.join(eval_test_cmd)}")
        ret_test = subprocess.run(eval_test_cmd)
        if ret_test.returncode != 0:
            print(f"[ERROR] Test evaluation failed for {item['exp_id']}")
            sys.exit(ret_test.returncode)

        print(f"\n[SUCCESS] Completed run: {item['exp_id']} -> {item['run_dir']}\n")

    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVARA Ablation Queue Runner (P4-04)")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["M0", "E1", "E2", "E3", "E4", "E5_8", "E5_6", "E5_4", "E7", "E8"],
        help="List of experiment IDs to run (default: all primary ablations)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--split", default=None, help="Override split for all selected experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--force", action="store_true", help="Re-run even if already completed")
    parser.add_argument("--runs-root", default="reports/runs", help="Root directory for runs")
    args = parser.parse_args()

    run_ablation_queue(
        experiment_ids=args.experiments,
        seed=args.seed,
        split_override=args.split,
        dry_run=args.dry_run,
        force=args.force,
        runs_root=args.runs_root,
    )
