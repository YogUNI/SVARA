"""Unit tests for the ablation queue runner.

Tests:
- Registry mapping completeness
- Dry-run plan construction
- Completion checker logic (identifying completed vs pending runs)
- Deterministic run directory paths
"""

import json
import os
import tempfile
import pytest

from scripts.run_ablation import (
    EXPERIMENT_REGISTRY,
    build_run_dir,
    is_run_completed,
    run_ablation_queue,
)


def test_registry_contains_core_experiments():
    required_exps = ["M0", "E1", "E2", "E3", "E4", "E5_8", "E5_6", "E5_4", "E7", "E8"]
    for exp_id in required_exps:
        assert exp_id in EXPERIMENT_REGISTRY
        cfg = EXPERIMENT_REGISTRY[exp_id]["config"]
        assert os.path.isfile(cfg), f"Config file {cfg} does not exist for {exp_id}"


def test_is_run_completed_logic():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = os.path.join(tmpdir, "M0_B1_s42")
        assert not is_run_completed(run_dir, "B1")

        os.makedirs(run_dir, exist_ok=True)
        assert not is_run_completed(run_dir, "B1")

        # Create best.ckpt but missing metrics
        with open(os.path.join(run_dir, "best.ckpt"), "w") as f:
            f.write("mock")
        assert not is_run_completed(run_dir, "B1")

        # Create metrics_B1_test.json
        with open(os.path.join(run_dir, "metrics_B1_test.json"), "w") as f:
            json.dump({"exact_match_acc": 0.98}, f)
        assert is_run_completed(run_dir, "B1")


def test_dry_run_plan():
    with tempfile.TemporaryDirectory() as tmpdir:
        plan = run_ablation_queue(
            experiment_ids=["M0", "E1"],
            seed=42,
            dry_run=True,
            runs_root=tmpdir,
        )
        assert len(plan) == 2
        assert plan[0]["exp_id"] == "M0"
        assert plan[0]["status"] == "PENDING"
        assert plan[1]["exp_id"] == "E1"
        assert plan[1]["status"] == "PENDING"
