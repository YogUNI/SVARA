"""Unit tests for evaluation metrics and statistical tests.

Tasks: P3-04
Reference: docs/01 §5
"""

import numpy as np
import pytest

from svara.eval.metrics import (
    compute_bootstrap_ci,
    compute_confusion_matrices,
    compute_exact_match,
    compute_metrics,
    mcnemar_test,
)


def test_compute_exact_match():
    pred_act = np.array([0, 1, 2, 0])
    true_act = np.array([0, 1, 2, 0])

    pred_obj = np.array([0, 1, 2, 1])
    true_obj = np.array([0, 1, 2, 0])  # sample 3 wrong

    pred_loc = np.array([0, 1, 0, 0])
    true_loc = np.array([0, 1, 1, 0])  # sample 2 wrong

    matches = compute_exact_match(pred_act, true_act, pred_obj, true_obj, pred_loc, true_loc)
    expected = np.array([True, True, False, False])
    np.testing.assert_array_equal(matches, expected)


def test_compute_metrics():
    pred_act = np.array([0, 1, 0, 1])
    true_act = np.array([0, 1, 0, 1])  # 4/4 = 1.0

    pred_obj = np.array([0, 1, 0, 0])
    true_obj = np.array([0, 1, 0, 1])  # 3/4 = 0.75

    pred_loc = np.array([0, 0, 0, 0])
    true_loc = np.array([0, 0, 0, 0])  # 4/4 = 1.0

    res = compute_metrics(pred_act, true_act, pred_obj, true_obj, pred_loc, true_loc)
    assert res["exact_match_acc"] == 0.75
    assert res["action_acc"] == 1.0
    assert res["object_acc"] == 0.75
    assert res["location_acc"] == 1.0
    assert res["num_samples"] == 4


def test_compute_bootstrap_ci():
    # 100 samples with 80% accuracy
    correctness = np.array([1] * 80 + [0] * 20, dtype=bool)
    mean_val, ci_lower, ci_upper = compute_bootstrap_ci(correctness, n_resamples=500, seed=42)

    assert mean_val == 0.8
    assert 0.70 < ci_lower < 0.80
    assert 0.80 < ci_upper < 0.90
    assert ci_lower < mean_val < ci_upper


def test_mcnemar_test():
    # Model A: [1, 1, 1, 0, 0]
    # Model B: [1, 0, 0, 1, 0]
    # b = A=1, B=0 -> 2 (indices 1, 2)
    # c = A=0, B=1 -> 1 (index 3)
    a = np.array([True, True, True, False, False])
    b = np.array([True, False, False, True, False])

    res = mcnemar_test(a, b)
    assert res["b"] == 2
    assert res["c"] == 1
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0


def test_confusion_matrices():
    pred_act = np.array([0, 1, 0])
    true_act = np.array([0, 0, 0])
    pred_obj = np.array([1, 1, 2])
    true_obj = np.array([1, 1, 2])
    pred_loc = np.array([0, 0, 0])
    true_loc = np.array([0, 0, 0])

    cm = compute_confusion_matrices(pred_act, true_act, pred_obj, true_obj, pred_loc, true_loc)
    assert "action" in cm
    assert "object" in cm
    assert "location" in cm
