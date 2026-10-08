"""Unit tests for confidence scoring, ECE, temperature scaling, and threshold selection (Task P6-05)."""

import numpy as np
import pytest

from svara.eval.reject import (
    compute_confidence_scores,
    compute_ece,
    compute_risk_coverage_curve,
    find_optimal_validation_threshold,
    fit_temperature_scaling,
)


def test_compute_confidence_scores():
    # 2 samples
    act_p = np.array([[0.8, 0.2], [0.95, 0.05]])
    obj_p = np.array([[0.7, 0.3], [0.90, 0.10]])
    loc_p = np.array([[0.9, 0.1], [0.85, 0.15]])

    # min_head: sample 0 min(0.8, 0.7, 0.9) = 0.7; sample 1 min(0.95, 0.9, 0.85) = 0.85
    min_scores = compute_confidence_scores(act_p, obj_p, loc_p, method="min_head")
    np.testing.assert_allclose(min_scores, [0.7, 0.85], rtol=1e-5)

    # product: sample 0 (0.8 * 0.7 * 0.9) = 0.504
    prod_scores = compute_confidence_scores(act_p, obj_p, loc_p, method="product")
    np.testing.assert_allclose(prod_scores[0], 0.504, rtol=1e-4)


def test_compute_ece():
    # Perfect calibration: confidence 0.8 with 80% accuracy, confidence 0.2 with 20% accuracy
    conf = np.array([0.8] * 100 + [0.2] * 100)
    acc = np.array([1] * 80 + [0] * 20 + [1] * 20 + [0] * 80)

    ece = compute_ece(conf, acc, n_bins=10)
    assert ece < 0.05, f"ECE should be close to 0 for perfectly calibrated data, got {ece}"


def test_temperature_scaling_optimization():
    np.random.seed(42)
    # Overconfident uncalibrated logits
    logits = np.random.randn(100, 4) * 5.0
    labels = np.random.randint(0, 4, size=100)

    temp = fit_temperature_scaling(logits, labels)
    assert 0.1 < temp < 5.0, f"Optimized temperature {temp} should be in reasonable bound"


def test_find_optimal_validation_threshold():
    # Samples with varying confidences and correctness
    confidences = np.array([0.95, 0.92, 0.88, 0.85, 0.70, 0.60, 0.50, 0.40])
    # The high confidence samples are all correct, low are wrong
    accuracies = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    th, metrics = find_optimal_validation_threshold(
        confidences, accuracies, target_accuracy=1.0, min_coverage=0.5
    )
    # Threshold should accept top 4 samples (th around 0.85)
    assert th <= 0.85
    assert metrics["selective_accuracy"] == 1.0
    assert metrics["coverage"] >= 0.5


def test_risk_coverage_curve():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    acc = np.array([1, 1, 1, 0])

    curve = compute_risk_coverage_curve(conf, acc, num_thresholds=10)
    assert "coverage" in curve
    assert "selective_accuracy" in curve
    assert "selective_risk" in curve
    assert len(curve["coverage"]) == 10
