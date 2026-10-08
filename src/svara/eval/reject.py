"""Confidence scoring, model calibration, selective classification, and rejection engine.

Core Capabilities:
1. Multi-head confidence estimation:
   - min_head_softmax: min(P(act), P(obj), P(loc))
   - product_softmax: P(act) * P(obj) * P(loc)
2. Temperature Scaling calibration (optimizes temperature T on validation set).
3. Expected Calibration Error (ECE) metric computation with reliability diagrams.
4. Optimal validation threshold selection:
   - Target selective accuracy (e.g. 98% or 99% accuracy on accepted queries).
   - Maximizing F1 between correct-accepted and incorrect-rejected.
5. Risk-Coverage curve computation (accuracy vs coverage across all possible thresholds).

Tasks: P6-05, P6-07, P6-08
Reference: docs/01 §9, docs/02 §7, docs/11 V-14 & §1
"""

import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize_scalar


def compute_confidence_scores(
    act_probs: np.ndarray,
    obj_probs: np.ndarray,
    loc_probs: np.ndarray,
    method: str = "min_head",
) -> np.ndarray:
    """Compute confidence scores given per-slot softmax probability arrays."""
    max_act = np.max(act_probs, axis=-1)
    max_obj = np.max(obj_probs, axis=-1)
    max_loc = np.max(loc_probs, axis=-1)

    if method == "min_head":
        return np.minimum(np.minimum(max_act, max_obj), max_loc)
    elif method == "product":
        return max_act * max_obj * max_loc
    elif method == "mean":
        return (max_act + max_obj + max_loc) / 3.0
    else:
        raise ValueError(f"Unsupported confidence method: {method}")


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        confidences: [N] float in [0, 1] representing predicted confidence.
        accuracies: [N] bool or int (1 for exact match correct, 0 for incorrect).
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE value (float in [0, 1]). Lower is better calibrated.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def fit_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Fit optimal temperature parameter T > 0 on validation set using NLL loss."""
    def nll_loss(temp: float) -> float:
        scaled = logits / temp
        e_x = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
        # Cross entropy loss
        correct_probs = probs[np.arange(len(labels)), labels]
        loss = -np.mean(np.log(np.clip(correct_probs, 1e-12, 1.0)))
        return loss

    res = minimize_scalar(nll_loss, bounds=(0.1, 5.0), method="bounded")
    return float(res.x)


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_thresholds: int = 50,
) -> Dict[str, List[float]]:
    """Compute Risk (error rate on accepted) vs Coverage (% accepted) across thresholds."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    coverages = []
    selective_accuracies = []
    selective_risks = []

    total_n = len(confidences)

    for th in thresholds:
        accepted_mask = confidences >= th
        n_accepted = np.sum(accepted_mask)

        if n_accepted == 0:
            cov = 0.0
            sel_acc = 1.0
            sel_risk = 0.0
        else:
            cov = float(n_accepted / total_n)
            sel_acc = float(np.mean(accuracies[accepted_mask]))
            sel_risk = 1.0 - sel_acc

        coverages.append(round(cov, 4))
        selective_accuracies.append(round(sel_acc, 4))
        selective_risks.append(round(sel_risk, 4))

    return {
        "thresholds": [round(float(t), 4) for t in thresholds],
        "coverage": coverages,
        "selective_accuracy": selective_accuracies,
        "selective_risk": selective_risks,
    }


def find_optimal_validation_threshold(
    val_confidences: np.ndarray,
    val_accuracies: np.ndarray,
    target_accuracy: float = 0.98,
    min_coverage: float = 0.50,
) -> Tuple[float, Dict[str, float]]:
    """Determine optimal rejection threshold exclusively from validation set (RULE: docs/11 §1).

    Selects the lowest threshold that guarantees selective accuracy >= target_accuracy
    while maintaining at least min_coverage. If target cannot be reached, selects threshold
    maximizing harmonic mean of coverage and selective accuracy.
    """
    thresholds = np.sort(np.unique(val_confidences))
    best_threshold = 0.85
    best_metrics = {"coverage": 0.0, "selective_accuracy": 0.0}

    # Evaluate all candidate thresholds
    candidates = []
    for th in thresholds:
        accepted = val_confidences >= th
        n_acc = np.sum(accepted)
        if n_acc == 0:
            continue
        cov = n_acc / len(val_confidences)
        acc = np.mean(val_accuracies[accepted])
        candidates.append((th, cov, acc))

    # Priority 1: Satisfies target_accuracy with highest coverage
    valid_candidates = [c for c in candidates if c[2] >= target_accuracy and c[1] >= min_coverage]
    if valid_candidates:
        # Choose candidate with maximum coverage (lowest required threshold)
        best_candidate = max(valid_candidates, key=lambda c: c[1])
        best_threshold = float(best_candidate[0])
        best_metrics = {"coverage": float(best_candidate[1]), "selective_accuracy": float(best_candidate[2])}
    elif candidates:
        # Priority 2: Maximize F1 between coverage and accuracy
        best_candidate = max(candidates, key=lambda c: 2 * (c[1] * c[2]) / (c[1] + c[2] + 1e-9))
        best_threshold = float(best_candidate[0])
        best_metrics = {"coverage": float(best_candidate[1]), "selective_accuracy": float(best_candidate[2])}

    return best_threshold, best_metrics
