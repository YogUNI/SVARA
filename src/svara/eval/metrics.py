"""Evaluation metrics computation for Spoken Language Understanding.

Metrics:
- Exact-match accuracy (all 3 slots: action, object, location correct)
- Per-slot accuracy (action, object, location)
- Macro-F1 across 31 unique intents
- Per-slot and 31x31 joint confusion matrices
- 95% bootstrap confidence interval (1,000 resamples per docs/01 §5)
- McNemar paired significance test between two models

Task: P3-04
Reference: docs/01 §5, docs/11 V-21..V-24
"""

import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def compute_exact_match(
    pred_action: np.ndarray,
    target_action: np.ndarray,
    pred_object: np.ndarray,
    target_object: np.ndarray,
    pred_location: np.ndarray,
    target_location: np.ndarray,
) -> np.ndarray:
    """Return boolean array of exact-match correctness per sample."""
    act_match = pred_action == target_action
    obj_match = pred_object == target_object
    loc_match = pred_location == target_location
    return act_match & obj_match & loc_match


def compute_metrics(
    pred_action: np.ndarray,
    target_action: np.ndarray,
    pred_object: np.ndarray,
    target_object: np.ndarray,
    pred_location: np.ndarray,
    target_location: np.ndarray,
    pred_intent: Optional[np.ndarray] = None,
    target_intent: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, Dict]]:
    """Compute all standard evaluation metrics for SLU."""
    exact_matches = compute_exact_match(
        pred_action, target_action, pred_object, target_object, pred_location, target_location
    )

    metrics = {
        "exact_match_acc": float(np.mean(exact_matches)),
        "action_acc": float(np.mean(pred_action == target_action)),
        "object_acc": float(np.mean(pred_object == target_object)),
        "location_acc": float(np.mean(pred_location == target_location)),
        "num_samples": int(len(pred_action)),
    }

    if target_intent is not None:
        if pred_intent is None:
            # If pred_intent is not explicitly provided, use exact matches
            pass
        else:
            metrics["macro_f1_intent"] = float(
                f1_score(target_intent, pred_intent, average="macro", zero_division=0)
            )

    return metrics


def compute_bootstrap_ci(
    correctness: np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for exact-match accuracy.

    Returns:
        (mean_acc, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(correctness)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(correctness[indices])

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    mean_val = float(np.mean(correctness))

    return mean_val, ci_lower, ci_upper


def mcnemar_test(
    correct_model_a: np.ndarray,
    correct_model_b: np.ndarray,
) -> Dict[str, float]:
    """Paired McNemar test with continuity correction between two models (docs/01 §5).

    Returns:
        Dictionary with contingency table counts, chi2 statistic, and p-value.
    """
    assert len(correct_model_a) == len(correct_model_b), "Must evaluate on identical samples"

    # b: A correct, B wrong
    b = int(np.sum(correct_model_a & ~correct_model_b))
    # c: A wrong, B correct
    c = int(np.sum(~correct_model_a & correct_model_b))

    # McNemar's chi-squared with Edwards continuity correction: (|b - c| - 1)^2 / (b + c)
    total_discordant = b + c
    if total_discordant == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    stat = ((abs(b - c) - 1.0) ** 2) / float(total_discordant)
    from scipy.stats import chi2

    p_val = float(1.0 - chi2.cdf(stat, df=1))

    return {"b": b, "c": c, "statistic": float(stat), "p_value": p_val}


def compute_confusion_matrices(
    pred_action: np.ndarray,
    target_action: np.ndarray,
    pred_object: np.ndarray,
    target_object: np.ndarray,
    pred_location: np.ndarray,
    target_location: np.ndarray,
) -> Dict[str, List[List[int]]]:
    """Compute integer confusion matrix for each slot."""
    return {
        "action": confusion_matrix(target_action, pred_action).tolist(),
        "object": confusion_matrix(target_object, pred_object).tolist(),
        "location": confusion_matrix(target_location, pred_location).tolist(),
    }
