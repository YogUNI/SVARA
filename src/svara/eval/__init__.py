"""Evaluation module exports."""

from svara.eval.metrics import (
    compute_bootstrap_ci,
    compute_confusion_matrices,
    compute_exact_match,
    compute_metrics,
    mcnemar_test,
)
from svara.eval.reject import (
    compute_confidence_scores,
    compute_ece,
    compute_risk_coverage_curve,
    find_optimal_validation_threshold,
    fit_temperature_scaling,
)

__all__ = [
    "compute_exact_match",
    "compute_metrics",
    "compute_bootstrap_ci",
    "mcnemar_test",
    "compute_confusion_matrices",
    "compute_confidence_scores",
    "compute_ece",
    "compute_risk_coverage_curve",
    "find_optimal_validation_threshold",
    "fit_temperature_scaling",
]
