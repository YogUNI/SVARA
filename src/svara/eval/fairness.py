"""Fairness and demographic bias evaluation module.

Evaluates exact-match accuracy disparities across:
- Gender (male, female)
- Age group ranges
- First language / accents
Computes 95% bootstrap confidence intervals per subgroup and flags groups with n < 30.

Task: P6-04
Reference: docs/01 §7, docs/10 P6-04
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from svara.eval.metrics import compute_bootstrap_ci


def evaluate_demographic_fairness(
    predictions_df: pd.DataFrame,
    demographics_csv_path: str = "data/raw/fluent_speech_commands_dataset/data/speaker_demographics.csv",
    min_subgroup_size: int = 30,
) -> Dict[str, pd.DataFrame]:
    """Join predictions with speaker demographics and compute per-group metrics."""
    if not os.path.exists(demographics_csv_path):
        raise FileNotFoundError(f"Demographics file not found at {demographics_csv_path}")

    demo_df = pd.read_csv(demographics_csv_path)
    merged = predictions_df.merge(demo_df, on="speakerId", how="left")

    results = {}
    demographic_keys = [col for col in ["gender", "age_range", "firstLanguage"] if col in merged.columns]

    for key in demographic_keys:
        group_rows = []
        for val, sub in merged.groupby(key):
            n_samples = len(sub)
            corrects = sub["correct_exact"].values.astype(bool)

            mean_acc, ci_low, ci_high = compute_bootstrap_ci(corrects, n_resamples=1000)

            group_rows.append({
                "subgroup": str(val),
                "count": n_samples,
                "exact_match_acc": round(mean_acc, 4),
                "ci95_lower": round(ci_low, 4),
                "ci95_upper": round(ci_high, 4),
                "low_confidence_flag": bool(n_samples < min_subgroup_size),
            })

        results[key] = pd.DataFrame(group_rows)

    return results
