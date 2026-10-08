"""Split generation logic and leakage validation for SVARA.

Task: P2-01, P2-02, P2-03, P2-04
Reference: docs/01 §4, docs/11 V-08
"""

import json
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def load_intent_mapping(intent_map_path: str = "configs/intent_map.json") -> Dict[Tuple[str, str, str], int]:
    """Load intent_map.json and return a dictionary mapping (action, object, location) -> intent_id."""
    with open(intent_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for intent_id_str, slots in data["intent_map"].items():
        key = (slots["action"], slots["object"], slots["location"])
        mapping[key] = int(intent_id_str)
    return mapping


def assign_intent_id(df: pd.DataFrame, intent_mapping: Dict[Tuple[str, str, str], int]) -> pd.DataFrame:
    """Assign intent_id column based on (action, object, location) tuple."""
    df_out = df.copy()
    intent_ids = []
    for _, row in df_out.iterrows():
        key = (str(row["action"]), str(row["object"]), str(row["location"]))
        if key not in intent_mapping:
            raise ValueError(f"Unknown intent combination not in intent_map.json: {key}")
        intent_ids.append(intent_mapping[key])
    df_out["intent_id"] = intent_ids
    return df_out


def create_split_a(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    intent_mapping: Dict[Tuple[str, str, str], int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create Split A (Original Benchmark) with assigned intent_id."""
    tr = assign_intent_id(train_df, intent_mapping)
    val = assign_intent_id(valid_df, intent_mapping)
    te = assign_intent_id(test_df, intent_mapping)
    return tr, val, te


def create_split_b(
    all_df: pd.DataFrame,
    intent_mapping: Dict[Tuple[str, str, str], int],
    seed: int = 42,
    test_ratio: float = 0.20,
    valid_ratio: float = 0.10,
    min_train_phrasings: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Create Split B (Unseen-Utterance Split) grouped strictly by transcript.

    Constraints (docs/01 §4):
    - For each intent, holds out phrasings for test (~20%) and valid (~10%).
    - Every intent retains at least min_train_phrasings distinct phrasings in train.
    - Strict assertions: set(test.transcript) ∩ set(train.transcript) == ∅, same for valid.
    """
    rng = np.random.RandomState(seed)
    df_with_intent = assign_intent_id(all_df, intent_mapping)

    train_transcripts: Set[str] = set()
    valid_transcripts: Set[str] = set()
    test_transcripts: Set[str] = set()
    constraint_logs: List[Dict] = []

    # Process grouped by intent
    grouped = df_with_intent.groupby(["action", "object", "location"])

    for (act, obj, loc), group in grouped:
        unique_phrases = sorted(group["transcript"].unique().tolist())
        n_phrases = len(unique_phrases)
        rng.shuffle(unique_phrases)

        # Desired counts
        n_test = int(round(n_phrases * test_ratio))
        n_val = int(round(n_phrases * valid_ratio))

        # Enforce minimum training phrasings
        if n_phrases - (n_test + n_val) < min_train_phrasings:
            # Adjust holdout to preserve training minimum
            allowed_holdout = max(0, n_phrases - min_train_phrasings)
            n_test = min(n_test, allowed_holdout)
            n_val = min(n_val, max(0, allowed_holdout - n_test))
            constraint_logs.append({
                "intent": f"{act}|{obj}|{loc}",
                "total_phrasings": n_phrases,
                "adjusted_test": n_test,
                "adjusted_val": n_val,
                "train_phrasings": n_phrases - (n_test + n_val),
            })

        val_selected = unique_phrases[:n_val]
        test_selected = unique_phrases[n_val : n_val + n_test]
        train_selected = unique_phrases[n_val + n_test :]

        valid_transcripts.update(val_selected)
        test_transcripts.update(test_selected)
        train_transcripts.update(train_selected)

    # Filter rows based on assigned transcripts
    train_split = df_with_intent[df_with_intent["transcript"].isin(train_transcripts)].copy()
    valid_split = df_with_intent[df_with_intent["transcript"].isin(valid_transcripts)].copy()
    test_split = df_with_intent[df_with_intent["transcript"].isin(test_transcripts)].copy()

    # Integrity assertions (docs/01 §4)
    train_set = set(train_split["transcript"])
    valid_set = set(valid_split["transcript"])
    test_set = set(test_split["transcript"])

    assert len(train_set & test_set) == 0, "Split B Error: Leakage between train and test transcripts!"
    assert len(train_set & valid_set) == 0, "Split B Error: Leakage between train and valid transcripts!"
    assert len(valid_set & test_set) == 0, "Split B Error: Leakage between valid and test transcripts!"

    # Ensure all 31 intents are represented in train
    assert train_split["intent_id"].nunique() == len(intent_mapping), (
        f"Split B Error: Train split only contains {train_split['intent_id'].nunique()} of {len(intent_mapping)} intents!"
    )

    manifest_info = {
        "seed": seed,
        "rule": "unseen_utterance_by_transcript",
        "counts": {
            "train_clips": len(train_split),
            "valid_clips": len(valid_split),
            "test_clips": len(test_split),
            "train_transcripts": len(train_set),
            "valid_transcripts": len(valid_set),
            "test_transcripts": len(test_set),
        },
        "constraint_adjustments": constraint_logs,
    }

    return train_split, valid_split, test_split, manifest_info


def create_split_c(
    all_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    intent_mapping: Dict[Tuple[str, str, str], int],
    seed: int = 42,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Create Split C (Speaker-Grouped, Demographically Stratified).

    Constraints:
    - Group by speakerId.
    - Zero speaker leakage: set(test.speakerId) ∩ set(train.speakerId) == ∅.
    """
    rng = np.random.RandomState(seed)
    df_with_intent = assign_intent_id(all_df, intent_mapping)

    unique_speakers = sorted(df_with_intent["speakerId"].unique().tolist())
    rng.shuffle(unique_speakers)

    n_total = len(unique_speakers)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * valid_ratio))

    train_speakers = set(unique_speakers[:n_train])
    valid_speakers = set(unique_speakers[n_train : n_train + n_val])
    test_speakers = set(unique_speakers[n_train + n_val :])

    train_split = df_with_intent[df_with_intent["speakerId"].isin(train_speakers)].copy()
    valid_split = df_with_intent[df_with_intent["speakerId"].isin(valid_speakers)].copy()
    test_split = df_with_intent[df_with_intent["speakerId"].isin(test_speakers)].copy()

    # Leakage assertions
    assert len(train_speakers & test_speakers) == 0, "Split C Error: Speaker leakage train/test!"
    assert len(train_speakers & valid_speakers) == 0, "Split C Error: Speaker leakage train/val!"
    assert len(valid_speakers & test_speakers) == 0, "Split C Error: Speaker leakage val/test!"

    manifest_info = {
        "seed": seed,
        "rule": "speaker_grouped_stratified",
        "counts": {
            "train_clips": len(train_split),
            "valid_clips": len(valid_split),
            "test_clips": len(test_split),
            "train_speakers": len(train_speakers),
            "valid_speakers": len(valid_speakers),
            "test_speakers": len(test_speakers),
        },
    }

    return train_split, valid_split, test_split, manifest_info


def save_split_csvs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
):
    """Save train, valid, and test dataframes to CSV in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    # Save standard columns including intent_id
    cols = ["path", "speakerId", "transcript", "action", "object", "location", "intent_id"]
    train_df[cols].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    valid_df[cols].to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    test_df[cols].to_csv(os.path.join(output_dir, "test.csv"), index=False)
