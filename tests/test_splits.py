"""Unit tests for split generation and leakage assertions.

Task: P2-04
Reference: docs/01 §4, docs/11 V-08
"""

import os

import pandas as pd
import pytest

from svara.data.splits import (
    create_split_b,
    create_split_c,
)


@pytest.fixture
def mock_dataset_for_splits():
    """Create a mock dataframe covering multiple intents and phrases."""
    intent_mapping = {
        ("activate", "lights", "kitchen"): 0,
        ("deactivate", "lights", "kitchen"): 1,
        ("increase", "volume", "none"): 2,
    }

    records = []
    # Create 4 phrases for each intent across 6 speakers
    for (act, obj, loc), intent_id in intent_mapping.items():
        for phrase_idx in range(4):
            phrase = f"{act} {obj} {loc} variant {phrase_idx}"
            for spk_idx in range(6):
                records.append({
                    "path": f"wavs/spk_{spk_idx}/clip_{intent_id}_{phrase_idx}.wav",
                    "speakerId": f"spk_{spk_idx}",
                    "transcript": phrase,
                    "action": act,
                    "object": obj,
                    "location": loc,
                })

    df = pd.DataFrame(records)
    demo_df = pd.DataFrame({
        "speakerId": [f"spk_{i}" for i in range(6)],
        "gender": ["F", "M", "F", "M", "F", "M"],
    })
    return df, demo_df, intent_mapping


def test_split_b_unseen_transcript_leakage(mock_dataset_for_splits):
    df, _, intent_mapping = mock_dataset_for_splits
    tr, val, te, info = create_split_b(df, intent_mapping, seed=42, test_ratio=0.25, valid_ratio=0.25, min_train_phrasings=2)

    tr_phrases = set(tr["transcript"])
    val_phrases = set(val["transcript"])
    te_phrases = set(te["transcript"])

    # Strict Zero-Leakage assertions
    assert len(tr_phrases & te_phrases) == 0, "Leakage detected between train and test phrases!"
    assert len(tr_phrases & val_phrases) == 0, "Leakage detected between train and val phrases!"
    assert len(val_phrases & te_phrases) == 0, "Leakage detected between val and test phrases!"

    # Ensure every intent has >= 2 training phrasings
    for _key, intent_id in intent_mapping.items():
        sub = tr[tr["intent_id"] == intent_id]
        assert sub["transcript"].nunique() >= 2


def test_split_c_speaker_disjoint_leakage(mock_dataset_for_splits):
    df, demo_df, intent_mapping = mock_dataset_for_splits
    tr, val, te, info = create_split_c(df, demo_df, intent_mapping, seed=42, train_ratio=0.6, valid_ratio=0.2)

    tr_spk = set(tr["speakerId"])
    val_spk = set(val["speakerId"])
    te_spk = set(te["speakerId"])

    assert len(tr_spk & te_spk) == 0, "Speaker leakage train and test!"
    assert len(tr_spk & val_spk) == 0, "Speaker leakage train and val!"
    assert len(val_spk & te_spk) == 0, "Speaker leakage val and test!"


def test_generated_splits_on_disk():
    """Verify that processed split files exist on disk and obey leakage constraints."""
    base_dir = "data/processed/splits"
    if not os.path.exists(base_dir):
        pytest.skip("Processed splits not generated yet.")

    for split_name in ["A", "B1", "B2", "B3", "C"]:
        s_dir = os.path.join(base_dir, split_name)
        assert os.path.exists(os.path.join(s_dir, "train.csv"))
        assert os.path.exists(os.path.join(s_dir, "valid.csv"))
        assert os.path.exists(os.path.join(s_dir, "test.csv"))

        tr = pd.read_csv(os.path.join(s_dir, "train.csv"))
        te = pd.read_csv(os.path.join(s_dir, "test.csv"))
        val = pd.read_csv(os.path.join(s_dir, "valid.csv"))

        # Verify all 31 intents exist in train
        assert tr["intent_id"].nunique() == 31, f"Split {split_name} train missing intents!"

        # Specific assertions for Split B (Unseen)
        if split_name.startswith("B"):
            assert len(set(tr["transcript"]) & set(te["transcript"])) == 0
            assert len(set(tr["transcript"]) & set(val["transcript"])) == 0

        # Specific assertions for Split C (Speaker disjoint)
        if split_name == "C":
            assert len(set(tr["speakerId"]) & set(te["speakerId"])) == 0
            assert len(set(tr["speakerId"]) & set(val["speakerId"])) == 0
