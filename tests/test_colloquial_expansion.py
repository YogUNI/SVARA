"""Unit tests for Indonesian Colloquial & Slang Corpus Expansion.

Task: Sub-step 13.3 (FASE 13: Real-World Speech Hardening)
Reference: docs/14_ROADMAP_PEMATANGAN_DEEP_LEARNING_REAL_WORLD.md, AGENTS.md
"""

import json
import os
import pandas as pd
import pytest

from scripts.expand_colloquial_smart_home import (
    audit_and_merge_corpus,
    generate_colloquial_dataset,
    load_legal_intent_map,
)


def test_colloquial_generation_volume_and_intents():
    """Verify >= 1,500 phrases generated across all 31 intents without missing any intent."""
    df = generate_colloquial_dataset(target_per_intent=50)
    assert len(df) >= 1500, f"Expected >= 1500 phrases, got {len(df)}"

    intents_present = set(df["intent_id"].unique())
    assert len(intents_present) == 31, f"Expected 31 intents, got {len(intents_present)}"
    assert intents_present == set(range(31)), "Missing some intent IDs"

    # Check minimum per intent
    counts = df["intent_id"].value_counts()
    assert counts.min() >= 45, f"Minimum per intent too low: {counts.min()}"


def test_colloquial_semantic_validity():
    """Verify all actions, objects, and locations strictly align with configs/intent_map.json."""
    legal_map = load_legal_intent_map()
    df = generate_colloquial_dataset(target_per_intent=10)

    for _, row in df.iterrows():
        iid = row["intent_id"]
        legal = legal_map[iid]
        assert row["action"] == legal["action"], f"Action mismatch on intent {iid}: {row['action']} vs {legal['action']}"
        assert row["object"] == legal["object"], f"Object mismatch on intent {iid}: {row['object']} vs {legal['object']}"
        assert row["location"] == legal["location"], f"Location mismatch on intent {iid}: {row['location']} vs {legal['location']}"


def test_colloquial_slang_vocabulary_presence():
    """Verify everyday casual spoken Indonesian words are heavily represented."""
    df = generate_colloquial_dataset(target_per_intent=50)
    all_text = " ".join(df["phrase"].str.lower())

    key_slang_words = [
        "nyalain",
        "matiin",
        "idupin",
        "padamin",
        "kencengin",
        "pelanin",
        "dinginin",
        "dong",
        "nih",
        "tolong",
    ]

    for word in key_slang_words:
        assert word in all_text, f"Key colloquial word '{word}' missing from generated corpus"


def test_master_corpus_integration(tmp_path):
    """Verify audit and merge creates valid master corpus and exports audit report."""
    out_colloquial = str(tmp_path / "colloquial.csv")
    out_master = str(tmp_path / "master.csv")
    report_path = str(tmp_path / "report.json")

    df_colloquial = generate_colloquial_dataset(target_per_intent=50)
    summary = audit_and_merge_corpus(
        df_colloquial=df_colloquial,
        output_colloquial_path=out_colloquial,
        output_master_path=out_master,
        report_path=report_path,
    )

    assert summary["verdict"] == "PASSED"
    assert summary["target_met"] is True
    assert summary["total_colloquial_phrases_generated"] >= 1500
    assert summary["semantic_validation"]["triplets_valid"] is True
    assert os.path.exists(out_colloquial)
    assert os.path.exists(out_master)
    assert os.path.exists(report_path)

    # Validate report contents
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["verdict"] == "PASSED"
    assert data["unique_intents_covered"] == 31
