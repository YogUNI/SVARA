"""Unit tests for the Balanced Bilingual Dataset Integration (Sub-step 5.1).

Phase 5: Balanced Bilingual Dataset Integration & Splitting
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md, docs/01_DATASET_AND_EVALUATION.md
"""

import os
import json
import pytest
import pandas as pd
import soundfile as sf

BILINGUAL_MANIFEST = "data/bilingual_massive/bilingual_manifest_24k.csv"
SUMMARY_REPORT = "reports/bilingual_dataset_integration_summary.json"
INTENT_MAP_PATH = "configs/intent_map.json"

EXPECTED_TOTAL_ROWS = 24000
EXPECTED_PER_LANG = 12000


def test_bilingual_manifest_row_count_and_columns():
    """Verify that bilingual_manifest_24k.csv exists, has 24,000 rows, and valid columns."""
    assert os.path.exists(BILINGUAL_MANIFEST), f"Manifest missing: {BILINGUAL_MANIFEST}"
    df = pd.read_csv(BILINGUAL_MANIFEST)
    assert len(df) == EXPECTED_TOTAL_ROWS, f"Expected {EXPECTED_TOTAL_ROWS} rows, got {len(df)}"

    expected_cols = [
        "audio_path", "speaker_id", "gender", "language", "transcript",
        "action", "object", "location", "intent_id", "domain", "duration",
        "acoustic_profile", "source_split_original"
    ]
    assert list(df.columns) == expected_cols, f"Column mismatch: {df.columns}"
    assert df.isnull().sum().sum() == 0, "Found NULL values in bilingual manifest"


def test_bilingual_language_balance_50_50():
    """Verify exact 50:50 language parity (12,000 Indonesian vs 12,000 English)."""
    df = pd.read_csv(BILINGUAL_MANIFEST)
    lang_counts = df["language"].value_counts().to_dict()

    assert lang_counts.get("id") == EXPECTED_PER_LANG, (
        f"Expected {EXPECTED_PER_LANG} Indonesian clips, got {lang_counts.get('id')}"
    )
    assert lang_counts.get("en") == EXPECTED_PER_LANG, (
        f"Expected {EXPECTED_PER_LANG} English clips, got {lang_counts.get('en')}"
    )


def test_bilingual_intent_balance_and_zero_skew():
    """Verify all 31 intents exist and have identical quotas between Indonesian and English."""
    df = pd.read_csv(BILINGUAL_MANIFEST)
    assert df["intent_id"].nunique() == 31, f"Expected 31 intents, got {df['intent_id'].nunique()}"

    intent_lang_table = df.groupby(["intent_id", "language"]).size().unstack(fill_value=0)
    for intent_id in range(31):
        assert intent_id in intent_lang_table.index, f"Intent {intent_id} missing from dataset"
        count_id = intent_lang_table.loc[intent_id, "id"]
        count_en = intent_lang_table.loc[intent_id, "en"]
        assert count_id == count_en, (
            f"Intent {intent_id} has language skew: ID={count_id} vs EN={count_en}"
        )
        assert count_id in [387, 388], f"Intent {intent_id} quota out of range: {count_id}"


def test_bilingual_indonesian_profiles_and_gender():
    """Verify Indonesian subset has 50:50 gender balance and balanced acoustic profiles."""
    df = pd.read_csv(BILINGUAL_MANIFEST)
    df_id = df[df["language"] == "id"]

    gender_counts = df_id["gender"].value_counts().to_dict()
    assert gender_counts.get("male") == 6000, f"Expected 6,000 male ID, got {gender_counts.get('male')}"
    assert gender_counts.get("female") == 6000, f"Expected 6,000 female ID, got {gender_counts.get('female')}"

    prof_counts = df_id["acoustic_profile"].value_counts().to_dict()
    assert len(prof_counts) == 3, f"Expected 3 acoustic profiles, got {len(prof_counts)}"
    # Each profile should be close to 4,000
    for prof, cnt in prof_counts.items():
        assert 3950 <= cnt <= 4050, f"Profile {prof} has abnormal count: {cnt}"


def test_bilingual_slot_alignment_with_intent_map():
    """Verify that action, object, location slots perfectly match configs/intent_map.json."""
    assert os.path.exists(INTENT_MAP_PATH)
    with open(INTENT_MAP_PATH, "r", encoding="utf-8") as f:
        raw_imap = json.load(f)
    imap = raw_imap.get("intent_map", raw_imap)

    df = pd.read_csv(BILINGUAL_MANIFEST)
    for _, r in df.iterrows():
        iid = str(r["intent_id"])
        expected = imap[iid]
        assert r["action"] == expected["action"], f"Action mismatch at row: {r['transcript']}"
        assert r["object"] == expected["object"], f"Object mismatch at row: {r['transcript']}"
        assert r["location"] == expected["location"], f"Location mismatch at row: {r['transcript']}"


def test_bilingual_audio_paths_exist_on_disk():
    """Sample audio clips across both languages to verify physical existence and 16 kHz format."""
    df = pd.read_csv(BILINGUAL_MANIFEST)
    sample_indices = list(range(0, len(df), len(df) // 100))

    for idx in sample_indices:
        rel_path = df.iloc[idx]["audio_path"]
        assert os.path.exists(rel_path), f"Audio file not found on disk: {rel_path}"

        info = sf.info(rel_path)
        assert info.samplerate == 16000, f"Expected 16kHz, got {info.samplerate} in {rel_path}"
        assert info.channels == 1, f"Expected mono (1 channel), got {info.channels} in {rel_path}"
        assert info.duration >= 0.3, f"Abnormally short duration: {info.duration}s in {rel_path}"


def test_bilingual_integration_summary_report():
    """Verify that bilingual_dataset_integration_summary.json exists and passes."""
    assert os.path.exists(SUMMARY_REPORT), f"Summary report missing: {SUMMARY_REPORT}"
    with open(SUMMARY_REPORT, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary.get("status") == "PASS", "Summary status must be PASS"
    assert summary["total_clips"] == EXPECTED_TOTAL_ROWS
    assert summary["intent_skew_count"] == 0
    assert summary["language_distribution"]["id"] == EXPECTED_PER_LANG
    assert summary["language_distribution"]["en"] == EXPECTED_PER_LANG
