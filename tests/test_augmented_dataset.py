"""Unit tests for the Massive Augmented Audio Dataset (Sub-step 4.4).

Phase 4: Environmental Acoustic Augmentation
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md
"""

import os
import json
import pandas as pd
import soundfile as sf
import pytest

MANIFEST_AUGMENTED_PATH = "data/synthetic_indonesian_massive/metadata_augmented.csv"
WAV_AUGMENTED_DIR = "data/synthetic_indonesian_massive/wavs_augmented"
SUMMARY_JSON_PATH = "reports/massive_augmented_dataset_summary.json"


def test_augmented_manifest_and_disk_bijection():
    """Verify that metadata_augmented.csv and wavs_augmented/ have strict 1-to-1 bijection."""
    if not os.path.exists(MANIFEST_AUGMENTED_PATH):
        pytest.skip("metadata_augmented.csv does not exist yet (batch still running).")

    df = pd.read_csv(MANIFEST_AUGMENTED_PATH)
    total_rows = len(df)
    assert total_rows == 26970, f"Expected 26,970 rows in metadata_augmented.csv, got {total_rows}"

    assert os.path.isdir(WAV_AUGMENTED_DIR), f"WAV directory missing: {WAV_AUGMENTED_DIR}"
    disk_files = set(os.listdir(WAV_AUGMENTED_DIR))
    assert len(disk_files) == 26970, f"Expected 26,970 files on disk, got {len(disk_files)}"

    manifest_files = {os.path.basename(p) for p in df["path"]}
    missing = manifest_files - disk_files
    orphan = disk_files - manifest_files

    assert len(missing) == 0, f"Found {len(missing)} missing files from manifest"
    assert len(orphan) == 0, f"Found {len(orphan)} orphan files on disk"


def test_augmented_profile_distribution():
    """Verify exact 1/3 distribution across 3 canonical environmental acoustic profiles."""
    if not os.path.exists(MANIFEST_AUGMENTED_PATH):
        pytest.skip("metadata_augmented.csv does not exist yet.")

    df = pd.read_csv(MANIFEST_AUGMENTED_PATH)
    prof_counts = df["variation_id"].value_counts().to_dict()

    assert prof_counts.get("var1_bedroom_slow") == 8990, (
        f"Expected 8,990 var1_bedroom_slow, got {prof_counts.get('var1_bedroom_slow')}"
    )
    assert prof_counts.get("var2_living_fast") == 8990, (
        f"Expected 8,990 var2_living_fast, got {prof_counts.get('var2_living_fast')}"
    )
    assert prof_counts.get("var3_kitchen_norm") == 8990, (
        f"Expected 8,990 var3_kitchen_norm, got {prof_counts.get('var3_kitchen_norm')}"
    )


def test_augmented_speaker_and_gender_balance():
    """Verify exact 50:50 speaker and gender balance across all 26,970 clips."""
    if not os.path.exists(MANIFEST_AUGMENTED_PATH):
        pytest.skip("metadata_augmented.csv does not exist yet.")

    df = pd.read_csv(MANIFEST_AUGMENTED_PATH)
    spk_counts = df["speakerId"].value_counts().to_dict()
    assert spk_counts.get("spk_ardi") == 13485, f"Expected 13,485 spk_ardi, got {spk_counts.get('spk_ardi')}"
    assert spk_counts.get("spk_gadis") == 13485, f"Expected 13,485 spk_gadis, got {spk_counts.get('spk_gadis')}"

    gender_counts = df["gender"].value_counts().to_dict()
    assert gender_counts.get("male") == 13485
    assert gender_counts.get("female") == 13485


def test_augmented_intent_balance():
    """Verify 31 intents with exactly 870 clips each and zero class imbalance std."""
    if not os.path.exists(MANIFEST_AUGMENTED_PATH):
        pytest.skip("metadata_augmented.csv does not exist yet.")

    df = pd.read_csv(MANIFEST_AUGMENTED_PATH)
    assert df["intent_id"].nunique() == 31, f"Expected 31 intents, got {df['intent_id'].nunique()}"

    intent_counts = df["intent_id"].value_counts()
    assert (intent_counts == 870).all(), "Every intent must have exactly 870 audio clips (290 * 3)"
    assert intent_counts.std() == 0.0, "Class distribution standard deviation must be 0.0"


def test_augmented_audio_header_compliance():
    """Sample audio clips to verify 16 kHz Mono PCM_16 header compliance."""
    if not os.path.exists(MANIFEST_AUGMENTED_PATH):
        pytest.skip("metadata_augmented.csv does not exist yet.")

    df = pd.read_csv(MANIFEST_AUGMENTED_PATH)
    base_dir = os.path.dirname(MANIFEST_AUGMENTED_PATH)

    # Test 50 stratified samples across the dataset
    sample_indices = list(range(0, len(df), len(df) // 50))
    for idx in sample_indices:
        rel_path = df.iloc[idx]["path"]
        abs_path = os.path.join(base_dir, rel_path)
        assert os.path.exists(abs_path), f"File not found: {abs_path}"

        info = sf.info(abs_path)
        assert info.samplerate == 16000, f"Expected 16kHz, got {info.samplerate} in {rel_path}"
        assert info.channels == 1, f"Expected mono, got {info.channels} in {rel_path}"
        assert info.subtype == "PCM_16", f"Expected PCM_16, got {info.subtype} in {rel_path}"
        assert info.format == "WAV", f"Expected WAV container, got {info.format} in {rel_path}"
        assert info.duration >= 0.4, f"Duration abnormally short: {info.duration}s in {rel_path}"
