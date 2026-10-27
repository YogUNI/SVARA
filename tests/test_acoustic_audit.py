"""Unit tests for the Synthetic Audio Dataset and Acoustic Audit.

Task: Sub-step 3.3 (Phase 3: Multi-Speaker Voice Synthesis)
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md
"""

import os
import json
import pandas as pd
import soundfile as sf
import pytest

METADATA_PATH = "data/synthetic_indonesian_massive/metadata.csv"
WAV_DIR = "data/synthetic_indonesian_massive/wavs_16k"
AUDIT_JSON_PATH = "reports/synthetic_acoustic_audit.json"
AUDIT_MD_PATH = "reports/synthetic_acoustic_audit.md"


def test_synthetic_manifest_and_disk_bijection():
    """Verify that metadata manifest and disk WAV files have a strict 1-to-1 bijection."""
    assert os.path.exists(METADATA_PATH), f"Metadata manifest missing: {METADATA_PATH}"
    assert os.path.isdir(WAV_DIR), f"WAV directory missing: {WAV_DIR}"

    df = pd.read_csv(METADATA_PATH)
    assert len(df) == 8990, f"Expected exactly 8,990 rows in metadata, got {len(df)}"

    disk_files = set(os.listdir(WAV_DIR))
    assert len(disk_files) == 8990, f"Expected 8,990 WAV files on disk, got {len(disk_files)}"

    manifest_files = {os.path.basename(p) for p in df["path"]}
    missing = manifest_files - disk_files
    orphan = disk_files - manifest_files

    assert len(missing) == 0, f"Found {len(missing)} missing files from manifest"
    assert len(orphan) == 0, f"Found {len(orphan)} orphan files on disk"


def test_synthetic_audio_header_compliance():
    """Verify that synthesized WAV files strictly follow 16 kHz, 1-channel mono, PCM_16 format."""
    df = pd.read_csv(METADATA_PATH)
    base_dir = os.path.dirname(METADATA_PATH)

    # Test a representative sample across the entire dataset (start, middle, end)
    sample_indices = list(range(0, len(df), 150)) + [len(df) - 1]
    for idx in sample_indices:
        rel_path = df.iloc[idx]["path"]
        abs_path = os.path.join(base_dir, rel_path)
        assert os.path.exists(abs_path), f"File not found: {abs_path}"

        info = sf.info(abs_path)
        assert info.samplerate == 16000, f"Expected 16kHz, got {info.samplerate} in {rel_path}"
        assert info.channels == 1, f"Expected mono (1 channel), got {info.channels} in {rel_path}"
        assert info.subtype == "PCM_16", f"Expected PCM_16 bit depth, got {info.subtype} in {rel_path}"
        assert info.format == "WAV", f"Expected WAV container, got {info.format} in {rel_path}"
        assert info.duration >= 0.5, f"Duration abnormally short: {info.duration}s in {rel_path}"
        assert abs(info.duration - float(df.iloc[idx]["duration"])) < 0.05, "Duration mismatch with metadata"


def test_synthetic_speaker_and_gender_balance():
    """Verify exact 50:50 gender and speaker distribution across 8,990 files."""
    df = pd.read_csv(METADATA_PATH)
    spk_counts = df["speakerId"].value_counts().to_dict()
    assert spk_counts.get("spk_ardi") == 4495, f"Expected 4,495 spk_ardi, got {spk_counts.get('spk_ardi')}"
    assert spk_counts.get("spk_gadis") == 4495, f"Expected 4,495 spk_gadis, got {spk_counts.get('spk_gadis')}"

    gender_counts = df["gender"].value_counts().to_dict()
    assert gender_counts.get("male") == 4495
    assert gender_counts.get("female") == 4495


def test_synthetic_intent_and_entropy_balance():
    """Verify 31 intents with exactly 290 clips each and maximum Shannon entropy ratio."""
    df = pd.read_csv(METADATA_PATH)
    assert df["intent_id"].nunique() == 31, f"Expected 31 intents, got {df['intent_id'].nunique()}"

    intent_counts = df["intent_id"].value_counts()
    assert (intent_counts == 290).all(), "Every intent must have exactly 290 audio clips"
    assert intent_counts.std() == 0.0, "Class distribution standard deviation must be zero"


def test_synthetic_audit_reports_exist_and_passed():
    """Verify that audit JSON and Markdown reports exist and indicate PASS status."""
    assert os.path.exists(AUDIT_JSON_PATH), f"Audit JSON missing: {AUDIT_JSON_PATH}"
    assert os.path.exists(AUDIT_MD_PATH), f"Audit Markdown missing: {AUDIT_MD_PATH}"

    with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("global_status") == "PASS", "Audit global status must be PASS"
    assert data["bijection_audit"]["status"] == "PASS"
    assert data["acoustic_header_audit"]["status"] == "PASS"
    assert data["signal_hygiene_audit"]["status"] == "PASS"
    assert data["signal_hygiene_audit"]["nan_count"] == 0
    assert data["signal_hygiene_audit"]["inf_count"] == 0
    assert data["signal_hygiene_audit"]["peak_violations"] == 0
    assert data["speaker_audit"]["status"] == "PASS"
    assert data["intent_audit"]["status"] == "PASS"
    assert data["intent_audit"]["entropy_ratio"] == 1.0
