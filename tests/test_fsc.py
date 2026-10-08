"""Unit tests for FSC data loader using synthetic fixtures.

Task: P1-07
Reference: docs/01 §2, docs/10 P1-07
"""

import os
import wave

import pandas as pd
import pytest

from svara.data.fsc import (
    FSCPaths,
    analyze_split_overlaps,
    extract_intent_vocabularies,
    load_fsc_splits,
    read_fsc_demographics_csv,
    read_fsc_split_csv,
)


@pytest.fixture
def synthetic_fsc_dataset(tmp_path):
    """Create a minimal synthetic FSC dataset structure with dummy wav files."""
    dataset_dir = tmp_path / "mock_fsc"
    data_dir = dataset_dir / "data"
    wavs_dir = dataset_dir / "wavs" / "speakers" / "spk_01"
    data_dir.mkdir(parents=True)
    wavs_dir.mkdir(parents=True)

    # Dummy wav file (16 kHz mono, 0.1s silence)
    wav_path = wavs_dir / "dummy_01.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)

    # Relative path from dataset root
    rel_wav_path = "wavs/speakers/spk_01/dummy_01.wav"

    train_data = {
        "": [0, 1],
        "path": [rel_wav_path, rel_wav_path],
        "speakerId": ["spk_01", "spk_01"],
        "transcription": ["turn on the lights", "switch off the music"],
        "action": ["activate", "deactivate"],
        "object": ["lights", "music"],
        "location": ["kitchen", "none"],
    }
    pd.DataFrame(train_data).to_csv(data_dir / "train_data.csv", index=False)

    valid_data = {
        "": [0],
        "path": [rel_wav_path],
        "speakerId": ["spk_02"],
        "transcription": ["turn on the lights"],
        "action": ["activate"],
        "object": ["lights"],
        "location": ["kitchen"],
    }
    pd.DataFrame(valid_data).to_csv(data_dir / "valid_data.csv", index=False)

    test_data = {
        "": [0],
        "path": [rel_wav_path],
        "speakerId": ["spk_03"],
        "transcription": ["turn on the lights"],
        "action": ["activate"],
        "object": ["lights"],
        "location": ["kitchen"],
    }
    pd.DataFrame(test_data).to_csv(data_dir / "test_data.csv", index=False)

    demo_data = {
        "speakerId": ["spk_01", "spk_02", "spk_03"],
        "Self-reported fluency level ": ["native", "native", "non-native"],
        "First Language spoken": ["English", "English", "Indonesian"],
        "Current language used for work/school": ["English", "English", "English"],
        "gender": ["female", "male", "female"],
        "ageRange": ["22-40", "22-40", "41-65"],
    }
    pd.DataFrame(demo_data).to_csv(data_dir / "speaker_demographics.csv", index=False)

    (dataset_dir / "Fluent Speech Commands Public License.pdf").write_text("Dummy License")
    (dataset_dir / "readme.md").write_text("Dummy Readme")

    return dataset_dir


def test_fsc_paths_resolution(synthetic_fsc_dataset):
    paths = FSCPaths.from_root(str(synthetic_fsc_dataset))
    assert os.path.exists(paths.train_csv)
    assert os.path.exists(paths.valid_csv)
    assert os.path.exists(paths.test_csv)
    assert os.path.exists(paths.demographics_csv)
    assert os.path.exists(paths.license_pdf)


def test_read_fsc_split_csv(synthetic_fsc_dataset):
    paths = FSCPaths.from_root(str(synthetic_fsc_dataset))
    df = read_fsc_split_csv(paths.train_csv, dataset_root=str(synthetic_fsc_dataset))
    assert len(df) == 2
    assert "transcript" in df.columns
    assert "transcription" in df.columns
    assert "abs_path" in df.columns
    assert os.path.exists(df["abs_path"].iloc[0])


def test_read_fsc_demographics(synthetic_fsc_dataset):
    paths = FSCPaths.from_root(str(synthetic_fsc_dataset))
    demo_df = read_fsc_demographics_csv(paths.demographics_csv)
    assert len(demo_df) == 3
    # Verify stripped column header
    assert "Self-reported fluency level" in demo_df.columns


def test_extract_intent_vocabularies(synthetic_fsc_dataset):
    train_df, _, _, _ = load_fsc_splits(str(synthetic_fsc_dataset))
    intent_map, slot_vocab = extract_intent_vocabularies(train_df)
    assert len(intent_map) == 2
    assert slot_vocab["action"] == ["activate", "deactivate"]
    assert slot_vocab["object"] == ["lights", "music"]
    assert slot_vocab["location"] == ["kitchen", "none"]


def test_analyze_split_overlaps(synthetic_fsc_dataset):
    train_df, valid_df, test_df, _ = load_fsc_splits(str(synthetic_fsc_dataset))
    overlaps = analyze_split_overlaps(train_df, valid_df, test_df)

    assert overlaps["speakers"]["train_test_overlap"] == 0  # Disjoint speakers
    assert overlaps["transcripts"]["train_test_overlap"] == 1  # Shared phrase
    assert overlaps["transcripts"]["test_utterances_in_train_pct"] == 100.0  # Phrase leakage check
