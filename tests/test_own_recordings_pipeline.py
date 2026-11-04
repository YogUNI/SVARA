"""Unit tests for Own Recordings Validation and Standardization Pipeline.

Task: Sub-step 13.2 / P5-02
Reference: docs/01 §8, docs/10 P5-02, docs/14 FASE 13
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import soundfile as sf

from scripts.validate_and_convert_own_recordings import (
    audit_own_recordings,
    convert_and_standardize_audio,
    parse_filename,
)


def test_parse_filename_formats():
    """Verify regex filename parsing for various speaker formats."""
    # Standard format
    assert parse_filename("own_haikal_q_00.wav") == ("own_haikal", "quiet", 0)
    assert parse_filename("own_yoga_n_15.wav") == ("own_yoga", "noisy", 15)
    assert parse_filename("own_s01_quiet_30.wav") == ("own_s01", "quiet", 30)

    # Simplified format
    assert parse_filename("haikal_q_05.wav") == ("own_haikal", "quiet", 5)
    assert parse_filename("yoga_12.wav") == ("own_yoga", "quiet", 12)

    # Invalid format
    assert parse_filename("random_audio_test.wav") is None


def test_convert_and_standardize_audio(tmp_path):
    """Verify audio conversion enforces 16 kHz Mono PCM_16 and peak <= 0.95."""
    # Create a stereo 44.1 kHz loud audio file
    sr_orig = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr_orig * duration))
    # Loud stereo signal (peak = 1.5)
    stereo_audio = np.stack([np.sin(2 * np.pi * 440 * t) * 1.5, np.cos(2 * np.pi * 440 * t) * 1.5], axis=1)

    in_file = str(tmp_path / "raw_stereo.wav")
    out_file = str(tmp_path / "standardized_mono.wav")
    sf.write(in_file, stereo_audio, samplerate=sr_orig)

    dur, peak, rms = convert_and_standardize_audio(in_file, out_file, target_sr=16000, headroom_peak=0.95)

    assert os.path.exists(out_file)
    data, sr = sf.read(out_file)
    assert sr == 16000
    assert data.ndim == 1  # Mono
    assert peak <= 0.950001
    assert abs(dur - duration) < 0.05


def test_audit_own_recordings_pipeline(tmp_path):
    """Verify end-to-end ingestion and metadata generation."""
    raw_dir = str(tmp_path / "raw")
    wavs_dir = str(tmp_path / "wavs")
    meta_path = str(tmp_path / "metadata.csv")
    report_path = str(tmp_path / "audit.json")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(wavs_dir, exist_ok=True)

    # Simulate Haikal dropping a raw file
    sr = 16000
    dummy_wav = np.sin(np.linspace(0, 100, sr * 2)).astype(np.float32) * 0.5
    raw_file = os.path.join(raw_dir, "haikal_q_00.wav")
    sf.write(raw_file, dummy_wav, samplerate=sr)

    # Mock legal map with intent 0
    legal_map = {0: {"action": "activate", "object": "lamp", "location": "none"}}

    summary = audit_own_recordings(
        raw_dir=raw_dir,
        wavs_dir=wavs_dir,
        metadata_path=meta_path,
        report_path=report_path,
        legal_map=legal_map,
    )

    assert summary["total_clips_registered"] == 1
    assert "own_haikal" in summary["speakers_found"]
    assert "quiet" in summary["conditions_found"]
    assert summary["unique_intents_covered"] == 1
    assert summary["status"] == "PASSED"
    assert os.path.exists(meta_path)
    assert os.path.exists(report_path)
