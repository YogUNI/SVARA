"""Unit tests for shared audio preprocessing and dataset collation.

Task: P2-05, P2-06
Reference: AGENTS.md rule 7, docs/02 §10
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from svara.data.audio import (
    TARGET_SAMPLE_RATE,
    normalize_waveform,
    preprocess_audio,
)
from svara.data.collate import collate_fn_pad


@pytest.fixture
def synthetic_wav_file(tmp_path):
    wav_p = tmp_path / "test_tone.wav"
    # Generate 1.0s sine wave at 16000 Hz
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(wav_p), sine, 16000)
    return str(wav_p)


def test_normalize_waveform():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    norm = normalize_waveform(data)
    assert np.isclose(np.mean(norm), 0.0, atol=1e-6)
    assert np.isclose(np.std(norm), 1.0, atol=1e-5)


def test_preprocess_audio_parity_file_and_array(synthetic_wav_file):
    """Verify that preprocessing from file path equals preprocessing from raw array (Rule 7)."""
    wav_from_file, len_f, sr_f = preprocess_audio(synthetic_wav_file, max_audio_seconds=2.0)
    data_raw, sr = sf.read(synthetic_wav_file, dtype="float32")
    wav_from_array, len_a, sr_a = preprocess_audio(data_raw, sample_rate=sr, max_audio_seconds=2.0)

    assert sr_f == TARGET_SAMPLE_RATE
    assert sr_a == TARGET_SAMPLE_RATE
    assert len_f == len_a
    assert np.allclose(wav_from_file, wav_from_array, atol=1e-5)


def test_preprocess_audio_crop_and_pad(synthetic_wav_file):
    # Test crop when audio exceeds max duration
    wav_cropped, length_c, _ = preprocess_audio(synthetic_wav_file, max_audio_seconds=0.5, pad=False)
    assert len(wav_cropped) == int(0.5 * TARGET_SAMPLE_RATE)
    assert length_c == int(0.5 * TARGET_SAMPLE_RATE)

    # Test pad when audio shorter than max duration
    wav_padded, length_p, _ = preprocess_audio(synthetic_wav_file, max_audio_seconds=2.0, pad=True)
    assert len(wav_padded) == int(2.0 * TARGET_SAMPLE_RATE)
    assert length_p == int(1.0 * TARGET_SAMPLE_RATE)


def test_collate_fn_pad_batch():
    batch = [
        {
            "waveform": torch.randn(16000),  # 1.0s
            "length": 16000,
            "action_id": torch.tensor(1),
            "object_id": torch.tensor(2),
            "location_id": torch.tensor(0),
            "intent_id": torch.tensor(5),
        },
        {
            "waveform": torch.randn(24000),  # 1.5s
            "length": 24000,
            "action_id": torch.tensor(3),
            "object_id": torch.tensor(1),
            "location_id": torch.tensor(2),
            "intent_id": torch.tensor(12),
        },
    ]

    collated = collate_fn_pad(batch)
    assert collated["waveform"].shape == (2, 24000)
    assert collated["lengths"].tolist() == [16000, 24000]
    assert collated["action_id"].shape == (2,)
    # Verify zero-padding in first element
    assert torch.all(collated["waveform"][0, 16000:] == 0.0)
