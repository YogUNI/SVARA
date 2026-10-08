"""Unit tests for audio augmentation module (Task P4-03)."""

import numpy as np
import pytest

from svara.data.augment import (
    AudioAugmentor,
    apply_random_gain,
    apply_speed_perturbation,
    mix_noise_at_snr,
)


def test_apply_random_gain():
    np.random.seed(42)
    clean = np.ones(16000, dtype=np.float32)
    augmented = apply_random_gain(clean, min_gain_db=-6.0, max_gain_db=6.0)

    assert augmented.shape == clean.shape
    # Check gain is within [-6dB, +6dB] linear bounds [0.50, 2.0]
    gain = augmented[0] / clean[0]
    assert 0.49 <= gain <= 2.01


def test_apply_speed_perturbation():
    clean = np.ones(16000, dtype=np.float32)

    # 0.9x speed (longer)
    faster = apply_speed_perturbation(clean, speed_factors=(0.9,))
    assert len(faster) > len(clean)

    # 1.1x speed (shorter)
    slower = apply_speed_perturbation(clean, speed_factors=(1.1,))
    assert len(slower) < len(clean)


def test_mix_noise_at_snr():
    np.random.seed(42)
    clean = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    noise = np.random.normal(0, 0.5, 16000).astype(np.float32)

    mixed = mix_noise_at_snr(clean, noise, snr_db=10.0)
    assert mixed.shape == clean.shape
    assert not np.isnan(mixed).any()
    assert not np.array_equal(mixed, clean)


def test_audio_augmentor_pipeline():
    augmentor = AudioAugmentor(gain_prob=1.0, speed_prob=0.0, noise_prob=0.0)
    wav = np.ones(16000, dtype=np.float32)
    out = augmentor(wav)

    assert out.shape == wav.shape
    assert not np.isnan(out).any()
