"""Unit tests for audio augmentation module (Task P4-03)."""

import numpy as np
import pytest

from svara.data.augment import (
    AudioAugmentor,
    apply_random_gain,
    apply_speed_perturbation,
    mix_noise_at_snr,
    apply_schroeder_reverb,
    SchroederReverberator,
    SCHROEDER_ROOM_PRESETS,
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


def test_apply_schroeder_reverb_shape_and_numerical_health():
    """Verify output shape preservation, zero NaN/Inf, and peak headroom <= 0.95."""
    np.random.seed(42)
    # 2 seconds of synthetic speech-like tone
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    waveform = (0.7 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)

    reverbed = apply_schroeder_reverb(waveform, sample_rate=16000, room_preset="bedroom")

    assert reverbed.shape == waveform.shape
    assert not np.isnan(reverbed).any(), "Reverberated audio must not contain NaN"
    assert not np.isinf(reverbed).any(), "Reverberated audio must not contain Inf"
    assert np.max(np.abs(reverbed)) <= 0.950001, "Peak amplitude must strictly obey <= 0.95 headroom limit"


def test_apply_schroeder_reverb_dry_passthrough():
    """Verify that wet=0.0 produces identical bit-level passthrough."""
    waveform = np.random.randn(16000).astype(np.float32) * 0.5
    passthrough = apply_schroeder_reverb(waveform, wet=0.0)

    assert np.array_equal(passthrough, waveform), "Dry passthrough must be bit-for-bit identical"


def test_apply_schroeder_reverb_silence_passthrough():
    """Verify that zero/silent input returns pure silence without artifact injection."""
    silence = np.zeros(16000, dtype=np.float32)
    out = apply_schroeder_reverb(silence, room_preset="living_room")

    assert np.array_equal(out, silence), "Silent input must output silence without artifacts"


def test_apply_schroeder_reverb_room_presets_tail_energy():
    """Verify physical room acoustic decay: living room has significantly higher tail energy than bedroom."""
    sr = 16000
    impulse = np.zeros(sr * 2, dtype=np.float32)
    impulse[0] = 1.0

    h_bedroom = apply_schroeder_reverb(impulse, sample_rate=sr, room_preset="bedroom", wet=1.0)
    h_living = apply_schroeder_reverb(impulse, sample_rate=sr, room_preset="living_room", wet=1.0)

    # Calculate decay tail energy between 200ms and 500ms
    tail_start = int(0.20 * sr)
    tail_end = int(0.50 * sr)
    tail_energy_bedroom = np.sum(h_bedroom[tail_start:tail_end] ** 2)
    tail_energy_living = np.sum(h_living[tail_start:tail_end] ** 2)

    assert tail_energy_living > tail_energy_bedroom * 5.0, (
        f"Living room tail energy ({tail_energy_living:.2e}) should be much higher than bedroom ({tail_energy_bedroom:.2e})"
    )


def test_schroeder_reverberator_class_and_augmentor_integration():
    """Verify SchroederReverberator class and AudioAugmentor pipeline integration."""
    reverb_obj = SchroederReverberator(default_room="bedroom", sample_rate=16000)
    wav = np.random.randn(16000).astype(np.float32) * 0.3

    out1 = reverb_obj(wav)
    assert out1.shape == wav.shape
    assert not np.isnan(out1).any()

    # With AudioAugmentor
    augmentor = AudioAugmentor(
        gain_prob=0.0,
        speed_prob=0.0,
        noise_prob=0.0,
        reverb_prob=1.0,
        reverb_rooms=("bedroom", "living_room"),
    )
    out2 = augmentor(wav)
    assert out2.shape == wav.shape
    assert not np.isnan(out2).any()
    assert not np.array_equal(out2, wav), "Augmentor with reverb_prob=1.0 must modify audio"

