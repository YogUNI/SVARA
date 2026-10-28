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
    apply_laptop_mic_filter,
    generate_ambient_noise,
    inject_household_noise,
    apply_laptop_mic_and_noise,
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


def test_apply_laptop_mic_filter_attenuation():
    """Verify 100 Hz highpass filter attenuates sub-bass rumble without touching vocal core."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    x_50 = np.sin(2 * np.pi * 50 * t).astype(np.float32)
    x_1000 = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    y_50 = apply_laptop_mic_filter(x_50, sample_rate=sr, cutoff_hz=100.0)
    y_1000 = apply_laptop_mic_filter(x_1000, sample_rate=sr, cutoff_hz=100.0)

    # Steady-state RMS ratio (excluding first 500 edge samples)
    mid = slice(500, -500)
    rms_x50 = np.sqrt(np.mean(x_50[mid] ** 2))
    rms_y50 = np.sqrt(np.mean(y_50[mid] ** 2))
    att_50 = 20 * np.log10(rms_y50 / rms_x50)

    rms_x1000 = np.sqrt(np.mean(x_1000[mid] ** 2))
    rms_y1000 = np.sqrt(np.mean(y_1000[mid] ** 2))
    att_1000 = 20 * np.log10(rms_y1000 / rms_x1000)

    assert att_50 < -10.0, f"50 Hz rumble must be attenuated by > 10 dB, got {att_50:.2f} dB"
    assert abs(att_1000) < 0.1, f"1000 Hz vocal tone must pass virtually unattenuated, got {att_1000:.2f} dB"


def test_generate_ambient_noise_profiles():
    """Verify ambient noise generator profiles (AC, fan, combined)."""
    n = 16000
    for ntype in ["ac", "fan", "ac_fan", "ambient_room"]:
        noise = generate_ambient_noise(n, sample_rate=16000, noise_type=ntype)
        assert len(noise) == n
        assert not np.isnan(noise).any()
        assert np.max(np.abs(noise)) > 0.05, f"Noise profile '{ntype}' should be audible"


def test_inject_household_noise_calibrated_snr():
    """Verify that injected noise adheres strictly to specified SNR [16, 26] dB."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    clean = (0.4 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
    clean_power = float(np.mean(clean ** 2))

    for target_snr in [16.0, 20.0, 26.0]:
        mixed, snr_used = inject_household_noise(clean, sample_rate=sr, snr_db=target_snr, noise_type="ac_fan")
        assert abs(snr_used - target_snr) < 1e-4

        # Injected noise alone = mixed - clean (since headroom clamp not triggered at 0.4 peak)
        injected = mixed - clean
        measured_snr = 10.0 * np.log10(clean_power / float(np.mean(injected ** 2)))
        assert abs(measured_snr - target_snr) < 0.2, f"Measured SNR {measured_snr:.2f} differs from target {target_snr:.2f}"
        assert np.max(np.abs(mixed)) <= 0.950001, "Mixed audio must not exceed headroom limit"


def test_apply_laptop_mic_and_noise_pipeline():
    """Verify chained filter and noise pipeline end-to-end."""
    waveform = np.random.randn(16000).astype(np.float32) * 0.3
    out = apply_laptop_mic_and_noise(waveform, sample_rate=16000, min_snr_db=16.0, max_snr_db=26.0)

    assert out.shape == waveform.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.max(np.abs(out)) <= 0.950001


def test_audio_augmentor_mic_and_noise_integration():
    """Verify AudioAugmentor pipeline with mic filter and ambient noise enabled."""
    wav = np.random.randn(16000).astype(np.float32) * 0.3
    augmentor = AudioAugmentor(
        gain_prob=0.0,
        speed_prob=0.0,
        reverb_prob=0.0,
        mic_filter_prob=1.0,
        noise_prob=1.0,
        min_snr_db=18.0,
        max_snr_db=22.0,
    )
    out = augmentor(wav)

    assert out.shape == wav.shape
    assert not np.isnan(out).any()
    assert not np.array_equal(out, wav)


