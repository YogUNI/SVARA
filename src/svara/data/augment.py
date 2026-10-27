"""Audio data augmentation module for SVARA training.

Implements:
- Additive noise mixing with controllable SNR [0, 20] dB
- Random gain scaling (±6 dB)
- Speed / pitch perturbation (0.9x, 1.0x, 1.1x)
- Schroeder digital reverberator for residential smart-home acoustics (Sub-step 4.1)
- Pipeline wrapper for training dataset transforms

Task: P4-03, Sub-step 4.1
Reference: docs/02 §4, docs/11 V-10, docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# Acoustic presets for smart-home environments (Sub-step 4.1)
SCHROEDER_ROOM_PRESETS: Dict[str, Dict[str, Any]] = {
    "bedroom": {
        "rt60": 0.28,
        "wet": 0.18,
        "comb_delays": (479, 587, 691, 797),
        "ap_delays": (89, 23),
        "ap_gain": 0.7,
        "description": "Kamar Tidur (Moderate absorption, soft furnishings, short reverberation)",
    },
    "living_room": {
        "rt60": 0.48,
        "wet": 0.28,
        "comb_delays": (739, 887, 1039, 1187),
        "ap_delays": (149, 43),
        "ap_gain": 0.7,
        "description": "Ruang Tamu / Keluarga (Larger volume, hard floor, moderate reverberation)",
    },
    "kitchen": {
        "rt60": 0.36,
        "wet": 0.22,
        "comb_delays": (599, 709, 829, 953),
        "ap_delays": (109, 31),
        "ap_gain": 0.7,
        "description": "Dapur (Reflective ceramic tiles, metallic fixtures, medium reverberation)",
    },
}


def _feedback_comb_filter(waveform: np.ndarray, delay: int, gain: float) -> np.ndarray:
    """Applies a feedback comb filter y[n] = x[n] + gain * y[n - delay] via chunk vectorization."""
    n_samples = len(waveform)
    if delay <= 0 or abs(gain) < 1e-7 or n_samples == 0:
        return waveform.copy()

    y = np.zeros(n_samples, dtype=np.float32)
    y[:min(delay, n_samples)] = waveform[:min(delay, n_samples)]

    num_chunks = int(np.ceil(n_samples / delay))
    for i in range(1, num_chunks):
        start = i * delay
        end = min((i + 1) * delay, n_samples)
        prev_start = (i - 1) * delay
        prev_end = prev_start + (end - start)
        y[start:end] = waveform[start:end] + gain * y[prev_start:prev_end]
    return y


def _schroeder_allpass_filter(waveform: np.ndarray, delay: int, gain: float = 0.7) -> np.ndarray:
    """Applies a Schroeder all-pass filter with unity magnitude response for echo diffusion."""
    n_samples = len(waveform)
    if delay <= 0 or abs(gain) < 1e-7 or n_samples == 0:
        return waveform.copy()

    v = _feedback_comb_filter(waveform, delay, gain)
    delayed_v = np.zeros(n_samples, dtype=np.float32)
    if delay < n_samples:
        delayed_v[delay:] = v[:-delay]
    return (-gain * v + delayed_v).astype(np.float32)


def apply_schroeder_reverb(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    room_preset: str = "bedroom",
    rt60: Optional[float] = None,
    wet: Optional[float] = None,
    headroom_peak: float = 0.95,
) -> np.ndarray:
    """Applies Schroeder digital reverberation modeling residential smart-home acoustics.

    Simulates room impulse response reflections via:
    1. Four parallel feedback comb filters with mutually prime delays (echo simulation).
    2. Two cascaded all-pass filters (modal density diffusion and phase decorrelation).
    3. Proportional wet/dry blending with automatic headroom peak normalization (<= 0.95).

    Args:
        waveform: 1D audio array (float32).
        sample_rate: Sampling frequency in Hz (default 16000).
        room_preset: Preconfigured acoustic room profile ('bedroom', 'living_room', 'kitchen').
        rt60: Optional override for reverberation time in seconds.
        wet: Optional override for wet reverberation ratio [0.0, 1.0].
        headroom_peak: Maximum peak amplitude allowed before scaling (anti-clipping).

    Returns:
        Reverberated audio array (float32) matching input length.
    """
    if len(waveform) == 0:
        return waveform.copy()

    # Pass through pure silence without unnecessary computation
    if np.max(np.abs(waveform)) < 1e-7:
        return waveform.copy()

    cfg = SCHROEDER_ROOM_PRESETS.get(room_preset, SCHROEDER_ROOM_PRESETS["bedroom"])
    rt = float(rt60 if rt60 is not None else cfg["rt60"])
    w = float(wet if wet is not None else cfg["wet"])

    if w <= 0.0:
        return waveform.copy()

    # Feedback comb filter gains derived from Eyring-Schroeder equation: g = 10^(-3 * D / (RT60 * fs))
    comb_delays = cfg["comb_delays"]
    comb_gains = [float(10.0 ** (-3.0 * d / (rt * sample_rate))) for d in comb_delays]

    y_comb = np.zeros_like(waveform, dtype=np.float32)
    for d, g in zip(comb_delays, comb_gains):
        y_comb += _feedback_comb_filter(waveform, d, g)
    y_comb /= float(len(comb_delays))

    # All-pass diffusion stages
    ap_delays = cfg["ap_delays"]
    ap_gain = float(cfg.get("ap_gain", 0.7))
    y_ap = y_comb
    for d in ap_delays:
        y_ap = _schroeder_allpass_filter(y_ap, d, ap_gain)

    # Wet / Dry blending
    out = ((1.0 - w) * waveform + w * y_ap).astype(np.float32)

    # Enforce headroom peak limit (Rule 7 AGENTS.md anti-clipping)
    peak = float(np.max(np.abs(out)))
    if peak > headroom_peak and peak > 0.0:
        out = (out / peak) * headroom_peak

    return out.astype(np.float32)


class SchroederReverberator:
    """Stateful reverberator wrapper supporting room preset switching."""

    def __init__(
        self,
        default_room: str = "bedroom",
        sample_rate: int = 16000,
        headroom_peak: float = 0.95,
    ):
        self.default_room = default_room
        self.sample_rate = sample_rate
        self.headroom_peak = headroom_peak

    def __call__(
        self,
        waveform: np.ndarray,
        room_preset: Optional[str] = None,
        wet: Optional[float] = None,
    ) -> np.ndarray:
        room = room_preset or self.default_room
        return apply_schroeder_reverb(
            waveform=waveform,
            sample_rate=self.sample_rate,
            room_preset=room,
            wet=wet,
            headroom_peak=self.headroom_peak,
        )


def apply_random_gain(waveform: np.ndarray, min_gain_db: float = -6.0, max_gain_db: float = 6.0) -> np.ndarray:
    """Apply random amplitude gain in decibels."""
    gain_db = np.random.uniform(min_gain_db, max_gain_db)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (waveform * gain_linear).astype(np.float32)


def apply_speed_perturbation(
    waveform: np.ndarray,
    speed_factors: Tuple[float, ...] = (0.9, 1.0, 1.1),
) -> np.ndarray:
    """Perturb playback speed via linear interpolation resampling."""
    factor = float(np.random.choice(speed_factors))
    if factor == 1.0:
        return waveform

    orig_len = len(waveform)
    new_len = int(round(orig_len / factor))
    perturbed = np.interp(
        np.linspace(0.0, 1.0, new_len, endpoint=False),
        np.linspace(0.0, 1.0, orig_len, endpoint=False),
        waveform,
    ).astype(np.float32)
    return perturbed


def mix_noise_at_snr(
    clean_audio: np.ndarray,
    noise_audio: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix additive noise into clean audio at specified Signal-to-Noise Ratio (SNR)."""
    # Calculate signal power
    clean_power = np.mean(clean_audio ** 2)
    if clean_power < 1e-9:
        return clean_audio

    # If noise is shorter, tile/repeat noise; if longer, take random slice
    if len(noise_audio) < len(clean_audio):
        repeat_count = int(np.ceil(len(clean_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeat_count)

    if len(noise_audio) > len(clean_audio):
        start = np.random.randint(0, len(noise_audio) - len(clean_audio) + 1)
        noise_segment = noise_audio[start : start + len(clean_audio)]
    else:
        noise_segment = noise_audio[: len(clean_audio)]

    noise_power = np.mean(noise_segment ** 2)
    if noise_power < 1e-9:
        return clean_audio

    # Target noise power for desired SNR: SNR = 10 * log10(clean_power / noise_power)
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)

    mixed = clean_audio + scale * noise_segment
    return mixed.astype(np.float32)


class AudioAugmentor:
    """Composited audio augmentation pipeline for training."""

    def __init__(
        self,
        noise_files: Optional[List[str]] = None,
        noise_prob: float = 0.5,
        min_snr_db: float = 0.0,
        max_snr_db: float = 20.0,
        gain_prob: float = 0.5,
        gain_db_range: Tuple[float, float] = (-6.0, 6.0),
        speed_prob: float = 0.3,
        speed_factors: Tuple[float, ...] = (0.9, 1.0, 1.1),
        reverb_prob: float = 0.0,
        reverb_rooms: Tuple[str, ...] = ("bedroom", "living_room"),
    ):
        self.noise_files = noise_files or []
        self.noise_prob = noise_prob
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.gain_prob = gain_prob
        self.gain_db_range = gain_db_range
        self.speed_prob = speed_prob
        self.speed_factors = speed_factors
        self.reverb_prob = reverb_prob
        self.reverb_rooms = reverb_rooms

        # Preloaded cache for small noise pools
        self._noise_cache = []

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random augmentations to waveform in place or copied."""
        out = waveform.copy()

        # 1. Room Reverberation (Sub-step 4.1)
        if self.reverb_prob > 0.0 and np.random.rand() < self.reverb_prob:
            room = str(np.random.choice(self.reverb_rooms))
            out = apply_schroeder_reverb(out, room_preset=room)

        # 2. Random Gain
        if np.random.rand() < self.gain_prob:
            out = apply_random_gain(out, self.gain_db_range[0], self.gain_db_range[1])

        # 3. Speed perturbation
        if np.random.rand() < self.speed_prob:
            out = apply_speed_perturbation(out, self.speed_factors)

        # 4. Additive noise (if noise sources provided)
        if self.noise_files and np.random.rand() < self.noise_prob:
            noise_path = np.random.choice(self.noise_files)
            # Read noise
            import soundfile as sf
            try:
                noise_wav, _ = sf.read(noise_path, dtype="float32")
                if noise_wav.ndim > 1:
                    noise_wav = np.mean(noise_wav, axis=1)
                snr = np.random.uniform(self.min_snr_db, self.max_snr_db)
                out = mix_noise_at_snr(out, noise_wav, snr)
            except Exception:
                pass

        return out
