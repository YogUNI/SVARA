"""Audio data augmentation module for SVARA training.

Implements:
- Additive noise mixing with controllable SNR [0, 20] dB
- Random gain scaling (±6 dB)
- Speed / pitch perturbation (0.9x, 1.0x, 1.1x)
- Pipeline wrapper for training dataset transforms

Task: P4-03
Reference: docs/02 §4, docs/11 V-10
"""

import os
from typing import List, Optional, Tuple

import numpy as np


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
    ):
        self.noise_files = noise_files or []
        self.noise_prob = noise_prob
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.gain_prob = gain_prob
        self.gain_db_range = gain_db_range
        self.speed_prob = speed_prob
        self.speed_factors = speed_factors

        # Preloaded cache for small noise pools
        self._noise_cache = []

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random augmentations to waveform in place or copied."""
        out = waveform.copy()

        # 1. Random Gain
        if np.random.rand() < self.gain_prob:
            out = apply_random_gain(out, self.gain_db_range[0], self.gain_db_range[1])

        # 2. Speed perturbation
        if np.random.rand() < self.speed_prob:
            out = apply_speed_perturbation(out, self.speed_factors)

        # 3. Additive noise (if noise sources provided)
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
