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
import fractions
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal

try:
    import torch
    import torchaudio
    HAS_TORCH_AUDIO = True
except ImportError:
    torch = None
    torchaudio = None
    HAS_TORCH_AUDIO = False


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


DEFAULT_SPEED_FACTORS: Tuple[float, ...] = (0.92, 1.0, 1.08)


def apply_speed_perturbation(
    waveform: np.ndarray,
    speed_factors: Tuple[float, ...] = DEFAULT_SPEED_FACTORS,
    factor: Optional[float] = None,
    method: str = "sinc",
    headroom_peak: float = 0.95,
) -> np.ndarray:
    """Perturb playback speed via high-fidelity polyphase sinc or linear resampling.

    Simulates speech tempo variability (Sub-step 4.3):
    - 0.92x: Slower cadence (duration increases by ~8.7%)
    - 1.00x: Normal cadence (duration unchanged)
    - 1.08x: Faster cadence (duration shortens by ~7.4%)

    Args:
        waveform: 1D audio array (float32).
        speed_factors: Tuple of possible speed multipliers (default: (0.92, 1.0, 1.08)).
        factor: Optional explicit speed factor. If None, chosen randomly from speed_factors.
        method: Resampling method ('sinc' for polyphase anti-aliased sinc, 'linear' for interp).
        headroom_peak: Maximum peak amplitude allowed before scaling (anti-clipping <= 0.95).

    Returns:
        Perturbed audio array (float32) with adjusted duration.
    """
    if len(waveform) == 0:
        return waveform.copy()

    chosen_factor = float(factor if factor is not None else np.random.choice(speed_factors))
    if abs(chosen_factor - 1.0) < 1e-4:
        return waveform.copy()

    orig_len = len(waveform)
    new_len = int(round(orig_len / chosen_factor))

    if method == "sinc":
        try:
            # Polyphase sinc resampling with rational fraction approximation
            frac = fractions.Fraction(chosen_factor).limit_denominator(50)
            up = frac.denominator
            down = frac.numerator
            perturbed = signal.resample_poly(waveform, up, down).astype(np.float32)
            # Ensure exact length match with target duration
            if len(perturbed) > new_len:
                perturbed = perturbed[:new_len]
            elif len(perturbed) < new_len:
                perturbed = np.pad(perturbed, (0, new_len - len(perturbed)), mode="constant")
        except Exception:
            # Robust fallback to linear interpolation if resample_poly fails
            perturbed = np.interp(
                np.linspace(0.0, 1.0, new_len, endpoint=False),
                np.linspace(0.0, 1.0, orig_len, endpoint=False),
                waveform,
            ).astype(np.float32)
    else:
        perturbed = np.interp(
            np.linspace(0.0, 1.0, new_len, endpoint=False),
            np.linspace(0.0, 1.0, orig_len, endpoint=False),
            waveform,
        ).astype(np.float32)

    # Peak headroom control (anti-clipping rule)
    peak = float(np.max(np.abs(perturbed)))
    if peak > headroom_peak and peak > 0.0:
        perturbed = (perturbed / peak) * headroom_peak

    return perturbed.astype(np.float32)


def apply_laptop_mic_filter(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    cutoff_hz: float = 100.0,
    order: int = 2,
) -> np.ndarray:
    """Applies highpass filter modeling internal laptop microphone chassis frequency response.

    Attenuates mechanical chassis rumble, keyboard/desk vibrations, and sub-100Hz rumble.
    Uses a 2nd-order Butterworth highpass filter implemented via Second-Order Sections (SOS).
    """
    if len(waveform) == 0:
        return waveform.copy()
    if np.max(np.abs(waveform)) < 1e-7:
        return waveform.copy()

    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    filtered = signal.sosfilt(sos, waveform).astype(np.float32)
    return filtered


def generate_ambient_noise(
    n_samples: int,
    sample_rate: int = 16000,
    noise_type: str = "ac_fan",
) -> np.ndarray:
    """Generates realistic residential ambient noise profiles (AC, fan, room hum).

    Profiles:
    - 'ac': Lowpass air circulation at 1200 Hz with 50/100 Hz compressor hum.
    - 'fan': Broadband aerodynamic blade noise (1600 Hz LP) with 120/240 Hz motor hum.
    - 'ac_fan': Combined residential room air conditioner and ceiling/desk fan noise.
    - 'ambient_room': Gentle room thermal and environmental floor noise.
    """
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    white = np.random.randn(n_samples).astype(np.float32)
    nyquist = sample_rate / 2.0

    if noise_type == "ac":
        sos = signal.butter(2, min(1200.0, nyquist - 100.0), btype="lowpass", fs=sample_rate, output="sos")
        noise = signal.sosfilt(sos, white)
        t = np.arange(n_samples, dtype=np.float32) / sample_rate
        hum = 0.12 * np.sin(2.0 * np.pi * 50.0 * t) + 0.08 * np.sin(2.0 * np.pi * 100.0 * t)
        noise = (noise + hum).astype(np.float32)
    elif noise_type == "fan":
        sos = signal.butter(2, min(1600.0, nyquist - 100.0), btype="lowpass", fs=sample_rate, output="sos")
        noise = signal.sosfilt(sos, white)
        t = np.arange(n_samples, dtype=np.float32) / sample_rate
        hum = 0.15 * np.sin(2.0 * np.pi * 120.0 * t) + 0.10 * np.sin(2.0 * np.pi * 240.0 * t)
        noise = (noise + hum).astype(np.float32)
    elif noise_type == "ambient_room":
        sos = signal.butter(1, min(800.0, nyquist - 100.0), btype="lowpass", fs=sample_rate, output="sos")
        noise = signal.sosfilt(sos, white).astype(np.float32)
    else:  # default 'ac_fan'
        sos = signal.butter(2, min(1400.0, nyquist - 100.0), btype="lowpass", fs=sample_rate, output="sos")
        noise = signal.sosfilt(sos, white)
        t = np.arange(n_samples, dtype=np.float32) / sample_rate
        hum = 0.10 * np.sin(2.0 * np.pi * 100.0 * t) + 0.06 * np.sin(2.0 * np.pi * 200.0 * t)
        noise = (noise + hum).astype(np.float32)

    return noise


def inject_household_noise(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    snr_db: Optional[float] = None,
    min_snr_db: float = 16.0,
    max_snr_db: float = 26.0,
    noise_type: str = "ac_fan",
    external_noise: Optional[np.ndarray] = None,
    headroom_peak: float = 0.95,
) -> Tuple[np.ndarray, float]:
    """Injects calibrated household background noise at specified or randomly chosen SNR [16, 26] dB."""
    if len(waveform) == 0:
        return waveform.copy(), 0.0

    clean_power = float(np.mean(waveform ** 2))
    if clean_power < 1e-9:
        return waveform.copy(), 0.0

    target_snr = float(snr_db if snr_db is not None else np.random.uniform(min_snr_db, max_snr_db))

    if external_noise is not None and len(external_noise) > 0:
        noise = external_noise.copy()
        if len(noise) < len(waveform):
            repeat_count = int(np.ceil(len(waveform) / len(noise)))
            noise = np.tile(noise, repeat_count)
        if len(noise) > len(waveform):
            start = np.random.randint(0, len(noise) - len(waveform) + 1)
            noise = noise[start : start + len(waveform)]
        else:
            noise = noise[: len(waveform)]
    else:
        noise = generate_ambient_noise(len(waveform), sample_rate=sample_rate, noise_type=noise_type)

    noise_power = float(np.mean(noise ** 2))
    if noise_power < 1e-9:
        return waveform.copy(), target_snr

    target_noise_power = clean_power / (10.0 ** (target_snr / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    mixed = (waveform + scale * noise).astype(np.float32)

    # Headroom peak control
    peak = float(np.max(np.abs(mixed)))
    if peak > headroom_peak and peak > 0.0:
        mixed = (mixed / peak) * headroom_peak

    return mixed.astype(np.float32), target_snr


def apply_laptop_mic_and_noise(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    cutoff_hz: float = 100.0,
    snr_db: Optional[float] = None,
    min_snr_db: float = 16.0,
    max_snr_db: float = 26.0,
    noise_type: str = "ac_fan",
    headroom_peak: float = 0.95,
) -> np.ndarray:
    """End-to-end pipeline combining 100 Hz highpass laptop mic filter and household AC/fan noise injection."""
    filtered = apply_laptop_mic_filter(waveform, sample_rate=sample_rate, cutoff_hz=cutoff_hz)
    mixed, _ = inject_household_noise(
        filtered,
        sample_rate=sample_rate,
        snr_db=snr_db,
        min_snr_db=min_snr_db,
        max_snr_db=max_snr_db,
        noise_type=noise_type,
        headroom_peak=headroom_peak,
    )
    return mixed


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


# --------------------------------------------------------------------------
# Dynamic Pitch & Prosody Shifter (Sub-step 12.1)
# --------------------------------------------------------------------------
_PITCH_SHIFTER_CACHE: Dict[Tuple[int, int, str], Any] = {}
DEFAULT_PITCH_STEPS: Tuple[int, ...] = (-4, -3, -2, 2, 3, 4)


def apply_pitch_shift_cuda(
    waveform: Union[np.ndarray, Any],
    n_steps: int,
    sr: int = 16000,
    headroom_peak: float = 0.95,
    device: Optional[Union[str, Any]] = None,
) -> Union[np.ndarray, Any]:
    """Applies high-fidelity pitch shifting accelerated by PyTorch CUDA GPU.

    Task: Sub-step 12.1 (Phase 12: Real-World Speech Hardening)
    Reference: docs/14_ROADMAP_PEMATANGAN_DEEP_LEARNING_REAL_WORLD.md, AGENTS.md

    Shifts pitch by n_steps semitones (e.g. +2, +3, +4 or -2, -3, -4) using
    Torchaudio's phase vocoder PitchShift transform. Runs natively on CUDA
    (e.g. RTX 4060) with automatic CPU fallback.
    Prevents digital clipping via headroom peak normalization (<= headroom_peak).

    Args:
        waveform: 1D audio array (float32 numpy array or torch Tensor).
        n_steps: Number of semitones to shift (positive for higher pitch, negative for lower).
                 If 0, returns a copy of waveform.
        sr: Sampling rate in Hz (default 16000).
        headroom_peak: Maximum peak amplitude guard (default 0.95) to prevent inter-sample clipping.
        device: Target torch device ('cuda', 'cpu', or None for auto-detection).

    Returns:
        Pitch-shifted waveform matching input type (np.ndarray or torch.Tensor)
        with exact original length preserved.
    """
    if n_steps == 0:
        if isinstance(waveform, np.ndarray):
            return waveform.copy()
        return waveform.clone()

    if not HAS_TORCH_AUDIO or torch is None or torchaudio is None:
        raise RuntimeError("torch and torchaudio are required for apply_pitch_shift_cuda.")

    is_numpy = isinstance(waveform, np.ndarray)
    if is_numpy:
        tensor = torch.from_numpy(waveform.astype(np.float32))
    elif isinstance(waveform, torch.Tensor):
        tensor = waveform.float()
    else:
        raise TypeError(f"Unsupported waveform type: {type(waveform)}. Expected np.ndarray or torch.Tensor.")

    orig_dim = tensor.dim()
    orig_len = tensor.shape[-1]

    # Reshape to (1, T) if 1D
    if orig_dim == 1:
        tensor = tensor.unsqueeze(0)

    # Device selection
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        target_device = torch.device(device)
    else:
        target_device = device

    cache_key = (sr, n_steps, str(target_device))
    if cache_key not in _PITCH_SHIFTER_CACHE:
        shifter = torchaudio.transforms.PitchShift(
            sample_rate=sr,
            n_steps=n_steps,
            bins_per_octave=12,
            n_fft=512,
            win_length=512,
            hop_length=128,
        ).to(target_device)
        _PITCH_SHIFTER_CACHE[cache_key] = shifter
    else:
        shifter = _PITCH_SHIFTER_CACHE[cache_key]

    with torch.no_grad():
        in_tensor = tensor.to(target_device)
        shifted = shifter(in_tensor)

        # Exact length alignment
        if shifted.shape[-1] > orig_len:
            shifted = shifted[..., :orig_len]
        elif shifted.shape[-1] < orig_len:
            pad_len = orig_len - shifted.shape[-1]
            shifted = torch.nn.functional.pad(shifted, (0, pad_len))

        # Headroom peak guard (<= headroom_peak)
        peak = float(torch.max(torch.abs(shifted)).item())
        if peak > headroom_peak and peak > 1e-6:
            shifted = shifted * (headroom_peak / peak)

    if is_numpy:
        out = shifted.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return out
    else:
        out = shifted.detach()
        if orig_dim == 1:
            out = out.squeeze(0)
        return out


class AudioAugmentor:
    """Composited audio augmentation pipeline for training."""

    def __init__(
        self,
        noise_files: Optional[List[str]] = None,
        noise_prob: float = 0.5,
        min_snr_db: float = 16.0,
        max_snr_db: float = 26.0,
        gain_prob: float = 0.5,
        gain_db_range: Tuple[float, float] = (-6.0, 6.0),
        speed_prob: float = 0.3,
        speed_factors: Tuple[float, ...] = DEFAULT_SPEED_FACTORS,
        pitch_prob: float = 0.0,
        pitch_steps: Tuple[int, ...] = DEFAULT_PITCH_STEPS,
        reverb_prob: float = 0.0,
        reverb_rooms: Tuple[str, ...] = ("bedroom", "living_room"),
        mic_filter_prob: float = 0.0,
        mic_cutoff_hz: float = 100.0,
    ):
        self.noise_files = noise_files or []
        self.noise_prob = noise_prob
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.gain_prob = gain_prob
        self.gain_db_range = gain_db_range
        self.speed_prob = speed_prob
        self.speed_factors = speed_factors
        self.pitch_prob = pitch_prob
        self.pitch_steps = pitch_steps
        self.reverb_prob = reverb_prob
        self.reverb_rooms = reverb_rooms
        self.mic_filter_prob = mic_filter_prob
        self.mic_cutoff_hz = mic_cutoff_hz

        # Preloaded cache for small noise pools
        self._noise_cache = []

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply random augmentations to waveform in place or copied."""
        out = waveform.copy()

        # 1. Room Reverberation (Sub-step 4.1)
        if self.reverb_prob > 0.0 and np.random.rand() < self.reverb_prob:
            room = str(np.random.choice(self.reverb_rooms))
            out = apply_schroeder_reverb(out, room_preset=room)

        # 2. Laptop Internal Mic Highpass Filter (Sub-step 4.2)
        if self.mic_filter_prob > 0.0 and np.random.rand() < self.mic_filter_prob:
            out = apply_laptop_mic_filter(out, cutoff_hz=self.mic_cutoff_hz)

        # 3. Dynamic Pitch Shifting (Sub-step 12.1)
        if self.pitch_prob > 0.0 and np.random.rand() < self.pitch_prob:
            step = int(np.random.choice(self.pitch_steps))
            out = apply_pitch_shift_cuda(out, n_steps=step)

        # 4. Random Gain
        if np.random.rand() < self.gain_prob:
            out = apply_random_gain(out, self.gain_db_range[0], self.gain_db_range[1])

        # 5. Speed perturbation
        if np.random.rand() < self.speed_prob:
            out = apply_speed_perturbation(out, self.speed_factors)

        # 6. Additive noise (External noise files OR realistic household ambient noise) (Sub-step 4.2)
        if np.random.rand() < self.noise_prob:
            if self.noise_files:
                noise_path = np.random.choice(self.noise_files)
                import soundfile as sf
                try:
                    noise_wav, _ = sf.read(noise_path, dtype="float32")
                    if noise_wav.ndim > 1:
                        noise_wav = np.mean(noise_wav, axis=1)
                    snr = np.random.uniform(self.min_snr_db, self.max_snr_db)
                    out = mix_noise_at_snr(out, noise_wav, snr)
                except Exception:
                    out, _ = inject_household_noise(out, min_snr_db=self.min_snr_db, max_snr_db=self.max_snr_db)
            else:
                out, _ = inject_household_noise(out, min_snr_db=self.min_snr_db, max_snr_db=self.max_snr_db)

        return out
