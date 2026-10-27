"""Unified audio preprocessing module for training, evaluation, and inference.

Task: P2-05, Sub-step 1.1, Sub-step 1.2, Sub-step 1.3
Reference: AGENTS.md rule 7, docs/01 §1, docs/02 §1, docs/05 §1, docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md

RULE 7 (AGENTS.md):
"Keep train/serve preprocessing identical. Same normalization, sample rate (16 kHz mono), max length.
Share one function in src/svara/data/audio.py."
"""

import os
from typing import Any, Optional, Tuple, Union

import numpy as np
from scipy import signal
import soundfile as sf

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

TARGET_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 5.0
DEFAULT_MAX_SAMPLES = int(DEFAULT_MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE)  # 80,000 samples

# Vocal Formant Range Butterworth Bandpass Filter (80 Hz - 7500 Hz)
# Eliminates hardware low-frequency body rumble and high-frequency electrical hiss
BANDPASS_LOW_CUT = 80.0
BANDPASS_HIGH_CUT = 7500.0


def apply_butterworth_bandpass(
    waveform: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    lowcut: float = BANDPASS_LOW_CUT,
    highcut: float = BANDPASS_HIGH_CUT,
    order: int = 2,
) -> np.ndarray:
    """Applies zero-phase 2nd order Butterworth bandpass filter to isolate human vocal range.

    Args:
        waveform: 1D float32 numpy array.
        sr: Sampling rate (16000 Hz).
        lowcut: Lower cutoff frequency (default 80.0 Hz).
        highcut: Upper cutoff frequency (default 7500.0 Hz).
        order: Filter order (default 2).

    Returns:
        Filtered 1D float32 waveform with mechanical rumble and extreme high hiss attenuated.
    """
    if len(waveform) <= 3 * (2 * order + 1):
        # Audio too short for filter pad
        return waveform

    nyq = 0.5 * sr
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = signal.butter(order, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, waveform)
    return filtered.astype(np.float32)


# Adaptive RMS Energy Normalization Target (-20 dBFS)
# Standardizes whisper to loud vocal projection before VAD and transformer feature extraction
DEFAULT_TARGET_RMS_DBFS = -20.0
DEFAULT_MAX_RMS_GAIN_DB = 30.0


def adaptive_rms_normalize(
    waveform: np.ndarray,
    target_dbfs: float = DEFAULT_TARGET_RMS_DBFS,
    max_gain_db: float = DEFAULT_MAX_RMS_GAIN_DB,
    min_rms_threshold: float = 1e-4,
    eps: float = 1e-7,
) -> np.ndarray:
    """Normalizes waveform energy to a target RMS level in dBFS with gain limiting.

    Ensures quiet whispers and loud close-mic utterances have consistent energy
    before VAD segmentation and acoustic feature extraction.

    Args:
        waveform: 1D float32 numpy array.
        target_dbfs: Target RMS energy in dBFS (default -20.0 dBFS).
        max_gain_db: Maximum amplification gain in dB to prevent background noise explosion (default 30.0 dB).
        min_rms_threshold: Minimum RMS threshold below which audio is treated as pure silence.
        eps: Small epsilon to prevent division by zero.

    Returns:
        RMS-normalized 1D float32 numpy array.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) == 0:
        return waveform

    current_rms = float(np.sqrt(np.mean(waveform ** 2)))
    if current_rms < min_rms_threshold:
        return waveform

    target_rms = 10.0 ** (target_dbfs / 20.0)
    raw_gain = target_rms / (current_rms + eps)

    max_gain = 10.0 ** (max_gain_db / 20.0)
    clamped_gain = min(raw_gain, max_gain)

    normalized = waveform * clamped_gain

    # Peak limiting guard to prevent digital clipping above 1.0
    peak = float(np.max(np.abs(normalized)))
    if peak > 1.0:
        normalized = normalized / peak

    return normalized.astype(np.float32)


# Dual-Threshold Energy VAD & Silence Trimming Defaults
# Retains 150ms acoustic buffer around speech to protect initial plosives (k, t, p) and word endings
DEFAULT_VAD_START_DBFS = -32.0
DEFAULT_VAD_STOP_DBFS = -42.0
DEFAULT_VAD_PAD_MS = 150.0
DEFAULT_VAD_MIN_SPEECH_MS = 200.0


def trim_silence_vad(
    waveform: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    vad_start_dbfs: float = DEFAULT_VAD_START_DBFS,
    vad_stop_dbfs: float = DEFAULT_VAD_STOP_DBFS,
    pad_ms: float = DEFAULT_VAD_PAD_MS,
    min_speech_duration_ms: float = DEFAULT_VAD_MIN_SPEECH_MS,
    return_indices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int, int]]:
    """Dual-threshold energy-based Voice Activity Detection and silence trimming.

    Removes dead silence before and after vocal commands while preserving
    initial plosive consonants and trailing word endings via safety padding buffers.

    Args:
        waveform: 1D float32 numpy array.
        sr: Sampling rate (16000 Hz).
        frame_ms: Short-term energy frame duration in ms (default 20.0 ms).
        hop_ms: Short-term energy hop size in ms (default 10.0 ms).
        vad_start_dbfs: Energy activation threshold in dBFS to detect speech onset (default -32.0 dBFS).
        vad_stop_dbfs: Lower energy continuation threshold in dBFS to track soft phonemes (default -42.0 dBFS).
        pad_ms: Pre- and post-padding buffer duration in ms to avoid chopping plosives (default 150.0 ms).
        min_speech_duration_ms: Minimum unpadded speech duration to filter out clicks/taps (default 200.0 ms).
        return_indices: If True, returns (trimmed_waveform, sample_start, sample_end). Else returns trimmed_waveform.

    Returns:
        Trimmed 1D float32 numpy array (or tuple with sample bounds if return_indices=True).
    """
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    frame_len = int(round(frame_ms * 1e-3 * sr))
    hop_len = int(round(hop_ms * 1e-3 * sr))
    pad_samples = int(round(pad_ms * 1e-3 * sr))
    min_speech_samples = int(round(min_speech_duration_ms * 1e-3 * sr))

    if len(waveform) <= frame_len or len(waveform) <= min_speech_samples:
        if return_indices:
            return waveform, 0, len(waveform)
        return waveform

    num_frames = 1 + (len(waveform) - frame_len) // hop_len
    if num_frames <= 0:
        if return_indices:
            return waveform, 0, len(waveform)
        return waveform

    shape = (num_frames, frame_len)
    strides = (waveform.strides[0] * hop_len, waveform.strides[0])
    frames = np.lib.stride_tricks.as_strided(waveform, shape=shape, strides=strides)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    frame_dbfs = 20.0 * np.log10(frame_rms)

    speech_starts = np.where(frame_dbfs >= vad_start_dbfs)[0]
    if len(speech_starts) == 0:
        # No speech detected above activation threshold; preserve signal safely
        if return_indices:
            return waveform, 0, len(waveform)
        return waveform

    first_start = speech_starts[0]
    idx = first_start
    while idx > 0 and frame_dbfs[idx - 1] >= vad_stop_dbfs:
        idx -= 1
    first_frame = idx

    last_start = speech_starts[-1]
    idx = last_start
    while idx < len(frame_dbfs) - 1 and frame_dbfs[idx + 1] >= vad_stop_dbfs:
        idx += 1
    last_frame = idx

    # Check unpadded active speech duration
    unpadded_samples = (last_frame - first_frame + 1) * hop_len
    if unpadded_samples < min_speech_samples:
        # Transient click or tap; fallback safely to untrimmed signal
        if return_indices:
            return waveform, 0, len(waveform)
        return waveform

    sample_start = max(0, first_frame * hop_len - pad_samples)
    sample_end = min(len(waveform), last_frame * hop_len + frame_len + pad_samples)

    trimmed = waveform[sample_start:sample_end]
    if return_indices:
        return trimmed, sample_start, sample_end
    return trimmed


def normalize_waveform(waveform: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Zero-mean, unit-variance per-utterance normalization.

    Used identically across wav2vec2 training and deployment serving.
    """
    mean = np.mean(waveform)
    std = np.std(waveform)
    return (waveform - mean) / (std + eps)


def preprocess_audio(
    audio_input: Any,
    sample_rate: Optional[int] = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    max_audio_seconds: Optional[float] = DEFAULT_MAX_AUDIO_SECONDS,
    normalize: bool = True,
    apply_bandpass: bool = True,
    apply_rms_norm: bool = True,
    target_rms_dbfs: float = DEFAULT_TARGET_RMS_DBFS,
    apply_vad_trim: bool = True,
    pad: bool = False,
    is_training: bool = False,
) -> Tuple[np.ndarray, int, int]:
    """Single shared audio preprocessing function for train, test, and live API serve.

    Args:
        audio_input: Filepath, bytes buffer, numpy array, or torch Tensor.
        sample_rate: Sampling rate of input if array/tensor passed.
        target_sample_rate: Target sampling rate (always 16000 Hz).
        max_audio_seconds: Maximum allowed duration. If None, no crop/padding.
        normalize: Whether to apply per-utterance zero-mean unit-variance.
        apply_bandpass: Whether to apply 80Hz-7.5kHz Butterworth bandpass filter.
        apply_rms_norm: Whether to apply Adaptive RMS energy normalization (-20 dBFS).
        target_rms_dbfs: Target RMS energy level in dBFS (default -20.0 dBFS).
        apply_vad_trim: Whether to apply Dual-Threshold Energy VAD & Silence Trimming (150ms buffer).
        pad: If True, pads short audio with zeros to max_audio_seconds * target_sample_rate.
        is_training: If True and cropping is required, perform random crop. Else deterministic crop from start.

    Returns:
        waveform: 1D float32 numpy array [samples].
        original_length: Number of samples before padding (useful for length-aware pooling).
        sample_rate: 16000.
    """
    # 1. Load audio
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file does not exist: {audio_input}")
        data, sr = sf.read(audio_input, dtype="float32")
    elif isinstance(audio_input, bytes):
        import io

        data, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
    elif HAS_TORCH and isinstance(audio_input, torch.Tensor):
        data = audio_input.detach().cpu().numpy().astype(np.float32)
        sr = sample_rate or target_sample_rate
    elif isinstance(audio_input, np.ndarray):
        data = audio_input.astype(np.float32)
        sr = sample_rate or target_sample_rate
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    # 2. Convert to Mono if stereo / multichannel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 3. Resample if necessary
    if sr != target_sample_rate:
        num_target_samples = int(round(len(data) * float(target_sample_rate) / sr))
        data = np.interp(
            np.linspace(0.0, 1.0, num_target_samples, endpoint=False),
            np.linspace(0.0, 1.0, len(data), endpoint=False),
            data,
        ).astype(np.float32)
        sr = target_sample_rate

    # 4. Butterworth Bandpass Filter (Vocal Formant Range: 80 Hz - 7500 Hz)
    if apply_bandpass:
        data = apply_butterworth_bandpass(data, sr=target_sample_rate)

    # 5. Adaptive RMS Energy Normalization (Target: -20 dBFS)
    if apply_rms_norm:
        data = adaptive_rms_normalize(data, target_dbfs=target_rms_dbfs)

    # 6. Dual-Threshold Energy VAD & Silence Trimming (150ms buffer)
    if apply_vad_trim:
        data = trim_silence_vad(data, sr=target_sample_rate)

    # 7. Normalization (per-utterance zero-mean unit-variance)
    if normalize:
        data = normalize_waveform(data)

    original_length = len(data)

    # 8. Crop / Pad to max_samples
    if max_audio_seconds is not None:
        max_samples = int(round(max_audio_seconds * target_sample_rate))
        if len(data) > max_samples:
            if is_training:
                # Random crop in training
                max_start = len(data) - max_samples
                start = np.random.randint(0, max_start + 1)
                data = data[start : start + max_samples]
            else:
                # Deterministic center or front crop in eval/serving
                data = data[:max_samples]
            original_length = max_samples
        elif pad and len(data) < max_samples:
            pad_width = max_samples - len(data)
            data = np.pad(data, (0, pad_width), mode="constant", constant_values=0.0)

    return data.astype(np.float32), original_length, target_sample_rate
