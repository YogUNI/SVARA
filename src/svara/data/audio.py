"""Unified audio preprocessing module for training, evaluation, and inference.

Task: P2-05
Reference: AGENTS.md rule 7, docs/01 §1, docs/02 §1, docs/05 §1

RULE 7 (AGENTS.md):
"Keep train/serve preprocessing identical. Same normalization, sample rate (16 kHz mono), max length.
Share one function in src/svara/data/audio.py."
"""

import os
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

TARGET_SAMPLE_RATE = 16000
DEFAULT_MAX_AUDIO_SECONDS = 5.0
DEFAULT_MAX_SAMPLES = int(DEFAULT_MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE)  # 80,000 samples


def normalize_waveform(waveform: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Zero-mean, unit-variance per-utterance normalization.

    Used identically across wav2vec2 training and deployment serving.
    """
    mean = np.mean(waveform)
    std = np.std(waveform)
    return (waveform - mean) / (std + eps)


def preprocess_audio(
    audio_input: Union[str, bytes, np.ndarray, torch.Tensor],
    sample_rate: Optional[int] = None,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    max_audio_seconds: Optional[float] = DEFAULT_MAX_AUDIO_SECONDS,
    normalize: bool = True,
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
    elif isinstance(audio_input, torch.Tensor):
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
        # Simple high-quality linear/scipy resampling without heavy torchaudio dependence
        num_target_samples = int(round(len(data) * float(target_sample_rate) / sr))
        data = np.interp(
            np.linspace(0.0, 1.0, num_target_samples, endpoint=False),
            np.linspace(0.0, 1.0, len(data), endpoint=False),
            data,
        ).astype(np.float32)
        sr = target_sample_rate

    # 4. Normalization (per-utterance zero-mean unit-variance)
    if normalize:
        data = normalize_waveform(data)

    original_length = len(data)

    # 5. Crop / Pad to max_samples
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
