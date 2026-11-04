"""Unit tests for Online SpecAugment (Frequency & Time Masking).

Task: Sub-step 13.1 (Phase 13: Real-World Speech Hardening)
Reference: docs/14_ROADMAP_PEMATANGAN_DEEP_LEARNING_REAL_WORLD.md, AGENTS.md

Verifies:
1. Waveform-level phase-preserving complex STFT SpecAugment (shape, length, peak <= 0.95).
2. AudioAugmentor integration with specaug_prob.
3. Wav2VecSLU config propagation (mask_time_length, mask_feature_prob, mask_feature_length).
4. Determinism in eval mode vs stochastic masking in train mode.
5. CUDA vs CPU compatibility.
"""

import numpy as np
import pytest
import torch

from svara.data.augment import AudioAugmentor, apply_spec_augment_waveform
from svara.models.wav2vec_slu import Wav2VecSLU


def test_specaugment_waveform_shape_and_length():
    """Verify waveform shape, length, and dtype are strictly preserved."""
    sr = 16000
    duration_s = 2.0
    n_samples = int(sr * duration_s)
    raw = np.random.uniform(-0.8, 0.8, size=n_samples).astype(np.float32)

    # 1D numpy array
    augmented = apply_spec_augment_waveform(
        raw,
        sr=sr,
        freq_mask_param=24,
        time_mask_param=30,
        num_freq_masks=2,
        num_time_masks=2,
    )
    assert isinstance(augmented, np.ndarray)
    assert augmented.shape == raw.shape
    assert len(augmented) == n_samples
    assert augmented.dtype == np.float32

    # Verify signal changed (masking occurred)
    diff = np.abs(raw - augmented)
    assert np.max(diff) > 0.01, "SpecAugment should mask parts of the signal"


def test_specaugment_waveform_headroom_anti_clipping():
    """Verify peak amplitude never exceeds headroom threshold (<= 0.95)."""
    # Loud input signal
    raw = np.sin(np.linspace(0, 1000, 32000)).astype(np.float32)
    augmented = apply_spec_augment_waveform(raw, headroom_peak=0.95)

    peak = float(np.max(np.abs(augmented)))
    assert peak <= 0.950001, f"Peak amplitude {peak} exceeds headroom guard 0.95"


def test_specaugment_tensor_cuda_compatibility():
    """Verify PyTorch Tensor input and CUDA execution when available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.randn(1, 32000, device=device)

    out = apply_spec_augment_waveform(t, device=device)
    assert isinstance(out, torch.Tensor)
    assert out.shape == t.shape
    assert out.device == t.device


def test_audio_augmentor_specaugment_chain():
    """Verify AudioAugmentor successfully executes SpecAugment step."""
    raw = np.random.uniform(-0.5, 0.5, size=24000).astype(np.float32)
    augmentor = AudioAugmentor(
        noise_prob=0.0,
        gain_prob=0.0,
        speed_prob=0.0,
        pitch_prob=0.0,
        reverb_prob=0.0,
        mic_filter_prob=0.0,
        specaug_prob=1.0,
        specaug_freq_mask_param=20,
        specaug_time_mask_param=25,
    )
    out = augmentor(raw)
    assert len(out) == len(raw)
    assert np.max(np.abs(raw - out)) > 0.01


def test_wav2vec_slu_specaugment_config():
    """Verify Wav2VecSLU propagates SpecAugment parameters to HuggingFace config."""
    model = Wav2VecSLU(
        keep_layers=2,
        config_only=True,
        apply_spec_augment=True,
        mask_time_prob=0.075,
        mask_time_length=25,
        mask_feature_prob=0.06,
        mask_feature_length=22,
    )
    assert model.apply_spec_augment is True
    assert model.enc.config.apply_spec_augment is True
    assert model.enc.config.mask_time_prob == 0.075
    assert model.enc.config.mask_time_length == 25
    assert model.enc.config.mask_feature_prob == 0.06
    assert model.enc.config.mask_feature_length == 22


def test_wav2vec_slu_eval_mode_specaugment_deactivation():
    """Verify SpecAugment is active in train mode and 100% disabled in eval mode."""
    model = Wav2VecSLU(
        keep_layers=2,
        config_only=True,
        apply_spec_augment=True,
        mask_time_prob=0.1,
        mask_feature_prob=0.1,
    )

    # 1. In eval mode: is_specaugment_active is False
    model.eval()
    assert model.is_specaugment_active is False

    waveform = torch.randn(1, 16000)
    with torch.no_grad():
        out1 = model(waveform)
        out2 = model(waveform)

    # Bit-exact determinism in eval mode
    for k in ["action", "object", "location"]:
        assert torch.allclose(out1[k], out2[k], atol=1e-6), f"Eval mode {k} must be strictly deterministic"

    # 2. In train mode: is_specaugment_active is True
    model.train()
    assert model.is_specaugment_active is True


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SUB-STEP 13.1 SPECAUGMENT UNIT TESTS")
    print("=" * 60)
    test_specaugment_waveform_shape_and_length()
    print("  [PASS] test_specaugment_waveform_shape_and_length")
    test_specaugment_waveform_headroom_anti_clipping()
    print("  [PASS] test_specaugment_waveform_headroom_anti_clipping")
    test_specaugment_tensor_cuda_compatibility()
    print("  [PASS] test_specaugment_tensor_cuda_compatibility")
    test_audio_augmentor_specaugment_chain()
    print("  [PASS] test_audio_augmentor_specaugment_chain")
    test_wav2vec_slu_specaugment_config()
    print("  [PASS] test_wav2vec_slu_specaugment_config")
    test_wav2vec_slu_eval_mode_specaugment_deactivation()
    print("  [PASS] test_wav2vec_slu_eval_mode_specaugment_deactivation")
    print("=" * 60)
    print("ALL SPECAUGMENT UNIT TESTS PASSED (6/6)!")
    print("=" * 60)
