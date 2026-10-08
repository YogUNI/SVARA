"""Unit tests for Wav2VecSLU model architecture and preprocessing parity.

Tasks: P3-01, P3-02, P3-18
Reference: docs/12 §2-§4, docs/11 V-11..V-18
"""

import numpy as np
import pytest
import torch
from transformers import Wav2Vec2FeatureExtractor

from svara.data.audio import normalize_waveform, preprocess_audio
from svara.models.wav2vec_slu import Wav2VecSLU


def test_wav2vec_slu_init_and_forward_shapes():
    """Verify forward pass output keys and tensor shapes on synthetic input."""
    batch_size = 2
    samples = 32000  # 2 seconds at 16 kHz
    waveform = torch.randn(batch_size, samples)
    lengths = torch.tensor([32000, 24000], dtype=torch.long)

    model = Wav2VecSLU(
        n_action=6,
        n_object=14,
        n_location=4,
        n_joint=31,
        keep_layers=2,  # fast test with 2 layers
        config_only=True,
    )
    model.eval()

    with torch.no_grad():
        out = model(waveform, lengths=lengths)

    assert "action" in out
    assert "object" in out
    assert "location" in out
    assert "joint" in out

    assert out["action"].shape == (batch_size, 6)
    assert out["object"].shape == (batch_size, 14)
    assert out["location"].shape == (batch_size, 4)
    assert out["joint"].shape == (batch_size, 31)


def test_wav2vec_slu_feature_encoder_frozen():
    """Verify temporal CNN feature encoder parameters have requires_grad=False (V-15)."""
    model = Wav2VecSLU(
        keep_layers=2,
        freeze_feature_encoder=True,
        config_only=True,
    )
    # Check conv layers parameters in feature_extractor
    for name, param in model.enc.feature_extractor.named_parameters():
        assert not param.requires_grad, f"Parameter {name} should be frozen"

    # Transformer layers should remain trainable by default
    for name, param in model.enc.encoder.layers.named_parameters():
        assert param.requires_grad, f"Transformer parameter {name} should be trainable"


def test_wav2vec_slu_freeze_encoder_e2():
    """Verify E2 configuration freezes all encoder parameters and eval mode behavior."""
    model = Wav2VecSLU(
        keep_layers=2,
        freeze_encoder=True,
        config_only=True,
    )
    for name, param in model.enc.named_parameters():
        assert not param.requires_grad, f"Encoder parameter {name} should be frozen in E2"

    # Heads must still be trainable
    for name, param in model.heads.named_parameters():
        assert param.requires_grad, f"Head parameter {name} must be trainable"

    # In train mode, encoder must stay in eval mode
    model.train()
    assert not model.enc.training, "Frozen encoder must remain in eval mode even when model.train() is called"


def test_wav2vec_slu_layer_truncation():
    """Verify layer truncation (P4-02 / E5)."""
    model = Wav2VecSLU(
        keep_layers=4,
        config_only=True,
    )
    assert len(model.enc.encoder.layers) == 4
    assert model.enc.config.num_hidden_layers == 4


def test_wav2vec_slu_eval_determinism():
    """Verify deterministic forward output in eval mode (V-17)."""
    waveform = torch.randn(1, 16000)
    model = Wav2VecSLU(keep_layers=2, config_only=True)
    model.eval()

    with torch.no_grad():
        out1 = model(waveform)
        out2 = model(waveform)

    for k in ["action", "object", "location"]:
        assert torch.allclose(out1[k], out2[k], atol=1e-6)


def test_preprocessing_parity_with_hf_feature_extractor():
    """Verify normalization parity against Wav2Vec2FeatureExtractor (P3-18 / docs/12 §4.6)."""
    np.random.seed(42)
    raw_audio = np.random.uniform(-1.0, 1.0, size=16000).astype(np.float32)

    # 1. SVARA normalization
    svara_norm = normalize_waveform(raw_audio)

    # 2. HuggingFace Wav2Vec2FeatureExtractor
    hf_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, do_normalize=True)
    hf_res = hf_extractor(raw_audio, sampling_rate=16000, return_tensors="np").input_values[0]

    # Verify difference is within floating point tolerance
    max_diff = np.max(np.abs(svara_norm - hf_res))
    assert max_diff < 1e-4, f"Normalization mismatch with HF: max diff {max_diff}"
