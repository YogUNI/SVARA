"""Unit tests for HF native baseline model (Task P3-17)."""

import pytest
import torch

from svara.models.hf_baseline import HFWav2Vec2ClassificationBaseline


def test_hf_baseline_init_and_forward():
    model = HFWav2Vec2ClassificationBaseline(
        num_labels=31,
        config_only=True,
    )
    model.eval()

    waveform = torch.randn(2, 32000)
    with torch.no_grad():
        out = model(waveform)

    assert "joint" in out
    assert "action" in out
    assert "object" in out
    assert "location" in out

    assert out["joint"].shape == (2, 31)
    assert out["action"].shape == (2, 6)
    assert out["object"].shape == (2, 14)
    assert out["location"].shape == (2, 4)


def test_hf_baseline_freeze_feature_encoder():
    model = HFWav2Vec2ClassificationBaseline(
        num_labels=31,
        freeze_feature_encoder=True,
        config_only=True,
    )
    for name, param in model.model.wav2vec2.feature_extractor.named_parameters():
        assert not param.requires_grad, f"Parameter {name} should be frozen"
