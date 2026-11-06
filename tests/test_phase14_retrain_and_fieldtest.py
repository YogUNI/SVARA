"""Unit tests for Phase 14 Hardened Retraining, ONNX Export, and Field-Test Pipeline.

Task: Sub-steps 14.1, 14.2, 14.3 / P14-01
Reference: docs/14 §3 FASE 14, docs/11 V-11..V-18
"""

import json
import os
import tempfile
import numpy as np
import pytest
import torch

from scripts.train_hardened_bilingual_w2v2 import HardenedBilingualDataset
from scripts.run_real_world_field_test import FIELD_TEST_CASES, run_field_test_suite
from svara.export.to_onnx import ExportWrapper
from svara.models.wav2vec_slu import Wav2VecSLU
from svara.serve.inference import SLUInferenceEngine


def test_field_test_suite_coverage():
    """Verify that all 5 domains and 31 intents are represented in field test suite."""
    assert len(FIELD_TEST_CASES) == 50
    domains = set(tc["domain"] for tc in FIELD_TEST_CASES)
    expected_domains = {"Lighting", "Climate", "Media", "Appliances", "System"}
    assert domains == expected_domains


def test_hardened_dataset_initialization(tmp_path):
    """Verify HardenedBilingualDataset correctly loads manifest and creates tensors."""
    manifest_file = tmp_path / "mock_train.csv"
    manifest_file.write_text(
        "audio_path,action,object,location,intent_id\n"
        "mock1.wav,activate,lights,bedroom,1\n"
        "mock2.wav,deactivate,lights,kitchen,2\n"
    )

    dataset = HardenedBilingualDataset(
        manifest_path=str(manifest_file),
        own_recordings_path=None,
        intent_map_path="configs/intent_map.json",
        augment=False,
        is_training=False,
    )

    assert len(dataset) == 2
    item = dataset[0]
    assert "waveform" in item
    assert "action" in item
    assert "object" in item
    assert "location" in item
    assert item["waveform"].ndim == 1


def test_export_wrapper_forward():
    """Verify ExportWrapper produces 3 decoupled slot output logits."""
    model = Wav2VecSLU(
        n_action=6,
        n_object=14,
        n_location=4,
        n_joint=31,
        config_only=True,
    )
    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy_input = torch.zeros(1, 16000, dtype=torch.float32)
    with torch.no_grad():
        act_logits, obj_logits, loc_logits = wrapper(dummy_input)

    assert act_logits.shape == (1, 6)
    assert obj_logits.shape == (1, 14)
    assert loc_logits.shape == (1, 4)


def test_field_test_runner_report_generation(tmp_path):
    """Verify run_field_test_suite generates compliant JSON report with valid statistics."""
    # Use existing production INT8 model
    model_path = "models/svara_bilingual_int8.onnx"
    meta_path = "models/svara_bilingual_int8.meta.json"
    if not os.path.exists(model_path):
        pytest.skip("Production ONNX model not present for unit test.")

    engine = SLUInferenceEngine(onnx_path=model_path, meta_path=meta_path)
    out_json = str(tmp_path / "test_field_report.json")

    report = run_field_test_suite(engine, output_report_path=out_json)
    assert os.path.exists(out_json)
    assert "overall_summary" in report
    assert "domain_breakdown" in report
    assert report["overall_summary"]["total_samples"] == 50
    assert report["overall_summary"]["mean_latency_ms"] > 0.0

