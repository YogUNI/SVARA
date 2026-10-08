"""Unit tests for ONNX export, parity check, quantization, and metadata creation.

Tasks: P7-01, P7-02, P7-03, P7-05
Reference: docs/05 §1, docs/11 V-40..V-41
"""

import os
import tempfile
import pytest
import torch

from svara.export.to_onnx import (
    create_metadata_file,
    export_to_onnx,
    quantize_onnx_dynamic,
    verify_onnx_parity,
)
from svara.models.wav2vec_slu import Wav2VecSLU


def test_onnx_export_and_parity_wav2vec_slu():
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = os.path.join(tmpdir, "test_w2v2.onnx")
        model = Wav2VecSLU(
            n_action=6,
            n_object=14,
            n_location=4,
            keep_layers=2,
            config_only=True,
        )

        # 1. Export to ONNX (P7-01)
        res_path = export_to_onnx(model, onnx_file, sample_length=16000, opset_version=14)
        assert os.path.exists(res_path)

        # 2. Verify parity across different lengths: 1s (16k) and 2s (32k) (P7-02 / V-40)
        ok = verify_onnx_parity(model, onnx_file, test_durations=(16000, 32000), tolerance=1e-3)
        assert ok

        # 3. Dynamic int8 quantization (P7-03 / V-41)
        quant_file = os.path.join(tmpdir, "test_w2v2_int8.onnx")
        q_path = quantize_onnx_dynamic(onnx_file, quant_file)
        assert os.path.exists(q_path)

        # 4. Metadata creation (P7-05)
        meta_file = os.path.join(tmpdir, "test_w2v2.meta.json")
        create_metadata_file(meta_file, run_id="test_run", precision="int8")
        assert os.path.exists(meta_file)
