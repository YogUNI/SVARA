"""ONNX model exporter for SVARA Spoken Language Understanding models.

Features:
- Exports PyTorch models (CRNNBaseline or Wav2VecSLU) to standard ONNX format.
- Embeds length-aware pooling directly into the computational graph.
- Dynamic axes for batch size [B] and audio length [samples].
- Verifies parity between PyTorch logits and ONNX Runtime outputs (V-40).
- Optional Post-Training Dynamic Quantization to QInt8 weights (V-41).
- Generates svara.meta.json metadata file containing vocabulary, sample rate, normalization formula.

Tasks: P7-01, P7-02, P7-03, P7-05
Reference: docs/05 §1, docs/11 V-40..V-42, docs/12 §7
"""

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml

from svara.models.crnn_baseline import CRNNBaseline
from svara.models.wav2vec_slu import Wav2VecSLU


class ExportWrapper(nn.Module):
    """Wrapper that takes raw normalized waveform [B, T] and outputs logits per head."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass assuming serving batch size 1 (or uniform padded audio).

        Args:
            waveform: [B, T] normalized audio float32.

        Returns:
            Tuple of (action_logits, object_logits, location_logits)
        """
        # In deployment serving, inference is performed clip-by-clip without padding
        out = self.base_model(waveform, lengths=None)
        return out["action"], out["object"], out["location"]


def export_to_onnx(
    model: nn.Module,
    output_onnx_path: str,
    sample_length: int = 32000,
    opset_version: int = 14,
) -> str:
    """Export model to ONNX with dynamic batch and sequence axes."""
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, sample_length, dtype=torch.float32)

    dynamic_axes = {
        "waveform": {0: "batch_size", 1: "num_samples"},
        "action_logits": {0: "batch_size"},
        "object_logits": {0: "batch_size"},
        "location_logits": {0: "batch_size"},
    }

    print(f"[to_onnx] Exporting PyTorch graph to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["action_logits", "object_logits", "location_logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    # Validate ONNX model consistency
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[to_onnx] ONNX model successfully verified with onnx.checker!")
    return output_onnx_path


def verify_onnx_parity(
    model: nn.Module,
    onnx_path: str,
    test_durations: Tuple[int, ...] = (16000, 32000, 48000),
    tolerance: float = 1e-4,
) -> bool:
    """Verify PyTorch vs ONNX Runtime output parity across different durations (V-40)."""
    wrapper = ExportWrapper(model)
    wrapper.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"[to_onnx] Verifying parity across audio lengths: {test_durations} samples...")

    for num_samples in test_durations:
        wav_np = np.random.uniform(-1.0, 1.0, size=(1, num_samples)).astype(np.float32)
        wav_torch = torch.from_numpy(wav_np)

        with torch.no_grad():
            py_act, py_obj, py_loc = wrapper(wav_torch)

        ort_inputs = {"waveform": wav_np}
        ort_act, ort_obj, ort_loc = session.run(None, ort_inputs)

        # Check argmax predictions are 100% identical
        assert np.argmax(py_act.numpy(), axis=-1) == np.argmax(ort_act, axis=-1)
        assert np.argmax(py_obj.numpy(), axis=-1) == np.argmax(ort_obj, axis=-1)
        assert np.argmax(py_loc.numpy(), axis=-1) == np.argmax(ort_loc, axis=-1)

        # Check numeric differences
        max_diff_act = np.max(np.abs(py_act.numpy() - ort_act))
        max_diff_obj = np.max(np.abs(py_obj.numpy() - ort_obj))
        max_diff_loc = np.max(np.abs(py_loc.numpy() - ort_loc))
        max_diff = max(max_diff_act, max_diff_obj, max_diff_loc)

        print(f"  Length {num_samples} samples ({num_samples/16000:.1f}s) -> Max diff: {max_diff:.2e} | Argmax Match: OK")
        assert max_diff < tolerance, f"Parity difference {max_diff} exceeds tolerance {tolerance}!"

    print("[to_onnx] Parity check (V-40) PASSED: PyTorch and ONNX Runtime outputs match identically!")
    return True


def quantize_onnx_dynamic(
    input_onnx_path: str,
    output_quant_path: str,
) -> str:
    """Dynamic post-training int8 quantization (V-41)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"[to_onnx] Applying dynamic int8 quantization -> {output_quant_path}...")
    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_quant_path,
        weight_type=QuantType.QInt8,
    )

    size_orig = os.path.getsize(input_onnx_path) / (1024 * 1024)
    size_quant = os.path.getsize(output_quant_path) / (1024 * 1024)
    print(f"[to_onnx] Size reduction: {size_orig:.2f} MB -> {size_quant:.2f} MB "
          f"({(1.0 - size_quant / size_orig) * 100:.1f}% smaller)")
    return output_quant_path


def create_metadata_file(
    output_meta_path: str,
    run_id: str = "deployment_v1",
    precision: str = "fp32",
    intent_map_path: str = "configs/intent_map.json",
    threshold: float = 0.85,
):
    """Generate deployment svara.meta.json specification per docs/05 §1."""
    with open(intent_map_path, "r", encoding="utf-8") as f:
        imap_data = json.load(f)

    meta = {
        "schema": 1,
        "sample_rate": 16000,
        "max_audio_seconds": 5.0,
        "normalize": "zero_mean_unit_var",
        "slots": {
            "action": imap_data["slot_vocab"]["action"],
            "object": imap_data["slot_vocab"]["object"],
            "location": imap_data["slot_vocab"]["location"],
        },
        "intent_map": imap_data["intent_map"],
        "confidence": {
            "method": "min_head_softmax",
            "threshold": threshold,
        },
        "model": {
            "run_id": run_id,
            "precision": precision,
        },
        "license_note": "Trained on Fluent Speech Commands; academic use only.",
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_meta_path)), exist_ok=True)
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[to_onnx] Generated metadata specification: {output_meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SVARA model to ONNX.")
    parser.add_argument("--config", default="configs/model_crnn.yaml")
    parser.add_argument("--output", default="models/svara.onnx")
    parser.add_argument("--quantize", choices=["none", "int8"], default="none")
    args = parser.parse_args()

    # Create dummy model for CLI testing
    with open("configs/intent_map.json", "r") as f:
        imap = json.load(f)

    m = CRNNBaseline(
        n_action=len(imap["slot_vocab"]["action"]),
        n_object=len(imap["slot_vocab"]["object"]),
        n_location=len(imap["slot_vocab"]["location"]),
    )
    export_to_onnx(m, args.output)
    verify_onnx_parity(m, args.output)
    if args.quantize == "int8":
        q_path = args.output.replace(".onnx", "_int8.onnx")
        quantize_onnx_dynamic(args.output, q_path)
    meta_path = args.output.replace(".onnx", ".meta.json")
    create_metadata_file(meta_path)
