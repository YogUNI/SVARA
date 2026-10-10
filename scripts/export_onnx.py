"""CLI export script for SVARA model to ONNX with int8 dynamic quantization and parity check.

Usage:
    python scripts/export_onnx.py --run reports/runs/20261009-1953_wav2vec2_A_s42 --quantize int8
"""

import argparse
import json
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

from svara.export.to_onnx import ExportWrapper, create_metadata_file
from svara.models.crnn_baseline import CRNNBaseline
from svara.models.wav2vec_slu import Wav2VecSLU


def load_model_from_run(run_dir: str):
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open("configs/intent_map.json", "r", encoding="utf-8") as f:
        imap = json.load(f)

    n_act = len(imap["slot_vocab"]["action"])
    n_obj = len(imap["slot_vocab"]["object"])
    n_loc = len(imap["slot_vocab"]["location"])
    m_type = cfg["model"].get("type", "crnn")

    if m_type == "wav2vec2":
        model = Wav2VecSLU(
            pretrained_model_name_or_path=cfg["model"].get("pretrained_model", "facebook/wav2vec2-base"),
            n_action=n_act,
            n_object=n_obj,
            n_location=n_loc,
            n_joint=31 if cfg["model"].get("use_joint_head", False) else None,
            keep_layers=cfg["model"].get("keep_layers", None),
            dropout=cfg["model"].get("dropout", 0.1),
        )
    else:
        model = CRNNBaseline(
            n_action=n_act,
            n_object=n_obj,
            n_location=n_loc,
            n_joint=31 if cfg["model"].get("use_joint_head", False) else None,
        )

    ckpt_path = os.path.join(run_dir, "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg, imap


def export_onnx(model, output_path: str, opset_version: int = 14):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 32000, dtype=torch.float32)
    dynamic_axes = {
        "waveform": {0: "batch_size", 1: "num_samples"},
        "action_logits": {0: "batch_size"},
        "object_logits": {0: "batch_size"},
        "location_logits": {0: "batch_size"},
    }

    print(f"[export_onnx] Exporting PyTorch graph to ONNX: {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["action_logits", "object_logits", "location_logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[export_onnx] ONNX model valid and verified!")


def verify_parity(model, onnx_path: str):
    wrapper = ExportWrapper(model)
    wrapper.eval()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print("[export_onnx] Running parity check across multiple audio durations...")

    durations = [16000, 32000, 48000]  # 1s, 2s, 3s
    for num_samples in durations:
        wav_np = np.random.uniform(-1.0, 1.0, size=(1, num_samples)).astype(np.float32)
        wav_torch = torch.from_numpy(wav_np)

        with torch.no_grad():
            py_act, py_obj, py_loc = wrapper(wav_torch)

        ort_act, ort_obj, ort_loc = session.run(None, {"waveform": wav_np})

        assert np.argmax(py_act.numpy(), axis=-1) == np.argmax(ort_act, axis=-1)
        assert np.argmax(py_obj.numpy(), axis=-1) == np.argmax(ort_obj, axis=-1)
        assert np.argmax(py_loc.numpy(), axis=-1) == np.argmax(ort_loc, axis=-1)

        diff = max(
            np.max(np.abs(py_act.numpy() - ort_act)),
            np.max(np.abs(py_obj.numpy() - ort_obj)),
            np.max(np.abs(py_loc.numpy() - ort_loc)),
        )
        print(f"  Duration {num_samples/16000:.1f}s ({num_samples} samples): Max diff = {diff:.2e} | Argmax Match = OK")

    print("[export_onnx] Parity check PASSED 100%!")


def quantize_int8(input_onnx_path: str, output_quant_path: str):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"[export_onnx] Applying dynamic INT8 quantization -> {output_quant_path}...")
    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_quant_path,
        weight_type=QuantType.QInt8,
    )
    s_orig = os.path.getsize(input_onnx_path) / (1024 * 1024)
    s_q = os.path.getsize(output_quant_path) / (1024 * 1024)
    print(f"[export_onnx] Model size reduced: {s_orig:.2f} MB -> {s_q:.2f} MB ({(1 - s_q/s_orig)*100:.1f}% reduction)!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SVARA model to ONNX")
    parser.add_argument("--run", required=True, help="Path to run directory")
    parser.add_argument("--output", default="models/svara.onnx", help="Target ONNX path")
    parser.add_argument("--quantize", choices=["none", "int8"], default="none")
    args = parser.parse_args()

    print(f"[export_onnx] Loading model from {args.run}...")
    model, cfg, imap = load_model_from_run(args.run)

    export_onnx(model, args.output)
    verify_parity(model, args.output)

    meta_path = args.output.replace(".onnx", ".meta.json")
    create_metadata_file(
        output_meta_path=meta_path,
        run_id=os.path.basename(os.path.normpath(args.run)),
        precision="fp32" if args.quantize == "none" else "int8",
    )

    if args.quantize == "int8":
        q_path = args.output.replace(".onnx", "_int8.onnx")
        quantize_int8(args.output, q_path)
        create_metadata_file(
            output_meta_path=q_path.replace(".onnx", ".meta.json"),
            run_id=os.path.basename(os.path.normpath(args.run)),
            precision="int8",
        )
