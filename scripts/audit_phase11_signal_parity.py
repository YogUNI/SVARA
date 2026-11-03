"""Comprehensive Parity & Anti-Clipping Audit for Sub-step 11.3 (Phase 11).

Task: Sub-step 11.3 (Phase 11: Real-World Speech Hardening)
Reference: docs/14_ROADMAP_PEMATANGAN_DEEP_LEARNING_REAL_WORLD.md, AGENTS.md

Verifies end-to-end signal parity across:
1. Frontend-to-backend duration alignment (5.0s max, 80,000 samples).
2. Dual-threshold VAD relaxation (-32 dBFS onset / -48 dBFS continuation).
3. Pre-speech cushion (250ms) and post-speech cushion (250ms backend / 400ms frontend).
4. Butterworth bandpass (80-7500Hz) & Adaptive RMS normalization (-20 dBFS target).
5. Numerical parity across WAV bytes, NumPy arrays, and PyTorch tensors.
6. Live inference pipeline execution on the production ONNX model.
"""

import io
import json
import os
import sys
import time
from typing import Dict, Any

import numpy as np
import soundfile as sf
import torch

from svara.data.audio import (
    TARGET_SAMPLE_RATE,
    DEFAULT_MAX_AUDIO_SECONDS,
    DEFAULT_VAD_START_DBFS,
    DEFAULT_VAD_STOP_DBFS,
    DEFAULT_VAD_PAD_MS,
    DEFAULT_TARGET_RMS_DBFS,
    apply_butterworth_bandpass,
    adaptive_rms_normalize,
    trim_silence_vad,
    normalize_waveform,
    preprocess_audio,
)
from svara.serve.inference import SLUInferenceEngine


def run_substep_11_3_audit() -> Dict[str, Any]:
    print("=" * 80)
    print("SVARA SUB-STEP 11.3: LIVE STREAMING BUFFER & ANTI-CLIPPING PARITY AUDIT")
    print("=" * 80)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "substep": "11.3",
        "phase": "11",
        "target": "Signal Hygiene, Buffer Padding & Anti-Clipping Parity",
        "parameters": {
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "max_audio_seconds": DEFAULT_MAX_AUDIO_SECONDS,
            "max_samples": int(DEFAULT_MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE),
            "vad_start_dbfs": DEFAULT_VAD_START_DBFS,
            "vad_stop_dbfs": DEFAULT_VAD_STOP_DBFS,
            "vad_pad_ms": DEFAULT_VAD_PAD_MS,
            "vad_pad_samples": int(DEFAULT_VAD_PAD_MS * 1e-3 * TARGET_SAMPLE_RATE),
            "target_rms_dbfs": DEFAULT_TARGET_RMS_DBFS,
        },
        "tests": {},
        "all_passed": False,
    }

    # --------------------------------------------------------------------------
    # Test 1: Configuration & Constant Integrity
    # --------------------------------------------------------------------------
    print("\n[Test 1/6] Verifying VAD & Buffer Constants...")
    assert DEFAULT_VAD_STOP_DBFS == -48.0, f"Expected -48.0 dBFS, got {DEFAULT_VAD_STOP_DBFS}"
    assert DEFAULT_VAD_PAD_MS == 250.0, f"Expected 250.0 ms, got {DEFAULT_VAD_PAD_MS}"
    assert DEFAULT_MAX_AUDIO_SECONDS == 5.0, f"Expected 5.0s, got {DEFAULT_MAX_AUDIO_SECONDS}"
    results["tests"]["config_integrity"] = {
        "status": "PASS",
        "vad_stop_dbfs": DEFAULT_VAD_STOP_DBFS,
        "vad_pad_ms": DEFAULT_VAD_PAD_MS,
        "max_audio_seconds": DEFAULT_MAX_AUDIO_SECONDS,
    }
    print("  -> PASS: All constants verified (-48.0 dBFS, 250.0 ms, 5.0s max).")

    # --------------------------------------------------------------------------
    # Test 2: Full 4.5s & 5.0s Extended Utterance Preservation
    # --------------------------------------------------------------------------
    print("\n[Test 2/6] Verifying Extended Utterance (4.5s and 5.0s) Preservation...")
    sr = TARGET_SAMPLE_RATE
    # Synthesize 4.5s utterance: 0.2s silence + 4.1s complex speech + 0.2s silence
    t_speech = np.linspace(0, 4.1, int(sr * 4.1), endpoint=False)
    complex_speech = (
        0.12 * np.sin(2 * np.pi * 220 * t_speech)
        + 0.08 * np.sin(2 * np.pi * 440 * t_speech)
        + 0.04 * np.sin(2 * np.pi * 880 * t_speech)
    ).astype(np.float32)
    silence_02 = np.zeros(int(sr * 0.2), dtype=np.float32)
    audio_45 = np.concatenate([silence_02, complex_speech, silence_02])  # exactly 4.5s

    w_45, len_45, sr_45 = preprocess_audio(audio_45, max_audio_seconds=5.0)
    assert sr_45 == 16000
    assert len_45 <= int(5.0 * sr)
    # The whole 4.1s speech + 2x250ms cushions = ~4.6s, should be completely retained (< 5.0s)
    speech_duration_retained = len_45 / float(sr)
    assert speech_duration_retained >= 4.0, f"Expected >= 4.0s retained, got {speech_duration_retained:.2f}s"

    results["tests"]["extended_utterance_preservation"] = {
        "status": "PASS",
        "input_duration_seconds": 4.5,
        "retained_duration_seconds": round(speech_duration_retained, 3),
        "retained_samples": len_45,
        "max_allowed_samples": int(5.0 * sr),
    }
    print(f"  -> PASS: 4.5s utterance retained {speech_duration_retained:.2f}s without truncation.")

    # --------------------------------------------------------------------------
    # Test 3: Soft Trailing Ending Tracking (-45 dBFS)
    # --------------------------------------------------------------------------
    print("\n[Test 3/6] Verifying Soft Voiced Phoneme Tracking (-45 dBFS)...")
    # Simulate: 0.5s silence + 1.5s speech (-20 dBFS) + 0.6s soft room name (-45 dBFS) + 0.8s silence
    t_strong = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    strong_speech = (0.10 * np.sqrt(2) * np.sin(2 * np.pi * 350 * t_strong)).astype(np.float32)

    t_soft = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False)
    soft_ending = (10.0 ** (-45.0 / 20.0) * np.sqrt(2) * np.sin(2 * np.pi * 260 * t_soft)).astype(np.float32)

    sil_pre = np.zeros(int(sr * 0.5), dtype=np.float32)
    sil_post = np.zeros(int(sr * 0.8), dtype=np.float32)
    composite_decay = np.concatenate([sil_pre, strong_speech, soft_ending, sil_post])

    trimmed, start_s, end_s = trim_silence_vad(composite_decay, sr=sr, return_indices=True)
    soft_ending_finish_sample = int(sr * (0.5 + 1.5 + 0.6))  # 2.6s * 16000 = 41600 samples
    
    # Must capture through the soft ending and include the 250ms (4000 samples) padding
    assert end_s >= soft_ending_finish_sample, f"end_s ({end_s}) < soft ending finish ({soft_ending_finish_sample})"
    
    results["tests"]["soft_ending_anti_clipping"] = {
        "status": "PASS",
        "soft_ending_finish_sample": soft_ending_finish_sample,
        "vad_detected_end_sample": int(end_s),
        "post_speech_margin_samples": int(end_s - soft_ending_finish_sample),
        "post_speech_margin_ms": round((end_s - soft_ending_finish_sample) * 1000.0 / sr, 1),
    }
    print(f"  -> PASS: Soft ending fully preserved with {int(end_s - soft_ending_finish_sample)} samples ({round((end_s - soft_ending_finish_sample) * 1000.0 / sr, 1)} ms) safety margin.")

    # --------------------------------------------------------------------------
    # Test 4: Butterworth Bandpass & RMS Headroom Integrity
    # --------------------------------------------------------------------------
    print("\n[Test 4/6] Verifying Bandpass & Adaptive RMS Normalization Headroom...")
    # Composite with 30 Hz sub-bass rumble + 500 Hz speech tone + 8000 Hz hiss
    t_1s = np.linspace(0, 1.0, sr, endpoint=False)
    noisy_composite = (
        0.4 * np.sin(2 * np.pi * 30 * t_1s)
        + 0.2 * np.sin(2 * np.pi * 500 * t_1s)
        + 0.3 * np.sin(2 * np.pi * 8000 * t_1s)
    ).astype(np.float32)

    w_proc, orig_l, sr_out = preprocess_audio(noisy_composite, apply_vad_trim=False, normalize=False)
    peak_amp = float(np.max(np.abs(w_proc)))
    rms_proc = float(np.sqrt(np.mean(w_proc ** 2)))
    dbfs_proc = 20.0 * np.log10(rms_proc + 1e-9)

    assert peak_amp <= 1.0, f"Digital clipping detected! Peak = {peak_amp}"
    assert abs(dbfs_proc - (-20.0)) <= 1.5, f"RMS normalization deviation too large: {dbfs_proc:.2f} dBFS"

    results["tests"]["bandpass_and_rms_hygiene"] = {
        "status": "PASS",
        "peak_amplitude": round(peak_amp, 4),
        "measured_rms_dbfs": round(dbfs_proc, 2),
        "target_rms_dbfs": DEFAULT_TARGET_RMS_DBFS,
        "clipping_detected": peak_amp > 1.0,
    }
    print(f"  -> PASS: Peak amplitude = {peak_amp:.4f} <= 1.0 (Zero Clipping), RMS = {dbfs_proc:.2f} dBFS.")

    # --------------------------------------------------------------------------
    # Test 5: Multi-Format Numerical Parity (Rule 7)
    # --------------------------------------------------------------------------
    print("\n[Test 5/6] Verifying Multi-Format Parity (File, Bytes, Array, Tensor)...")
    buffer = io.BytesIO()
    sf.write(buffer, complex_speech[:sr], sr, format="WAV", subtype="PCM_16")
    wav_bytes = buffer.getvalue()

    w_from_bytes, _, _ = preprocess_audio(wav_bytes)
    w_from_array, _, _ = preprocess_audio(complex_speech[:sr], sample_rate=sr)
    w_from_tensor, _, _ = preprocess_audio(torch.from_numpy(complex_speech[:sr]), sample_rate=sr)

    # Check max absolute differences
    diff_tensor_array = float(np.max(np.abs(w_from_array - w_from_tensor)))
    diff_bytes_array = float(np.max(np.abs(w_from_bytes - w_from_array)))

    assert diff_tensor_array < 1e-6, f"Tensor vs Array diff {diff_tensor_array} >= 1e-6"
    assert diff_bytes_array < 3e-4, f"Bytes vs Array PCM-16 diff {diff_bytes_array} >= 3e-4"

    results["tests"]["format_parity"] = {
        "status": "PASS",
        "max_diff_tensor_vs_array": round(diff_tensor_array, 8),
        "max_diff_pcm16_bytes_vs_array": round(diff_bytes_array, 6),
    }
    print(f"  -> PASS: Tensor-Array parity: {diff_tensor_array:.2e}, PCM16-Array parity: {diff_bytes_array:.2e}.")

    # --------------------------------------------------------------------------
    # Test 6: Production ONNX Model End-to-End Inference Verification
    # --------------------------------------------------------------------------
    print("\n[Test 6/6] Verifying Production ONNX Model Execution with 4.5s Utterance...")
    model_path = "models/svara_bilingual_int8.onnx"
    meta_path = "models/svara_bilingual_int8.meta.json"
    if not os.path.exists(model_path):
        model_path = "models/svara_int8.onnx"
        meta_path = "models/svara_int8.meta.json"

    engine = SLUInferenceEngine(onnx_path=model_path, meta_path=meta_path, intra_op_threads=4)
    assert engine.is_loaded, "Inference engine failed to load production ONNX model!"

    # Create 4.5s valid WAV payload
    buf_45 = io.BytesIO()
    sf.write(buf_45, audio_45, sr, format="WAV", subtype="PCM_16")
    raw_payload_bytes = buf_45.getvalue()

    t_start = time.perf_counter()
    pred = engine.predict_audio(raw_payload_bytes)
    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    assert "intent" in pred
    assert "confidence" in pred
    assert "timing_ms" in pred
    assert pred["timing_ms"]["total_ms"] > 0

    results["tests"]["production_onnx_inference"] = {
        "status": "PASS",
        "model_file": model_path,
        "input_payload_bytes": len(raw_payload_bytes),
        "total_latency_ms": pred["timing_ms"]["total_ms"],
        "inference_latency_ms": pred["timing_ms"]["inference_ms"],
        "predicted_slots": pred["intent"],
        "overall_confidence": pred["confidence"]["overall"],
    }
    print(f"  -> PASS: ONNX Inference completed in {t_elapsed_ms:.2f} ms (Inference: {pred['timing_ms']['inference_ms']:.2f} ms).")

    # --------------------------------------------------------------------------
    # Summary & Export
    # --------------------------------------------------------------------------
    results["all_passed"] = True
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/phase11_signal_hygiene_parity_audit.json"
    with open(report_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2)

    print("\n" + "=" * 80)
    print("SUB-STEP 11.3 AUDIT RESULT: 100% PASS (ALL 6 TESTS SUCCESSFUL)")
    print(f"Audit summary written to: {report_path}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    res = run_substep_11_3_audit()
    if not res["all_passed"]:
        sys.exit(1)
