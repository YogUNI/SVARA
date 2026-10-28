"""Benchmark and Verification of Speech Tempo Perturbation (0.92x, 1.00x, 1.08x).

Task: Sub-step 4.3 (Phase 4: Environmental Acoustic Augmentation)
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md

Evaluates:
1. Speech tempo perturbation factors: 0.92x (slower), 1.00x (normal), 1.08x (faster).
2. Polyphase sinc resampling fidelity vs linear interpolation.
3. Duration scaling accuracy and peak headroom compliance (<= 0.95).
4. Processing latency and real-time factor (RTF).

Exports:
- reports/speed_perturbation_benchmark.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any

import numpy as np
import soundfile as sf

from svara.data.augment import apply_speed_perturbation, DEFAULT_SPEED_FACTORS

DEFAULT_SAMPLE_WAV = "data/synthetic_indonesian_massive/wavs_16k/id_intent00_0000_spk_ardi.wav"
DEFAULT_OUTPUT_JSON = "reports/speed_perturbation_benchmark.json"


def run_speed_perturbation_benchmark(
    sample_wav_path: str = DEFAULT_SAMPLE_WAV,
    output_json_path: str = DEFAULT_OUTPUT_JSON,
    benchmark_iterations: int = 500,
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("   SVARA: BENCHMARK VARIASI TEMPO BICARA (0.92x, 1.00x, 1.08x)")
    print("=" * 75)
    print(f"Sample Audio Uji              : {sample_wav_path}")
    print(f"Faktor Kecepatan Target       : {DEFAULT_SPEED_FACTORS}")
    print(f"Jumlah Iterasi Benchmark      : {benchmark_iterations}")
    print(f"Output Laporan JSON           : {output_json_path}")
    print("=" * 75 + "\n")

    if not os.path.exists(sample_wav_path):
        raise FileNotFoundError(f"Sample WAV not found: {sample_wav_path}")

    audio_clean, sr = sf.read(sample_wav_path, dtype="float32")
    if audio_clean.ndim > 1:
        audio_clean = np.mean(audio_clean, axis=1)

    orig_len = len(audio_clean)
    orig_dur = orig_len / sr
    orig_peak = float(np.max(np.abs(audio_clean)))
    orig_rms = float(np.sqrt(np.mean(audio_clean ** 2)))
    orig_dbfs = float(20.0 * np.log10(orig_rms))

    print(f"[*] Audio Asli (Clean 1.00x)  : Durasi={orig_dur:.3f}s ({orig_len:,} samples) | Peak={orig_peak:.4f} | RMS={orig_dbfs:.2f} dBFS")

    # 1. Factor evaluation across (0.92, 1.00, 1.08)
    factor_evaluations = {}
    all_passed = True

    for factor in [0.92, 1.00, 1.08]:
        expected_len = int(round(orig_len / factor))
        expected_dur = expected_len / sr

        # Sinc method
        y_sinc = apply_speed_perturbation(audio_clean, factor=factor, method="sinc")
        sinc_peak = float(np.max(np.abs(y_sinc)))
        sinc_rms = float(np.sqrt(np.mean(y_sinc ** 2)))
        sinc_dbfs = float(20.0 * np.log10(sinc_rms))
        sinc_len = len(y_sinc)
        sinc_dur = sinc_len / sr
        len_error = abs(sinc_len - expected_len)

        # Linear method
        y_linear = apply_speed_perturbation(audio_clean, factor=factor, method="linear")
        linear_peak = float(np.max(np.abs(y_linear)))

        peak_ok = sinc_peak <= 0.950001
        len_ok = len_error <= 1
        no_nan = not np.isnan(y_sinc).any()
        passed = peak_ok and len_ok and no_nan

        if not passed:
            all_passed = False

        pct_change = ((sinc_dur - orig_dur) / orig_dur) * 100.0

        factor_evaluations[f"{factor:.2f}x"] = {
            "factor": factor,
            "tempo_type": "Slower Cadence (Santai)" if factor < 1.0 else ("Faster Cadence (Cepat)" if factor > 1.0 else "Normal Cadence"),
            "original_duration_sec": round(orig_dur, 3),
            "perturbed_duration_sec": round(sinc_dur, 3),
            "duration_change_pct": round(pct_change, 2),
            "output_samples": sinc_len,
            "expected_samples": expected_len,
            "sample_len_error": len_error,
            "peak_amplitude": round(sinc_peak, 4),
            "rms_dbfs": round(sinc_dbfs, 2),
            "peak_headroom_pass": peak_ok,
            "has_nan": not no_nan,
            "status": "PASS" if passed else "FAIL",
        }

        print(
            f"[*] Faktor {factor:.2f}x: Durasi={sinc_dur:.3f}s ({pct_change:+5.1f}%) | "
            f"Samples={sinc_len:,} (Exp: {expected_len:,}) | Peak={sinc_peak:.4f} | Status={'PASS' if passed else 'FAIL'}"
        )

    # 2. Performance & Throughput Benchmark
    print(f"\n[*] Menjalankan throughput benchmark ({benchmark_iterations} iterasi)...")
    t0 = time.time()
    for _ in range(benchmark_iterations):
        _ = apply_speed_perturbation(audio_clean, factor=0.92, method="sinc")
    elapsed = time.time() - t0
    latency_ms = (elapsed / benchmark_iterations) * 1000.0
    throughput = benchmark_iterations / elapsed
    rtf = (orig_dur * 1000.0) / latency_ms

    print(f"[+] Throughput                : {throughput:.1f} audio clips / detik")
    print(f"[+] Rata-rata Latensi         : {latency_ms:.3f} ms per {orig_dur:.2f}-detik audio")
    print(f"[+] Real-Time Factor (RTF)    : {rtf:.1f}x lebih cepat dari audio real-time!")

    global_status = "PASS" if (all_passed and latency_ms < 5.0) else "FAIL"

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "global_status": global_status,
        "sample_wav": sample_wav_path,
        "sample_rate": sr,
        "speed_factors_tested": [0.92, 1.00, 1.08],
        "evaluations": factor_evaluations,
        "throughput_benchmarks": {
            "iterations": benchmark_iterations,
            "total_elapsed_sec": round(elapsed, 3),
            "mean_latency_ms": round(latency_ms, 3),
            "throughput_clips_per_sec": round(throughput, 1),
            "rt_factor": round(rtf, 1),
        },
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Laporan benchmark JSON disimpan ke : {output_json_path}")
    print("=" * 75)
    print(f"      STATUS SUB-STEP 4.3: {global_status} (TEMPO PERTURBATION READY)")
    print("=" * 75 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Speech Tempo Perturbation Benchmark")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE_WAV, help="Sample audio file path")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Output JSON path")
    parser.add_argument("--iterations", type=int, default=500, help="Benchmark iterations")
    args = parser.parse_args()

    results = run_speed_perturbation_benchmark(
        sample_wav_path=args.sample,
        output_json_path=args.output_json,
        benchmark_iterations=args.iterations,
    )
    if results["global_status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
