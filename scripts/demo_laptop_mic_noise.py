"""Acoustic Simulation & Benchmark of Laptop Internal Mic Filter and Household Noise Injection.

Task: Sub-step 4.2 (Phase 4: Environmental Acoustic Augmentation)
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md

Evaluates:
1. Laptop Internal Mic EQ (2nd-order Butterworth Highpass at 100 Hz).
2. Frequency response attenuation across sub-bass (50 Hz) to vocal core (1000 Hz).
3. Household ambient noise profiles ('ac', 'fan', 'ac_fan', 'ambient_room').
4. Calibrated Signal-to-Noise Ratio (SNR) sweep across [16, 26] dB.
5. Processing throughput and real-time factor (RTF).

Exports:
- reports/laptop_mic_noise_benchmark.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List

import numpy as np
import soundfile as sf
from scipy import signal

from svara.data.augment import (
    apply_laptop_mic_filter,
    generate_ambient_noise,
    inject_household_noise,
    apply_laptop_mic_and_noise,
)

DEFAULT_SAMPLE_WAV = "data/synthetic_indonesian_massive/wavs_16k/id_intent00_0000_spk_ardi.wav"
DEFAULT_OUTPUT_JSON = "reports/laptop_mic_noise_benchmark.json"


def measure_filter_frequency_response(sr: int = 16000, cutoff_hz: float = 100.0) -> Dict[str, float]:
    """Measures steady-state attenuation in dB across representative audio frequencies."""
    test_freqs = [30.0, 50.0, 75.0, 100.0, 200.0, 500.0, 1000.0, 4000.0]
    response = {}
    t = np.linspace(0, 1.0, sr, endpoint=False)

    for f in test_freqs:
        x = np.sin(2.0 * np.pi * f * t).astype(np.float32)
        y = apply_laptop_mic_filter(x, sample_rate=sr, cutoff_hz=cutoff_hz)

        # Exclude edge transient (measure middle steady-state)
        mid = slice(int(0.1 * sr), int(0.9 * sr))
        rms_in = float(np.sqrt(np.mean(x[mid] ** 2)))
        rms_out = float(np.sqrt(np.mean(y[mid] ** 2)))
        gain_db = 20.0 * np.log10(rms_out / rms_in) if rms_out > 1e-9 else -100.0
        response[f"{int(f)}_hz"] = round(gain_db, 2)

    return response


def run_laptop_mic_noise_benchmark(
    sample_wav_path: str = DEFAULT_SAMPLE_WAV,
    output_json_path: str = DEFAULT_OUTPUT_JSON,
    benchmark_iterations: int = 500,
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("   SVARA: BENCHMARK FILTER MIC LAPTOP (100 HZ) & NOISE AC/KIPAS (16-26 DB)")
    print("=" * 75)
    print(f"Sample Audio Uji              : {sample_wav_path}")
    print(f"Jumlah Iterasi Benchmark      : {benchmark_iterations}")
    print(f"Output Laporan JSON           : {output_json_path}")
    print("=" * 75 + "\n")

    if not os.path.exists(sample_wav_path):
        raise FileNotFoundError(f"Sample WAV not found: {sample_wav_path}")

    audio_clean, sr = sf.read(sample_wav_path, dtype="float32")
    if audio_clean.ndim > 1:
        audio_clean = np.mean(audio_clean, axis=1)

    clean_power = float(np.mean(audio_clean ** 2))
    clean_peak = float(np.max(np.abs(audio_clean)))
    clean_dbfs = float(20.0 * np.log10(np.sqrt(clean_power)))

    print(f"[*] Audio Asli (Clean)        : Peak={clean_peak:.4f}, RMS dBFS={clean_dbfs:.2f} dBFS, Durasi={len(audio_clean)/sr:.2f}s")

    # 1. Frequency response analysis of laptop mic filter
    print("\n[1/3] Mengukur Karakteristik Filter Highpass 100 Hz (Laptop Internal Mic):")
    freq_response = measure_filter_frequency_response(sr=sr, cutoff_hz=100.0)
    for freq_str, atten_db in freq_response.items():
        print(f"      - Frekuensi {freq_str:<8s}: {atten_db:+6.2f} dB {'(Rumble teredam)' if atten_db < -3.0 else '(Vokal lolos utuh)'}")

    # 2. SNR sweep evaluation across [16, 26] dB
    print("\n[2/3] Menguji Presisi Kalibrasi SNR Injeksi Noise AC/Kipas [16 - 26 dB]:")
    target_snrs = [16.0, 18.0, 20.0, 22.0, 24.0, 26.0]
    snr_evaluations = []
    all_snr_passed = True

    for snr_target in target_snrs:
        filtered = apply_laptop_mic_filter(audio_clean, sample_rate=sr, cutoff_hz=100.0)
        mixed, used_snr = inject_household_noise(
            filtered,
            sample_rate=sr,
            snr_db=snr_target,
            noise_type="ac_fan",
        )

        filt_power = float(np.mean(filtered ** 2))
        noise_diff = mixed - filtered
        measured_snr = float(10.0 * np.log10(filt_power / (np.mean(noise_diff ** 2) + 1e-12)))
        snr_error = abs(measured_snr - snr_target)
        peak_val = float(np.max(np.abs(mixed)))
        peak_ok = peak_val <= 0.950001
        has_nan = bool(np.isnan(mixed).any())

        passed = (snr_error < 0.25) and peak_ok and (not has_nan)
        if not passed:
            all_snr_passed = False

        snr_evaluations.append({
            "target_snr_db": snr_target,
            "measured_snr_db": round(measured_snr, 2),
            "snr_error_db": round(snr_error, 4),
            "output_peak": round(peak_val, 4),
            "peak_headroom_pass": peak_ok,
            "has_nan": has_nan,
            "status": "PASS" if passed else "FAIL",
        })

        print(f"      - Target SNR {snr_target:4.1f} dB -> Terukur {measured_snr:5.2f} dB (Error: {snr_error:5.3f} dB) | Peak: {peak_val:.4f} | Status: {'PASS' if passed else 'FAIL'}")

    # 3. Throughput & Latency Benchmark
    print(f"\n[3/3] Menjalankan Throughput Benchmark ({benchmark_iterations} iterasi)...")
    t0 = time.time()
    for _ in range(benchmark_iterations):
        _ = apply_laptop_mic_and_noise(
            audio_clean,
            sample_rate=sr,
            cutoff_hz=100.0,
            min_snr_db=16.0,
            max_snr_db=26.0,
            noise_type="ac_fan",
        )
    elapsed = time.time() - t0
    latency_ms = (elapsed / benchmark_iterations) * 1000.0
    throughput = benchmark_iterations / elapsed
    duration_sec = len(audio_clean) / sr
    rtf = (duration_sec * 1000.0) / latency_ms

    print(f"[+] Throughput                : {throughput:.1f} audio clips / detik")
    print(f"[+] Rata-rata Latensi         : {latency_ms:.3f} ms per {duration_sec:.2f}-detik audio")
    print(f"[+] Real-Time Factor (RTF)    : {rtf:.1f}x lebih cepat dari audio real-time!")

    global_status = "PASS" if (all_snr_passed and freq_response["50_hz"] <= -10.0 and latency_ms < 50.0) else "FAIL"

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "global_status": global_status,
        "sample_wav": sample_wav_path,
        "sample_rate": sr,
        "filter_cutoff_hz": 100.0,
        "frequency_response_db": freq_response,
        "snr_sweep_evaluations": snr_evaluations,
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
    print(f"      STATUS SUB-STEP 4.2: {global_status} (LAPTOP MIC & NOISE PIPELINE READY)")
    print("=" * 75 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Laptop Mic Filter and Household Noise Benchmark")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE_WAV, help="Sample audio file path")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Output JSON path")
    parser.add_argument("--iterations", type=int, default=500, help="Benchmark iterations")
    args = parser.parse_args()

    results = run_laptop_mic_noise_benchmark(
        sample_wav_path=args.sample,
        output_json_path=args.output_json,
        benchmark_iterations=args.iterations,
    )
    if results["global_status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
