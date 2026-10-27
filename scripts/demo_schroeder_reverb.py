"""Acoustic Simulation & Benchmark of Schroeder Reverberator for Smart Home Rooms.

Task: Sub-step 4.1 (Phase 4: Environmental Acoustic Augmentation)
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md, Manfred Schroeder (1962)

Simulates acoustic reflections in residential environments:
- Bedroom preset (Kamar Tidur): RT60 = 0.28s, wet = 0.18
- Living Room preset (Ruang Tamu): RT60 = 0.48s, wet = 0.28
- Kitchen preset (Dapur): RT60 = 0.36s, wet = 0.22

Exports:
- reports/schroeder_reverb_benchmark.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any

import numpy as np
import soundfile as sf

from svara.data.augment import apply_schroeder_reverb, SCHROEDER_ROOM_PRESETS

DEFAULT_SAMPLE_WAV = "data/synthetic_indonesian_massive/wavs_16k/id_intent00_0000_spk_ardi.wav"
DEFAULT_OUTPUT_JSON = "reports/schroeder_reverb_benchmark.json"


def compute_audio_stats(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """Computes acoustic descriptors of audio signal."""
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
    dbfs = float(20.0 * np.log10(rms)) if rms > 1e-6 else -100.0

    # Spectral centroid calculation
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    spectral_centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-9))

    return {
        "duration_sec": round(len(audio) / sr, 3),
        "peak": round(peak, 5),
        "rms": round(rms, 5),
        "dbfs": round(dbfs, 2),
        "spectral_centroid_hz": round(spectral_centroid, 1),
    }


def run_schroeder_benchmark(
    sample_wav_path: str = DEFAULT_SAMPLE_WAV,
    output_json_path: str = DEFAULT_OUTPUT_JSON,
    benchmark_iterations: int = 500,
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("   SVARA: BENCHMARK & SIMULASI AKUSTIK SCHROEDER REVERBERATOR")
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

    clean_stats = compute_audio_stats(audio_clean, sr)
    print(f"[*] Audio Asli (Dry)          : Peak={clean_stats['peak']}, RMS={clean_stats['rms']} ({clean_stats['dbfs']} dBFS), Centroid={clean_stats['spectral_centroid_hz']} Hz")

    results: Dict[str, Any] = {
        "sample_wav": sample_wav_path,
        "sample_rate": sr,
        "clean_dry_stats": clean_stats,
        "room_evaluations": {},
        "throughput_benchmarks": {},
    }

    # Evaluate each room preset
    for room_name, room_cfg in SCHROEDER_ROOM_PRESETS.items():
        reverbed = apply_schroeder_reverb(audio_clean, sample_rate=sr, room_preset=room_name)
        stats = compute_audio_stats(reverbed, sr)

        # Numerical integrity checks
        has_nan = bool(np.isnan(reverbed).any())
        has_inf = bool(np.isinf(reverbed).any())
        peak_ok = stats["peak"] <= 0.950001

        results["room_evaluations"][room_name] = {
            "description": room_cfg["description"],
            "rt60_seconds": room_cfg["rt60"],
            "wet_ratio": room_cfg["wet"],
            "stats": stats,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "peak_headroom_pass": peak_ok,
            "status": "PASS" if (not has_nan and not has_inf and peak_ok) else "FAIL",
        }
        print(f"[*] Preset '{room_name:12s}': Peak={stats['peak']:<6.4f} | RMS={stats['rms']:<6.4f} ({stats['dbfs']:<5.1f} dBFS) | RT60={room_cfg['rt60']}s | Status={results['room_evaluations'][room_name]['status']}")

    # Performance Benchmark
    print(f"\n[*] Menjalankan throughput benchmark ({benchmark_iterations} iterasi)...")
    t0 = time.time()
    for _ in range(benchmark_iterations):
        _ = apply_schroeder_reverb(audio_clean, sample_rate=sr, room_preset="bedroom")
    elapsed_bedroom = time.time() - t0
    latency_ms = (elapsed_bedroom / benchmark_iterations) * 1000.0
    throughput_clips_per_sec = benchmark_iterations / elapsed_bedroom

    results["throughput_benchmarks"] = {
        "iterations": benchmark_iterations,
        "total_elapsed_sec": round(elapsed_bedroom, 3),
        "mean_latency_ms": round(latency_ms, 3),
        "throughput_clips_per_sec": round(throughput_clips_per_sec, 1),
        "rt_factor": round((clean_stats["duration_sec"] * 1000.0) / latency_ms, 1),
    }

    print(f"[+] Throughput                : {throughput_clips_per_sec:.1f} audio clips / detik")
    print(f"[+] Rata-rata Latensi         : {latency_ms:.3f} ms per 3-detik klip audio")
    print(f"[+] Real-Time Factor (RTF)    : {results['throughput_benchmarks']['rt_factor']}x lebih cepat dari audio real-time!")

    # Check overall pass
    all_pass = all(e["status"] == "PASS" for e in results["room_evaluations"].values())
    results["global_status"] = "PASS" if all_pass and latency_ms < 10.0 else "FAIL"

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Laporan benchmark JSON disimpan ke : {output_json_path}")
    print("=" * 75)
    print(f"      STATUS SUB-STEP 4.1: {results['global_status']} (SCHROEDER REVERB READY)")
    print("=" * 75 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Schroeder Reverberator Benchmark")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE_WAV, help="Sample audio file path")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Output JSON path")
    parser.add_argument("--iterations", type=int, default=500, help="Benchmark iterations")
    args = parser.parse_args()

    results = run_schroeder_benchmark(
        sample_wav_path=args.sample,
        output_json_path=args.output_json,
        benchmark_iterations=args.iterations,
    )
    if results["global_status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
