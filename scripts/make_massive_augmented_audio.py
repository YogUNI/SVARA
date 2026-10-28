"""Massive Environmental Acoustic Augmentation Pipeline for Indonesian Speech Dataset.

Task: Sub-step 4.4 (Phase 4: Environmental Acoustic Augmentation)
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md

Generates 3 realistic environmental acoustic variations for every clean audio clip:
- Var 1 (Bedroom Slow): Bedroom Reverb (RT60=0.28s) + 100Hz Mic HPF + AC Noise (SNR=24dB) + 0.92x Tempo.
- Var 2 (Living Fast): Living Room Reverb (RT60=0.48s) + 100Hz Mic HPF + Fan Noise (SNR=18dB) + 1.08x Tempo.
- Var 3 (Kitchen Norm): Kitchen Reverb (RT60=0.36s) + 100Hz Mic HPF + AC/Fan Noise (SNR=21dB) + 1.00x Tempo.

Total output: 8,990 clean * 3 variations = 26,970 robust audio clips in wavs_augmented/
Manifest: data/synthetic_indonesian_massive/metadata_augmented.csv
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import pandas as pd

from svara.data.augment import (
    apply_schroeder_reverb,
    apply_laptop_mic_filter,
    inject_household_noise,
    apply_speed_perturbation,
)

DEFAULT_INPUT_CSV = "data/synthetic_indonesian_massive/metadata.csv"
DEFAULT_INPUT_DIR = "data/synthetic_indonesian_massive"
DEFAULT_OUTPUT_DIR = "data/synthetic_indonesian_massive/wavs_augmented"
DEFAULT_MANIFEST_OUT = "data/synthetic_indonesian_massive/metadata_augmented.csv"
DEFAULT_SUMMARY_JSON = "reports/massive_augmented_dataset_summary.json"
TARGET_SR = 16000

# 3 Canonical Environmental Acoustic Profiles
AUGMENTATION_PROFILES = [
    {
        "id": "var1_bedroom_slow",
        "name": "Kamar Tidur (Santai, AC Tenang)",
        "room_preset": "bedroom",
        "rt60": 0.28,
        "wet": 0.18,
        "mic_hpf_cutoff": 100.0,
        "noise_type": "ac",
        "target_snr_db": 24.0,
        "speed_factor": 0.92,
    },
    {
        "id": "var2_living_fast",
        "name": "Ruang Tamu (Cepat, Derau Kipas)",
        "room_preset": "living_room",
        "rt60": 0.48,
        "wet": 0.28,
        "mic_hpf_cutoff": 100.0,
        "noise_type": "fan",
        "target_snr_db": 18.0,
        "speed_factor": 1.08,
    },
    {
        "id": "var3_kitchen_norm",
        "name": "Dapur (Normal, AC & Kipas)",
        "room_preset": "kitchen",
        "rt60": 0.36,
        "wet": 0.22,
        "mic_hpf_cutoff": 100.0,
        "noise_type": "ac_fan",
        "target_snr_db": 21.0,
        "speed_factor": 1.00,
    },
]


def verify_wav(path: str, deep: bool = False) -> bool:
    """Verifies that a WAV file exists, is valid, non-empty, and readable."""
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        return False
    if not deep:
        return True
    try:
        info = sf.info(path)
        return info.samplerate == TARGET_SR and info.channels == 1 and info.duration > 0.3
    except Exception:
        return False


def process_single_audio_variations(
    row_dict: Dict[str, Any],
    input_base_dir: str,
    output_wav_dir: str,
) -> List[Dict[str, Any]]:
    """Generates 3 variations for a single clean audio file."""
    rel_clean_path = row_dict["path"]
    abs_clean_path = os.path.join(input_base_dir, rel_clean_path)

    if not os.path.exists(abs_clean_path):
        return []

    try:
        clean_audio, sr = sf.read(abs_clean_path, dtype="float32")
        if clean_audio.ndim > 1:
            clean_audio = np.mean(clean_audio, axis=1)
    except Exception:
        return []

    base_name = os.path.splitext(os.path.basename(rel_clean_path))[0]
    results = []

    for prof in AUGMENTATION_PROFILES:
        prof_id = prof["id"]
        out_filename = f"{base_name}_{prof_id}.wav"
        out_abs_path = os.path.join(output_wav_dir, out_filename)
        out_rel_path = f"wavs_augmented/{out_filename}".replace("\\", "/")

        # Check existing valid file for resumption
        if verify_wav(out_abs_path):
            info = sf.info(out_abs_path)
            results.append({
                "path": out_rel_path,
                "clean_source_path": rel_clean_path,
                "variation_id": prof_id,
                "room_preset": prof["room_preset"],
                "speed_factor": prof["speed_factor"],
                "noise_type": prof["noise_type"],
                "snr_db": prof["target_snr_db"],
                "speakerId": row_dict["speakerId"],
                "gender": row_dict["gender"],
                "voice_name": row_dict["voice_name"],
                "transcript": row_dict["transcript"],
                "action": row_dict["action"],
                "object": row_dict["object"],
                "location": row_dict["location"],
                "domain": row_dict.get("domain", "General"),
                "intent_id": int(row_dict["intent_id"]),
                "duration": round(info.duration, 3),
            })
            continue

        # Pipeline: Reverb -> Mic HPF 100 Hz -> Noise Injection -> Tempo Perturbation
        # 1. Schroeder Reverberator
        audio_rev = apply_schroeder_reverb(
            clean_audio,
            sample_rate=sr,
            room_preset=prof["room_preset"],
            rt60=prof["rt60"],
            wet=prof["wet"],
            headroom_peak=0.95,
        )

        # 2. Laptop Internal Mic Filter (Highpass 100 Hz)
        audio_mic = apply_laptop_mic_filter(
            audio_rev,
            sample_rate=sr,
            cutoff_hz=prof["mic_hpf_cutoff"],
        )

        # 3. Household AC / Fan Noise Injection (calibrated SNR)
        audio_noisy, _ = inject_household_noise(
            audio_mic,
            sample_rate=sr,
            snr_db=prof["target_snr_db"],
            noise_type=prof["noise_type"],
            headroom_peak=0.95,
        )

        # 4. Speech Tempo Perturbation
        audio_aug = apply_speed_perturbation(
            audio_noisy,
            factor=prof["speed_factor"],
            method="sinc",
            headroom_peak=0.95,
        )

        # Write to disk as 16-bit PCM Mono WAV
        sf.write(out_abs_path, audio_aug, TARGET_SR, subtype="PCM_16")

        duration = float(len(audio_aug) / TARGET_SR)
        results.append({
            "path": out_rel_path,
            "clean_source_path": rel_clean_path,
            "variation_id": prof_id,
            "room_preset": prof["room_preset"],
            "speed_factor": prof["speed_factor"],
            "noise_type": prof["noise_type"],
            "snr_db": prof["target_snr_db"],
            "speakerId": row_dict["speakerId"],
            "gender": row_dict["gender"],
            "voice_name": row_dict["voice_name"],
            "transcript": row_dict["transcript"],
            "action": row_dict["action"],
            "object": row_dict["object"],
            "location": row_dict["location"],
            "domain": row_dict.get("domain", "General"),
            "intent_id": int(row_dict["intent_id"]),
            "duration": round(duration, 3),
        })

    return results


def run_massive_augmentation(
    input_csv: str = DEFAULT_INPUT_CSV,
    input_base_dir: str = DEFAULT_INPUT_DIR,
    output_wav_dir: str = DEFAULT_OUTPUT_DIR,
    manifest_out_path: str = DEFAULT_MANIFEST_OUT,
    summary_json_path: str = DEFAULT_SUMMARY_JSON,
    workers: int = 8,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("   SVARA: GENERATOR AUGMENTASI AKUSTIK MASSAL (SUB-STEP 4.4)")
    print("=" * 75)
    print(f"File Manifest Bersih Asli     : {input_csv}")
    print(f"Direktori Output Audio Robust  : {output_wav_dir}")
    print(f"File Manifest Robust Baru      : {manifest_out_path}")
    print(f"Jumlah Profil Variasi          : {len(AUGMENTATION_PROFILES)} variasi per audio")
    print(f"Jumlah Worker Paralel          : {workers} processes")
    print("=" * 75 + "\n")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input manifest not found: {input_csv}")

    os.makedirs(output_wav_dir, exist_ok=True)
    df_clean = pd.read_csv(input_csv)
    if limit is not None and limit > 0:
        df_clean = df_clean.iloc[:limit]

    total_clean_files = len(df_clean)
    total_target_files = total_clean_files * len(AUGMENTATION_PROFILES)
    print(f"[*] Total file audio bersih yang akan diproses : {total_clean_files:,} audio")
    print(f"[*] Total file audio robust yang akan dibuat   : {total_target_files:,} audio")

    # Load existing manifest if resuming
    existing_records = []
    completed_keys = set()
    if os.path.exists(manifest_out_path):
        try:
            with open(manifest_out_path, "r", encoding="utf-8") as f_prev:
                reader = csv.DictReader(f_prev)
                for r in reader:
                    abs_p = os.path.join(input_base_dir, r["path"])
                    if verify_wav(abs_p):
                        r["duration"] = float(r["duration"])
                        r["intent_id"] = int(r["intent_id"])
                        r["speed_factor"] = float(r["speed_factor"])
                        r["snr_db"] = float(r["snr_db"])
                        existing_records.append(r)
                        completed_keys.add((r["clean_source_path"], r["variation_id"]))
            print(f"[*] Resuming from existing manifest: found {len(existing_records):,} existing valid files.")
        except Exception as e:
            print(f"[!] Warning reading existing manifest: {e}")

    rows_to_process = []
    for _, row in df_clean.iterrows():
        r_dict = row.to_dict()
        # Check if all 3 variations already exist
        all_done = all(
            (r_dict["path"], prof["id"]) in completed_keys
            for prof in AUGMENTATION_PROFILES
        )
        if not all_done:
            rows_to_process.append(r_dict)

    print(f"[*] Sisa file audio yang perlu diproses        : {len(rows_to_process):,} dari {total_clean_files:,}")

    start_time = time.time()
    fieldnames = [
        "path", "clean_source_path", "variation_id", "room_preset", "speed_factor",
        "noise_type", "snr_db", "speakerId", "gender", "voice_name", "transcript",
        "action", "object", "location", "domain", "intent_id", "duration"
    ]

    all_records = list(existing_records)
    processed_count = len(existing_records)

    def _save_manifest():
        tmp_p = manifest_out_path + ".tmp"
        with open(tmp_p, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        for attempt in range(5):
            try:
                os.replace(tmp_p, manifest_out_path)
                break
            except PermissionError:
                time.sleep(0.15 * (attempt + 1))
        else:
            import shutil
            shutil.copyfile(tmp_p, manifest_out_path)
            try:
                os.remove(tmp_p)
            except OSError:
                pass

    if rows_to_process:
        print(f"\n[*] Menjalankan pemrosesan paralel dengan {workers} thread workers...")
        last_save_time = time.time()
        last_report_count = processed_count
        chunk_report_interval = 1000

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_audio_variations, r, input_base_dir, output_wav_dir): r
                for r in rows_to_process
            }
            for fut in as_completed(futures):
                res_list = fut.result()
                for item in res_list:
                    k = (item["clean_source_path"], item["variation_id"])
                    if k not in completed_keys:
                        all_records.append(item)
                        completed_keys.add(k)
                        processed_count += 1

                if processed_count - last_report_count >= chunk_report_interval or processed_count == total_target_files:
                    now = time.time()
                    elapsed_chunk = now - last_save_time
                    clips_in_chunk = processed_count - last_report_count
                    rate = clips_in_chunk / elapsed_chunk if elapsed_chunk > 0 else 0
                    pct = (processed_count / total_target_files) * 100.0
                    remaining_clips = total_target_files - processed_count
                    eta_sec = remaining_clips / rate if rate > 0 else 0
                    eta_min = eta_sec / 60.0
                    print(
                        f"[*] Progress: {processed_count:,} / {total_target_files:,} audio ({pct:5.1f}%) "
                        f"| Speed: {rate:.1f} clips/s | ETA: {eta_min:.1f} min",
                        flush=True,
                    )
                    _save_manifest()
                    last_save_time = now
                    last_report_count = processed_count

    _save_manifest()
    total_elapsed = time.time() - start_time

    durations = [r["duration"] for r in all_records]
    total_hours = sum(durations) / 3600.0 if durations else 0.0

    print("\n\n" + "=" * 75)
    print("      RINGKASAN EKSEKUSI DATASET AUGMENTASI AKUSTIK MASSAL")
    print("=" * 75)
    print(f"Total Audio Robust Berhasil Disimpan : {len(all_records):,} / {total_target_files:,} file")
    print(f"Total Durasi Audio Robust            : {total_hours:.2f} jam ({sum(durations):,.1f} detik)")
    if durations:
        print(f"Rata-rata Durasi per Klip            : {np.mean(durations):.2f} detik (Min: {min(durations):.2f}s, Max: {max(durations):.2f}s)")
    print(f"Waktu Eksekusi Total                 : {total_elapsed:.2f} detik ({total_elapsed / 60:.2f} menit)")
    print(f"File Manifest Output                 : {manifest_out_path}")
    print("=" * 75 + "\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_clean_input_files": total_clean_files,
        "variations_per_file": len(AUGMENTATION_PROFILES),
        "total_augmented_files": len(all_records),
        "total_duration_hours": round(total_hours, 3),
        "mean_duration_seconds": round(float(np.mean(durations)), 3) if durations else 0.0,
        "elapsed_seconds": round(total_elapsed, 2),
        "manifest_path": manifest_out_path,
        "status": "PASS" if len(all_records) == total_target_files else "INCOMPLETE",
    }

    os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Massive Environmental Acoustic Augmentation Pipeline")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Clean dataset manifest CSV")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Base directory containing input audio")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for augmented WAVs")
    parser.add_argument("--manifest-out", default=DEFAULT_MANIFEST_OUT, help="Path for output manifest CSV")
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON, help="Path for summary JSON report")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clean files (for testing)")
    args = parser.parse_args()

    run_massive_augmentation(
        input_csv=args.input_csv,
        input_base_dir=args.input_dir,
        output_wav_dir=args.output_dir,
        manifest_out_path=args.manifest_out,
        summary_json_path=args.summary_json,
        workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
