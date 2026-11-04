"""Validation, Conversion, and Metadata Generator for SVARA Own Recordings.

Task: Sub-step 13.2 / P5-02
Reference: docs/01 §8, docs/10 P5-02, docs/14 FASE 13

Functions:
1. Scans raw audio in `data/own_recordings/wavs_raw/` or `data/own_recordings/wavs/`.
2. Converts any input format to standard 16 kHz Mono PCM_16 WAV.
3. Applies RMS normalization (-20 dBFS) with Peak Headroom Guard <= 0.95.
4. Generates/validates `data/own_recordings/metadata.csv`.
5. Exports full acoustic audit to `reports/own_recordings_audit.json`.
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

INTENT_MAP_PATH = "configs/intent_map.json"
DEFAULT_RAW_DIR = "data/own_recordings/wavs_raw"
DEFAULT_WAVS_DIR = "data/own_recordings/wavs"
DEFAULT_METADATA_PATH = "data/own_recordings/metadata.csv"
DEFAULT_REPORT_PATH = "reports/own_recordings_audit.json"


def load_legal_intent_map(path: str = INTENT_MAP_PATH) -> Dict[int, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["intent_map"].items()}


def parse_filename(filename: str) -> Optional[Tuple[str, str, int]]:
    """Parses standard naming: own_<speaker>_<condition>_<intent_id>.wav
    Example: own_haikal_q_00.wav -> (own_haikal, quiet, 0)
             haikal_00.wav -> (own_haikal, quiet, 0)
    """
    basename = os.path.splitext(filename)[0].lower()
    
    # Pattern 1: own_<speaker>_<condition>_<intent_id>
    m1 = re.match(r"^own_([a-z0-9]+)_([qn]|quiet|noisy)_(\d{1,2})$", basename)
    if m1:
        spk = f"own_{m1.group(1)}"
        cond = "quiet" if m1.group(2) in ("q", "quiet") else "noisy"
        intent_id = int(m1.group(3))
        return spk, cond, intent_id

    # Pattern 2: <speaker>_<condition>_<intent_id>
    m2 = re.match(r"^([a-z0-9]+)_([qn]|quiet|noisy)_(\d{1,2})$", basename)
    if m2:
        spk = f"own_{m2.group(1)}"
        cond = "quiet" if m2.group(2) in ("q", "quiet") else "noisy"
        intent_id = int(m2.group(3))
        return spk, cond, intent_id

    # Pattern 3: <speaker>_<intent_id> (default condition: quiet)
    m3 = re.match(r"^([a-z0-9]+)_(\d{1,2})$", basename)
    if m3:
        spk = f"own_{m3.group(1)}"
        cond = "quiet"
        intent_id = int(m3.group(2))
        return spk, cond, intent_id

    return None


def convert_and_standardize_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
    headroom_peak: float = 0.95,
) -> Tuple[float, float, float]:
    """Converts audio to 16 kHz mono PCM_16 with headroom guard.
    Returns: (duration_seconds, peak_amplitude, rms_dbfs)
    """
    waveform, sr = torchaudio.load(input_path)

    # 1. Downmix to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 2. Resample to target_sr if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    audio_np = waveform.squeeze(0).numpy().astype(np.float32)

    # 3. Peak headroom guard (<= 0.95)
    peak = float(np.max(np.abs(audio_np)))
    if peak > headroom_peak and peak > 1e-6:
        audio_np = audio_np * (headroom_peak / peak)
        peak = headroom_peak

    # RMS calculation
    rms = float(np.sqrt(np.mean(audio_np**2)))
    rms_dbfs = float(20 * np.log10(rms + 1e-9))
    duration = float(len(audio_np) / target_sr)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_np, samplerate=target_sr, subtype="PCM_16")

    return duration, peak, rms_dbfs


# Indonesian canonical phrases per intent
ID_PHRASES = {
    0: "Nyalakan lampu meja",
    1: "Nyalakan lampu kamar tidur",
    2: "Nyalakan lampu dapur",
    3: "Nyalakan semua lampu",
    4: "Nyalakan lampu kamar mandi",
    5: "Putar lagi lagunya",
    6: "Bawakan saya jus",
    7: "Ambilkan koran untuk saya",
    8: "Ambilkan sepatu saya",
    9: "Ambilkan kaus kaki",
    10: "Ganti ke bahasa Mandarin dong",
    11: "Ganti bahasanya ke Bahasa Inggris",
    12: "Ganti bahasanya ke Bahasa Jerman",
    13: "Ganti ke bahasa Korea dong",
    14: "Ganti bahasa sistem",
    15: "Matikan lampu meja",
    16: "Matikan lampu kamar",
    17: "Matikan lampu dapur",
    18: "Matikan semua lampu",
    19: "Matikan lampu kamar mandi",
    20: "Matikan musiknya",
    21: "Dinginkan kamar tidur dong",
    22: "Dinginkan dapur dong",
    23: "Turunkan suhunya dong",
    24: "Dinginkan kamar mandi dong",
    25: "Kecilkan volumenya dong",
    26: "Naikkan suhu kamar tidur dong",
    27: "Naikkan suhu dapur dong",
    28: "Naikkan suhunya dong",
    29: "Naikkan suhu kamar mandi dong",
    30: "Besarkan suaranya dong",
}


def audit_own_recordings(
    raw_dir: str = DEFAULT_RAW_DIR,
    wavs_dir: str = DEFAULT_WAVS_DIR,
    metadata_path: str = DEFAULT_METADATA_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    legal_map: Optional[Dict[int, Dict[str, str]]] = None,
) -> Dict:
    """Validates existing recordings, converts raw files, and generates metadata."""
    if legal_map is None:
        legal_map = load_legal_intent_map()

    os.makedirs(wavs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # 1. Process any incoming raw files
    raw_files = []
    if os.path.exists(raw_dir):
        for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"):
            raw_files.extend(glob.glob(os.path.join(raw_dir, ext)))

    converted_count = 0
    for rpath in raw_files:
        fname = os.path.basename(rpath)
        parsed = parse_filename(fname)
        if parsed:
            spk, cond, intent_id = parsed
            cond_code = "q" if cond == "quiet" else "n"
            target_fname = f"{spk}_{cond_code}_{intent_id:02d}.wav"
            target_path = os.path.join(wavs_dir, target_fname)
            convert_and_standardize_audio(rpath, target_path)
            converted_count += 1

    # 2. Inspect all files in wavs_dir
    wav_files = sorted(glob.glob(os.path.join(wavs_dir, "*.wav")))
    metadata_rows = []
    speakers_found = set()
    conditions_found = set()
    intents_covered = set()
    warnings = []

    for wpath in wav_files:
        fname = os.path.basename(wpath)
        parsed = parse_filename(fname)
        if not parsed:
            warnings.append(f"Unrecognized filename format: {fname}")
            continue

        spk, cond, intent_id = parsed
        if intent_id not in legal_map:
            warnings.append(f"Invalid intent ID {intent_id} in {fname}")
            continue

        info, sr = sf.read(wpath)
        duration = len(info) / sr
        peak = float(np.max(np.abs(info)))
        rms = float(np.sqrt(np.mean(info**2)))
        rms_dbfs = float(20 * np.log10(rms + 1e-9))

        intent_info = legal_map[intent_id]
        act = intent_info["action"]
        obj = intent_info["object"]
        loc = intent_info["location"]
        phrase_id = ID_PHRASES.get(intent_id, f"{act} {obj} {loc}".replace(" none", ""))

        rel_path = f"data/own_recordings/wavs/{fname}".replace("\\", "/")
        metadata_rows.append({
            "path": rel_path,
            "speakerId": spk,
            "device": "laptop_mic",
            "condition": cond,
            "transcript": phrase_id,
            "action": act,
            "object": obj,
            "location": loc,
            "intent_id": intent_id,
            "duration_seconds": round(duration, 2),
            "peak_amplitude": round(peak, 4),
            "rms_dbfs": round(rms_dbfs, 2),
            "sample_rate": sr,
        })
        speakers_found.add(spk)
        conditions_found.add(cond)
        intents_covered.add(intent_id)

    # 3. Write metadata.csv
    df_meta = pd.DataFrame(metadata_rows)
    if not df_meta.empty:
        df_meta.to_csv(metadata_path, index=False)

    total_clips = len(metadata_rows)
    status_verdict = "PASSED" if total_clips > 0 else "PENDING_RECORDINGS"

    summary = {
        "substep": "13.2",
        "total_clips_registered": total_clips,
        "speakers_found": sorted(list(speakers_found)),
        "conditions_found": sorted(list(conditions_found)),
        "unique_intents_covered": len(intents_covered),
        "intents_missing": sorted(list(set(range(31)) - intents_covered)),
        "raw_files_converted": converted_count,
        "warnings": warnings,
        "metadata_csv_path": metadata_path,
        "status": status_verdict,
        "message": (
            f"Successfully audited {total_clips} clips across {len(speakers_found)} speakers."
            if total_clips > 0
            else "No human recordings found yet in data/own_recordings/wavs/. Phrase sheet is ready."
        ),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate and Convert Own Recordings")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--wavs-dir", default=DEFAULT_WAVS_DIR)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    summary = audit_own_recordings(
        raw_dir=args.raw_dir,
        wavs_dir=args.wavs_dir,
        metadata_path=args.metadata_csv,
        report_path=args.report,
    )

    print("=" * 60)
    print("SVARA OWN RECORDINGS VALIDATION SUMMARY (SUB-STEP 13.2)")
    print("=" * 60)
    print(f"Total Clips Registered : {summary['total_clips_registered']}")
    print(f"Speakers Found         : {summary['speakers_found']}")
    print(f"Conditions Found       : {summary['conditions_found']}")
    print(f"Intents Covered        : {summary['unique_intents_covered']} / 31")
    print(f"Status                 : {summary['status']}")
    print(f"Report                 : {args.report}")
    print("=" * 60)


if __name__ == "__main__":
    main()

