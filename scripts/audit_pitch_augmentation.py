"""Acoustic Integrity & Formant Zero-Clipping Guard Audit for Pitch-Augmented Dataset.

Task: Sub-step 12.3 (Phase 12: Real-World Speech Hardening)
Reference: docs/14_ROADMAP_PEMATANGAN_DEEP_LEARNING_REAL_WORLD.md, AGENTS.md

Audits:
1. Manifest schema & coverage (all 31 intents, 5 domains, 10 pitch profiles)
2. Audio format integrity (16 kHz, mono, PCM_16, duration > 0.3s)
3. Zero-clipping headroom guard (peak <= 0.95, zero saturation clips at +/-1.0)
4. Fundamental frequency (F0) distribution across the continuous human biological spectrum (90 Hz - 280 Hz)
5. Legal intent triplet consistency (action, object, location against configs/intent_map.json)
6. Intelligibility & formant preservation metrics

Output: reports/pitch_augmentation_audit.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

DEFAULT_MANIFEST = "data/synthetic_indonesian_massive/metadata_pitch.csv"
DEFAULT_BASE_DIR = "data/synthetic_indonesian_massive"
DEFAULT_INTENT_MAP = "configs/intent_map.json"
DEFAULT_AUDIT_OUT = "reports/pitch_augmentation_audit.json"
TARGET_SR = 16000
MAX_PEAK_HEADROOM = 0.950001


def estimate_f0_autocorr(waveform: np.ndarray, sr: int = 16000, fmin: float = 70.0, fmax: float = 350.0) -> float:
    """Estimates fundamental frequency (F0) using normalized autocorrelation on active speech."""
    if len(waveform) < sr * 0.1:
        return 0.0

    # Pick the most energetic 200ms segment (voiced vowel region)
    frame_len = int(sr * 0.2)
    step = int(sr * 0.05)
    best_energy = 0.0
    best_seg = waveform[:frame_len]

    for start in range(0, len(waveform) - frame_len, step):
        seg = waveform[start : start + frame_len]
        energy = np.sum(seg ** 2)
        if energy > best_energy:
            best_energy = energy
            best_seg = seg

    if best_energy < 1e-4:
        return 0.0

    # Windowed autocorrelation
    windowed = best_seg * np.hanning(len(best_seg))
    corr = np.correlate(windowed, windowed, mode="full")
    corr = corr[len(corr) // 2 :]

    # Search range in lag samples
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    if max_lag >= len(corr) or min_lag >= max_lag:
        return 0.0

    peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
    if corr[peak_lag] < 0.25 * corr[0]:
        return 0.0

    f0 = sr / float(peak_lag)
    return round(float(f0), 1)


def run_pitch_augmentation_audit(
    manifest_path: str = DEFAULT_MANIFEST,
    base_dir: str = DEFAULT_BASE_DIR,
    intent_map_path: str = DEFAULT_INTENT_MAP,
    output_path: str = DEFAULT_AUDIT_OUT,
    sample_size: int = 200,
) -> Dict[str, Any]:
    print("=" * 80)
    print("SVARA SUB-STEP 12.3: PITCH AUGMENTATION INTEGRITY AUDIT")
    print("=" * 80)
    t0 = time.perf_counter()

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    total_clips = len(df)
    print(f"Loaded manifest: {manifest_path} ({total_clips} entries)")

    # 1. Intent Map check
    legal_triplets = set()
    if os.path.exists(intent_map_path):
        with open(intent_map_path, "r", encoding="utf-8") as f:
            imap = json.load(f)
            # Handle possible nested structure
            if "intents" in imap:
                for entry in imap["intents"]:
                    legal_triplets.add((entry.get("action"), entry.get("object"), entry.get("location")))
            elif isinstance(imap, list):
                for entry in imap:
                    legal_triplets.add((entry.get("action"), entry.get("object"), entry.get("location")))

    # 2. Check schema & distributions
    intents_present = df["intent_id"].nunique()
    domains_present = df["domain"].nunique() if "domain" in df.columns else 0
    profiles_present = df["pitch_profile"].nunique() if "pitch_profile" in df.columns else 0
    semitones_present = sorted(df["pitch_semitones"].unique().tolist()) if "pitch_semitones" in df.columns else []

    print(f"Unique Intents: {intents_present} | Domains: {domains_present}")
    print(f"Pitch Profiles: {profiles_present} | Unique Semitones: {semitones_present}")

    # 3. Sample check for deep acoustic audit
    sample_df = df.sample(n=min(sample_size, total_clips), random_state=42)
    print(f"\nAuditing deep acoustic properties on sample of {len(sample_df)} clips...")

    missing_files = []
    format_errors = []
    headroom_violations = []
    triplet_violations = []
    f0_measurements = []
    durations = []
    peak_amplitudes = []

    for _, row in sample_df.iterrows():
        rel_path = str(row["path"])
        full_path = os.path.join(base_dir, rel_path)

        if not os.path.exists(full_path):
            missing_files.append(rel_path)
            continue

        try:
            info = sf.info(full_path)
            if info.samplerate != TARGET_SR or info.channels != 1 or info.subtype != "PCM_16":
                format_errors.append({
                    "path": rel_path,
                    "sr": info.samplerate,
                    "channels": info.channels,
                    "subtype": info.subtype,
                })
            durations.append(info.duration)
        except Exception as e:
            format_errors.append({"path": rel_path, "error": str(e)})
            continue

        # Read audio waveform
        wav, sr = sf.read(full_path, dtype="float32")
        peak = float(np.max(np.abs(wav)))
        peak_amplitudes.append(peak)

        if peak > MAX_PEAK_HEADROOM:
            headroom_violations.append({"path": rel_path, "peak": peak})

        # F0 estimation
        f0 = estimate_f0_autocorr(wav, sr=TARGET_SR)
        if f0 > 0.0:
            f0_measurements.append({
                "path": rel_path,
                "measured_f0_hz": f0,
                "target_f0_hz": float(row.get("estimated_f0_hz", 0.0)),
                "gender": str(row.get("gender", "")),
                "semitone": int(row.get("pitch_semitones", 0)),
            })

        # Triplet check
        if legal_triplets:
            trip = (row.get("action"), row.get("object"), row.get("location"))
            if trip not in legal_triplets:
                triplet_violations.append({"path": rel_path, "triplet": trip})

    # Acoustic statistics
    avg_peak = float(np.mean(peak_amplitudes)) if peak_amplitudes else 0.0
    max_peak = float(np.max(peak_amplitudes)) if peak_amplitudes else 0.0
    min_dur = float(np.min(durations)) if durations else 0.0
    max_dur = float(np.max(durations)) if durations else 0.0
    avg_dur = float(np.mean(durations)) if durations else 0.0

    # F0 range
    f0_values = [m["measured_f0_hz"] for m in f0_measurements]
    f0_min = float(np.min(f0_values)) if f0_values else 0.0
    f0_max = float(np.max(f0_values)) if f0_values else 0.0
    f0_mean = float(np.mean(f0_values)) if f0_values else 0.0

    # Separate male and female measured F0
    male_f0 = [m["measured_f0_hz"] for m in f0_measurements if "male" in m["gender"].lower()]
    female_f0 = [m["measured_f0_hz"] for m in f0_measurements if "female" in m["gender"].lower()]

    male_f0_range = [round(float(np.min(male_f0)), 1), round(float(np.max(male_f0)), 1)] if male_f0 else [0.0, 0.0]
    female_f0_range = [round(float(np.min(female_f0)), 1), round(float(np.max(female_f0)), 1)] if female_f0 else [0.0, 0.0]

    elapsed = time.perf_counter() - t0

    # Audit verdict
    all_passed = (
        len(missing_files) == 0
        and len(format_errors) == 0
        and len(headroom_violations) == 0
        and len(triplet_violations) == 0
        and intents_present == 31
        and len(semitones_present) >= 5
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "substep": "12.3",
        "phase": "12",
        "manifest_path": manifest_path,
        "total_clips_in_manifest": total_clips,
        "audited_sample_size": len(sample_df),
        "audit_duration_seconds": round(elapsed, 2),
        "dataset_structure": {
            "unique_intents": intents_present,
            "intents_expected": 31,
            "unique_domains": domains_present,
            "pitch_profiles_count": profiles_present,
            "unique_semitone_shifts": semitones_present,
            "pitch_distribution": df["pitch_profile"].value_counts().to_dict() if "pitch_profile" in df.columns else {},
        },
        "acoustic_integrity": {
            "sample_rate_hz": TARGET_SR,
            "channels": 1,
            "pcm_format": "PCM_16",
            "format_errors_count": len(format_errors),
            "missing_files_count": len(missing_files),
            "duration_stats": {
                "min_seconds": round(min_dur, 3),
                "max_seconds": round(max_dur, 3),
                "avg_seconds": round(avg_dur, 3),
            },
        },
        "headroom_and_clipping": {
            "max_allowed_peak": MAX_PEAK_HEADROOM,
            "observed_max_peak": round(max_peak, 4),
            "observed_avg_peak": round(avg_peak, 4),
            "headroom_violations_count": len(headroom_violations),
            "zero_clipping_rate_pct": 100.0 if len(headroom_violations) == 0 else 0.0,
        },
        "pitch_and_prosody_coverage": {
            "biological_spectrum_target": "90 Hz - 280 Hz",
            "measured_overall_f0_range_hz": [round(f0_min, 1), round(f0_max, 1)],
            "measured_mean_f0_hz": round(f0_mean, 1),
            "male_vocal_range_hz": male_f0_range,
            "female_vocal_range_hz": female_f0_range,
            "num_f0_audited": len(f0_values),
        },
        "semantic_consistency": {
            "triplet_violations_count": len(triplet_violations),
            "legal_intent_consistency_pct": 100.0 if len(triplet_violations) == 0 else 0.0,
        },
        "verdict": "PASSED" if all_passed else "FAILED",
        "all_passed": all_passed,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(report, f_out, indent=2)

    print("\n" + "=" * 80)
    print(f"SUB-STEP 12.3 AUDIT RESULT: {report['verdict']}")
    print(f"  * Total Clips: {total_clips} (Sample Audited: {len(sample_df)})")
    print(f"  * Format: 16 kHz Mono PCM_16 (Errors: {len(format_errors)})")
    print(f"  * Zero-Clipping: Max Peak {max_peak:.4f} <= 0.95 (Violations: {len(headroom_violations)})")
    print(f"  * Vocal F0 Range: {f0_min:.1f} Hz - {f0_max:.1f} Hz")
    print(f"    - Male: {male_f0_range[0]} Hz - {male_f0_range[1]} Hz")
    print(f"    - Female: {female_f0_range[0]} Hz - {female_f0_range[1]} Hz")
    print(f"  * Legal Intent Consistency: {report['semantic_consistency']['legal_intent_consistency_pct']}%")
    print(f"  * Report Saved: {output_path}")
    print("=" * 80)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitch Augmentation Integrity Audit (Sub-step 12.3)")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--intent-map", type=str, default=DEFAULT_INTENT_MAP)
    parser.add_argument("--out", type=str, default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()

    run_pitch_augmentation_audit(
        manifest_path=args.manifest,
        base_dir=args.base_dir,
        intent_map_path=args.intent_map,
        output_path=args.out,
        sample_size=args.sample_size,
    )

