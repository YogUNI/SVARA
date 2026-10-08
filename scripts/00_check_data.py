"""Comprehensive dataset audit script for Fluent Speech Commands (FSC).

Performs all 11 audit checks from docs/01 §2 and deep learning verification checklist (docs/11 V-01..V-08).
Outputs: reports/data_audit.json and configs/intent_map.json

Task: P1-02, P1-03, P1-04
Reference: docs/01 §2, docs/11 §2
"""

import argparse
import glob
import hashlib
import json
import os
import wave

import numpy as np
import pandas as pd

from svara.data.fsc import (
    FSCPaths,
    analyze_split_overlaps,
    extract_intent_vocabularies,
    load_fsc_splits,
)


def compute_file_hash(filepath: str, block_size: int = 65536) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def run_audit(dataset_root: str, sample_audio_count: int = 50) -> dict:
    paths = FSCPaths.from_root(dataset_root)
    print(f"[00_check_data] Auditing dataset at: {paths.root_dir}")

    # Check 1: Existence of root and CSVs, print real headers
    csv_exists = {
        "train_csv": os.path.exists(paths.train_csv),
        "valid_csv": os.path.exists(paths.valid_csv),
        "test_csv": os.path.exists(paths.test_csv),
        "demographics_csv": os.path.exists(paths.demographics_csv),
        "license_pdf": os.path.exists(paths.license_pdf),
        "readme_md": os.path.exists(paths.readme_md),
    }

    raw_headers = {}
    for name, p in [
        ("train", paths.train_csv),
        ("valid", paths.valid_csv),
        ("test", paths.test_csv),
        ("demographics", paths.demographics_csv),
    ]:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                raw_headers[name] = f.readline().strip()

    train_df, valid_df, test_df, demo_df = load_fsc_splits(paths.root_dir)

    # Check 2: Row counts vs paper target
    counts = {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
        "total": len(train_df) + len(valid_df) + len(test_df),
        "demographics": len(demo_df),
    }
    paper_expected = {"train": 23132, "valid": 3118, "test": 3793, "total": 30043}
    count_diffs = {k: counts[k] - paper_expected[k] for k in paper_expected}

    # Check 3: Check existence of every audio path on disk
    all_splits_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    all_wavs_on_disk = {
        os.path.normpath(p)
        for p in glob.glob(os.path.join(paths.wavs_dir, "*", "*.wav"))
    }

    missing_files = []
    for abs_p in all_splits_df["abs_path"]:
        if abs_p not in all_wavs_on_disk:
            missing_files.append(abs_p)

    # Check 4: Audio sanity stats on sample
    np.random.seed(42)
    sample_paths = np.random.choice(
        list(all_wavs_on_disk),
        size=min(sample_audio_count, len(all_wavs_on_disk)),
        replace=False,
    )

    sample_durations = []
    sample_rates = set()
    channels_set = set()
    clipped_files = 0
    silence_ratios = []

    for p in sample_paths:
        with wave.open(p, "rb") as wf:
            n_ch = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            dur = n_frames / sr
            frames = wf.readframes(n_frames)

        sample_rates.add(sr)
        channels_set.add(n_ch)
        sample_durations.append(dur)

        # Basic signal analysis
        samples = np.frombuffer(frames, dtype=np.int16)
        if len(samples) > 0:
            if np.max(np.abs(samples)) >= 32767:
                clipped_files += 1
            # Frames with amplitude below 1% of max possible considered silence
            silence_ratio = np.mean(np.abs(samples) < 327)
            silence_ratios.append(float(silence_ratio))

    audio_sanity = {
        "sampled_clips_count": len(sample_paths),
        "sample_rates_observed": list(sample_rates),
        "channels_observed": list(channels_set),
        "all_16k_mono": bool(sample_rates == {16000} and channels_set == {1}),
        "duration_min_s": round(float(np.min(sample_durations)), 2),
        "duration_mean_s": round(float(np.mean(sample_durations)), 2),
        "duration_p95_s": round(float(np.percentile(sample_durations, 95)), 2),
        "duration_p99_s": round(float(np.percentile(sample_durations, 99)), 2),
        "duration_max_s": round(float(np.max(sample_durations)), 2),
        "clipped_clips_count": clipped_files,
        "mean_silence_ratio": round(float(np.mean(silence_ratios)), 3),
    }

    # Check 5: Observed label sets vs README
    observed_actions = sorted(all_splits_df["action"].dropna().unique().tolist())
    observed_objects = sorted(all_splits_df["object"].dropna().unique().tolist())
    observed_locations = sorted(all_splits_df["location"].dropna().unique().tolist())

    readme_actions = sorted(["change language", "activate", "deactivate", "increase", "decrease", "bring"])
    readme_objects = sorted([
        "none", "music", "lights", "volume", "heat", "lamp",
        "newspaper", "juice", "socks", "shoes",
        "Chinese", "Korean", "English", "German",
    ])
    readme_locations = sorted(["none", "kitchen", "bedroom", "washroom"])

    labels_check = {
        "actions_match_readme": observed_actions == readme_actions,
        "objects_match_readme": observed_objects == readme_objects,
        "locations_match_readme": observed_locations == readme_locations,
        "observed_actions": observed_actions,
        "observed_objects": observed_objects,
        "observed_locations": observed_locations,
    }

    # Check 6: Extract intent vocabulary & mappings (expect 31)
    intent_map, slot_vocab = extract_intent_vocabularies(all_splits_df)

    # Check 7: Overlap analysis between splits
    overlap_results = analyze_split_overlaps(train_df, valid_df, test_df)

    # Check 8: Demographics join
    split_speakers = set(all_splits_df["speakerId"].unique())
    demo_speakers = set(demo_df["speakerId"].unique())
    missing_demo_speakers = list(split_speakers - demo_speakers)

    # Check 9: Label consistency (each transcript maps to exactly one intent)
    transcript_to_intents = (
        all_splits_df.groupby("transcript")[["action", "object", "location"]]
        .apply(lambda g: g.drop_duplicates().to_dict(orient="records"))
        .to_dict()
    )
    conflicting_transcripts = {
        txt: intents for txt, intents in transcript_to_intents.items() if len(intents) > 1
    }

    # Check 10: Folder speaker equals speakerId column & content duplicates
    speaker_folder_mismatches = []
    for _, row in all_splits_df.iterrows():
        # path is like 'wavs/speakers/<speakerId>/<filename>.wav'
        parts = os.path.normpath(row["path"]).split(os.sep)
        if len(parts) >= 3:
            folder_spk = parts[-2]
            if folder_spk != row["speakerId"]:
                speaker_folder_mismatches.append({"path": row["path"], "expected": row["speakerId"], "found": folder_spk})

    # Sample hash duplicates check across 100 clips
    hash_to_path = {}
    duplicate_audio_hashes = []
    sample_for_hash = np.random.choice(
        list(all_wavs_on_disk),
        size=min(200, len(all_wavs_on_disk)),
        replace=False,
    )
    for p in sample_for_hash:
        h = compute_file_hash(p)
        if h in hash_to_path:
            duplicate_audio_hashes.append({"file1": hash_to_path[h], "file2": p, "hash": h})
        else:
            hash_to_path[h] = p

    # Check 11: Phrasings per intent in every split & class imbalance ratios
    phrasings_per_intent = {
        "train": train_df.groupby(["action", "object", "location"])["transcript"].nunique().to_dict(),
        "valid": valid_df.groupby(["action", "object", "location"])["transcript"].nunique().to_dict(),
        "test": test_df.groupby(["action", "object", "location"])["transcript"].nunique().to_dict(),
    }
    # Convert tuple keys to strings for JSON serialization
    phrasings_per_intent_json = {
        split: {f"{k[0]}|{k[1]}|{k[2]}": count for k, count in d.items()}
        for split, d in phrasings_per_intent.items()
    }

    action_counts = all_splits_df["action"].value_counts().to_dict()
    object_counts = all_splits_df["object"].value_counts().to_dict()
    location_counts = all_splits_df["location"].value_counts().to_dict()

    imbalance_ratios = {
        "action_max_to_min": round(max(action_counts.values()) / min(action_counts.values()), 2),
        "object_max_to_min": round(max(object_counts.values()) / min(object_counts.values()), 2),
        "location_max_to_min": round(max(location_counts.values()) / min(location_counts.values()), 2),
    }

    audit_report = {
        "dataset_root": paths.root_dir,
        "files_presence": csv_exists,
        "raw_headers": raw_headers,
        "counts": counts,
        "paper_comparison": {
            "expected": paper_expected,
            "differences": count_diffs,
            "exact_match": count_diffs == {"train": 0, "valid": 0, "test": 0, "total": 0},
        },
        "audio_files": {
            "total_wavs_on_disk": len(all_wavs_on_disk),
            "missing_files_count": len(missing_files),
        },
        "audio_sanity": audio_sanity,
        "labels": labels_check,
        "unique_intents_count": len(intent_map),
        "overlaps": overlap_results,
        "demographics_join": {
            "all_speakers_in_demographics": len(missing_demo_speakers) == 0,
            "missing_speakers": missing_demo_speakers,
        },
        "label_consistency": {
            "conflicting_transcripts_count": len(conflicting_transcripts),
            "conflicts": conflicting_transcripts,
        },
        "speaker_folder_consistency": {
            "mismatches_count": len(speaker_folder_mismatches),
        },
        "hash_duplicates_sampled": len(duplicate_audio_hashes),
        "imbalance_ratios": imbalance_ratios,
        "phrasings_per_intent": phrasings_per_intent_json,
    }

    return audit_report, intent_map, slot_vocab


def main():
    parser = argparse.ArgumentParser(description="Run complete FSC dataset audit (docs/01 §2).")
    parser.add_argument(
        "--root",
        default="data/raw/fluent_speech_commands_dataset",
        help="Path to Fluent Speech Commands dataset root directory",
    )
    parser.add_argument(
        "--output-audit",
        default="reports/data_audit.json",
        help="Path to output data audit JSON",
    )
    parser.add_argument(
        "--output-intent-map",
        default="configs/intent_map.json",
        help="Path to output intent map JSON",
    )
    args = parser.parse_args()

    audit_report, intent_map, slot_vocab = run_audit(args.root)

    # Save audit report
    os.makedirs(os.path.dirname(args.output_audit), exist_ok=True)
    with open(args.output_audit, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"[00_check_data] Saved audit report to: {args.output_audit}")

    # Save intent map
    os.makedirs(os.path.dirname(args.output_intent_map), exist_ok=True)
    intent_payload = {
        "intent_map": intent_map,
        "slot_vocab": slot_vocab,
        "num_intents": len(intent_map),
    }
    with open(args.output_intent_map, "w", encoding="utf-8") as f:
        json.dump(intent_payload, f, indent=2)
    print(f"[00_check_data] Saved intent map to: {args.output_intent_map}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("FSC DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Audio Rows      : {audit_report['counts']['total']} (Matches paper: {audit_report['paper_comparison']['exact_match']})")
    print(f"Audio Files on Disk   : {audit_report['audio_files']['total_wavs_on_disk']} (Missing: {audit_report['audio_files']['missing_files_count']})")
    print(f"Unique Intents        : {audit_report['unique_intents_count']} (Expected: 31)")
    print(f"Audio Specs           : 16 kHz Mono = {audit_report['audio_sanity']['all_16k_mono']}")
    print(f"Duration Stats        : min={audit_report['audio_sanity']['duration_min_s']}s, p95={audit_report['audio_sanity']['duration_p95_s']}s, p99={audit_report['audio_sanity']['duration_p99_s']}s, max={audit_report['audio_sanity']['duration_max_s']}s")
    print(f"Label Consistency     : {audit_report['label_consistency']['conflicting_transcripts_count']} conflicts")
    print(f"Phrase Leakage (Test) : {audit_report['overlaps']['transcripts']['test_utterances_in_train_pct']}% of test phrases appear in train!")
    print(f"Speaker Leakage (Test): {audit_report['overlaps']['speakers']['train_test_overlap']} speakers shared")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
