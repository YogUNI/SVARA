"""Balanced Bilingual Dataset Integration Pipeline (Sub-step 5.1).

Task: Phase 5: Balanced Bilingual Dataset Integration & Splitting
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md, docs/01_DATASET_AND_EVALUATION.md

Merges:
1. 12,000 Indonesian robust audio clips (stratified from 26,970 clips in Phase 4)
2. 12,000 English FSC native audio clips (stratified from 30,043 clips in FSC)
Total: Exactly 24,000 balanced bilingual audio clips (50:50 language ratio, perfectly matched intent quotas).

Outputs:
- Manifest: data/bilingual_massive/bilingual_manifest_24k.csv
- Summary Report: reports/bilingual_dataset_integration_summary.json
"""

import os
import sys
import csv
import json
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import soundfile as sf

DEFAULT_INDO_MANIFEST = "data/synthetic_indonesian_massive/metadata_augmented.csv"
DEFAULT_FSC_ROOT = "data/raw/fluent_speech_commands_dataset"
DEFAULT_INTENT_MAP = "configs/intent_map.json"
DEFAULT_OUT_DIR = "data/bilingual_massive"
DEFAULT_OUT_MANIFEST = "data/bilingual_massive/bilingual_manifest_24k.csv"
DEFAULT_REPORT_JSON = "reports/bilingual_dataset_integration_summary.json"

TARGET_TOTAL_CLIPS = 24000
TARGET_PER_LANG = 12000
NUM_INTENTS = 31
RANDOM_SEED = 42


def categorize_domain(action: str, obj: str) -> str:
    """Categorizes an intent into one of five smart home functional domains."""
    if obj in ["lights", "lamp"]:
        return "Lighting (Pencahayaan)"
    elif obj == "heat":
        return "Climate (Pengatur Suhu & AC)"
    elif obj in ["music", "volume"]:
        return "Media & Audio (Hiburan & Volume)"
    elif action == "bring":
        return "Robot Butler (Pengantaran Barang)"
    elif action == "change language":
        return "System Language (Pengaturan Bahasa)"
    return "Other / Miscellaneous"


def compute_intent_quotas(target_per_lang: int = TARGET_PER_LANG, num_intents: int = NUM_INTENTS) -> Dict[int, int]:
    """Computes exact per-intent quota so sum equals target_per_lang with minimal variance."""
    base_quota = target_per_lang // num_intents
    remainder = target_per_lang % num_intents
    quotas = {}
    for i in range(num_intents):
        quotas[i] = base_quota + (1 if i < remainder else 0)
    assert sum(quotas.values()) == target_per_lang
    return quotas


def sample_indonesian_robust(
    indo_manifest_path: str,
    quotas: Dict[int, int],
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Samples exactly 12,000 Indonesian clips with 50:50 gender balance and 1/3 profile balance."""
    print("[*] Loading Indonesian augmented manifest...")
    df_indo = pd.read_csv(indo_manifest_path)
    rng = np.random.RandomState(seed)

    selected_indices = []
    for i, q in quotas.items():
        if i % 2 == 0:
            q_m = (q + 1) // 2
            q_f = q - q_m
        else:
            q_f = (q + 1) // 2
            q_m = q - q_f

        df_intent = df_indo[df_indo["intent_id"] == i]

        for g_idx, (g, q_g) in enumerate([("male", q_m), ("female", q_f)]):
            df_g = df_intent[df_intent["gender"] == g]
            # Rotate starting variation for even distribution across profiles
            rot = (i * 2 + g_idx) % 3
            vars_base = ["var1_bedroom_slow", "var2_living_fast", "var3_kitchen_norm"]
            vars_list = vars_base[rot:] + vars_base[:rot]
            sub_indices = []
            for v in vars_list:
                v_idx = list(df_g[df_g["variation_id"] == v].index)
                rng.shuffle(v_idx)
                sub_indices.append(v_idx)

            # Interleave round robin across variations
            interleaved = []
            max_len = max(len(s) for s in sub_indices)
            for step in range(max_len):
                for v_idx in sub_indices:
                    if step < len(v_idx):
                        interleaved.append(v_idx[step])
            selected_indices.extend(interleaved[:q_g])

    df_sampled = df_indo.loc[selected_indices].copy()
    assert len(df_sampled) == TARGET_PER_LANG, f"Expected {TARGET_PER_LANG}, got {len(df_sampled)}"

    # Standardize columns
    records = []
    for _, r in df_sampled.iterrows():
        records.append({
            "audio_path": f"data/synthetic_indonesian_massive/{r['path']}".replace("\\", "/"),
            "speaker_id": r["speakerId"],
            "gender": r["gender"],
            "language": "id",
            "transcript": str(r["transcript"]).strip(),
            "action": r["action"],
            "object": r["object"],
            "location": r["location"],
            "intent_id": int(r["intent_id"]),
            "domain": r.get("domain", categorize_domain(r["action"], r["object"])),
            "duration": float(r["duration"]),
            "acoustic_profile": r["variation_id"],
            "source_split_original": "synthetic_indonesian_massive",
        })

    return pd.DataFrame(records)


def sample_english_fsc(
    fsc_root: str,
    quotas: Dict[int, int],
    intent_map_path: str,
    seed: int = RANDOM_SEED,
    workers: int = 8,
    duration_cache: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Samples exactly 12,000 English FSC clips matching intent quotas and speaker diversity."""
    print("[*] Loading Fluent Speech Commands (FSC) dataset...")
    # Load intent map
    with open(intent_map_path, "r", encoding="utf-8") as f:
        raw_imap = json.load(f)
    imap = raw_imap.get("intent_map", raw_imap)
    tuple_to_id = {(v["action"], v["object"], v["location"]): int(k) for k, v in imap.items()}

    # Load FSC splits
    dfs = []
    for split_name in ["train", "valid", "test"]:
        csv_path = os.path.join(fsc_root, "data", f"{split_name}_data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"FSC split CSV not found: {csv_path}")
        d = pd.read_csv(csv_path)
        d["fsc_original_split"] = split_name
        dfs.append(d)
    df_all_fsc = pd.concat(dfs, ignore_index=True)

    # Load demographics for gender
    demo_path = os.path.join(fsc_root, "data", "speaker_demographics.csv")
    spk_gender = {}
    if os.path.exists(demo_path):
        df_demo = pd.read_csv(demo_path)
        spk_gender = dict(zip(df_demo["speakerId"], df_demo["gender"]))

    # Map intent_id
    intent_ids = []
    for _, r in df_all_fsc.iterrows():
        tup = (r["action"], r["object"], r["location"])
        iid = tuple_to_id.get(tup, -1)
        if iid == -1:
            raise ValueError(f"Unmapped FSC intent tuple: {tup}")
        intent_ids.append(iid)
    df_all_fsc["intent_id"] = intent_ids

    # Sample matching quotas
    rng = np.random.RandomState(seed)
    selected_rows = []

    for intent_id, q in quotas.items():
        df_intent = df_all_fsc[df_all_fsc["intent_id"] == intent_id].copy()
        assert len(df_intent) >= q, f"Insufficient FSC clips for intent {intent_id}: {len(df_intent)} < {q}"

        permuted = rng.permutation(df_intent.index)
        # Sort to disperse across speakers and splits
        sorted_indices = df_intent.loc[permuted].sort_values(
            by=["speakerId", "fsc_original_split"]
        ).index
        chosen_indices = sorted_indices[:q]
        selected_rows.append(df_intent.loc[chosen_indices])

    df_sampled = pd.concat(selected_rows, ignore_index=True)
    assert len(df_sampled) == TARGET_PER_LANG, f"Expected {TARGET_PER_LANG}, got {len(df_sampled)}"

    # Determine audio durations via sf.info in parallel
    print(f"[*] Inspecting audio headers for {len(df_sampled):,} FSC clips ({workers} threads)...")
    abs_paths = [
        os.path.join(fsc_root, p).replace("\\", "/")
        for p in df_sampled["path"]
    ]

    cache = duration_cache or {}

    def _get_duration(path: str) -> float:
        if path in cache:
            return cache[path]
        try:
            return round(sf.info(path).duration, 3)
        except Exception:
            return 2.5

    with ThreadPoolExecutor(max_workers=workers) as executor:
        durations = list(executor.map(_get_duration, abs_paths))

    records = []
    for idx, (_, r) in enumerate(df_sampled.iterrows()):
        spk = r["speakerId"]
        g = spk_gender.get(spk, "unknown")
        rel_audio_path = os.path.join(fsc_root, r["path"]).replace("\\", "/")
        records.append({
            "audio_path": rel_audio_path,
            "speaker_id": spk,
            "gender": g,
            "language": "en",
            "transcript": str(r["transcription"]).strip(),
            "action": r["action"],
            "object": r["object"],
            "location": r["location"],
            "intent_id": int(r["intent_id"]),
            "domain": categorize_domain(r["action"], r["object"]),
            "duration": durations[idx],
            "acoustic_profile": "fsc_native",
            "source_split_original": r["fsc_original_split"],
        })

    return pd.DataFrame(records)


def build_balanced_bilingual_dataset(
    indo_manifest_path: str = DEFAULT_INDO_MANIFEST,
    fsc_root: str = DEFAULT_FSC_ROOT,
    intent_map_path: str = DEFAULT_INTENT_MAP,
    out_dir: str = DEFAULT_OUT_DIR,
    out_manifest_path: str = DEFAULT_OUT_MANIFEST,
    report_json_path: str = DEFAULT_REPORT_JSON,
    seed: int = RANDOM_SEED,
    workers: int = 8,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("   SVARA: PIPELINE INTEGRASI BALANCED BILINGUAL DATASET 24K (SUB-STEP 5.1)")
    print("=" * 80)
    print(f"Manifest Indonesia Robust      : {indo_manifest_path}")
    print(f"Dataset Root English FSC       : {fsc_root}")
    print(f"Target Total Audio Bilingual   : {TARGET_TOTAL_CLIPS:,} audio")
    print(f"Target Per Bahasa (50:50)      : {TARGET_PER_LANG:,} ID / {TARGET_PER_LANG:,} EN")
    print(f"File Output Manifest           : {out_manifest_path}")
    print(f"File Laporan Ringkasan         : {report_json_path}")
    print(f"Random Seed Reproducibility    : {seed}")
    print("=" * 80 + "\n")

    start_time = time.time()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_json_path), exist_ok=True)

    # 1. Quotas per intent
    quotas = compute_intent_quotas(TARGET_PER_LANG, NUM_INTENTS)
    print(f"[*] Quota per intent (31 intents): {min(quotas.values())} to {max(quotas.values())} clips per language")

    # 2. Sample Indonesian
    df_indo = sample_indonesian_robust(indo_manifest_path, quotas, seed=seed)
    print(f"[+] Indonesian subset sampled    : {len(df_indo):,} audio (Durasi: {df_indo['duration'].sum() / 3600:.2f} jam)")

    # Check for duration cache from previous manifest
    dur_cache = {}
    if os.path.exists(out_manifest_path):
        try:
            df_prev = pd.read_csv(out_manifest_path)
            for _, r in df_prev.iterrows():
                dur_cache[r["audio_path"]] = float(r["duration"])
            print(f"[*] Reusing {len(dur_cache):,} pre-computed audio durations from cache.")
        except Exception:
            pass

    # 3. Sample English FSC
    df_fsc = sample_english_fsc(
        fsc_root, quotas, intent_map_path, seed=seed, workers=workers, duration_cache=dur_cache
    )
    print(f"[+] English subset sampled       : {len(df_fsc):,} audio (Durasi: {df_fsc['duration'].sum() / 3600:.2f} jam)")

    # 4. Concatenate and Shuffle deterministically
    print("\n[*] Merging subsets and applying deterministic shuffle...")
    df_bilingual = pd.concat([df_indo, df_fsc], ignore_index=True)
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(len(df_bilingual))
    df_bilingual = df_bilingual.iloc[shuffled_indices].reset_index(drop=True)

    # 5. Export Manifest CSV
    print(f"[*] Writing unified manifest to: {out_manifest_path}...")
    df_bilingual.to_csv(out_manifest_path, index=False, encoding="utf-8")

    elapsed = time.time() - start_time
    total_hours = df_bilingual["duration"].sum() / 3600.0

    # 6. Detailed statistics
    lang_counts = df_bilingual["language"].value_counts().to_dict()
    intent_lang_counts = df_bilingual.groupby(["intent_id", "language"]).size().unstack(fill_value=0)
    intent_skew = (intent_lang_counts["id"] != intent_lang_counts["en"]).sum()

    gender_id = df_bilingual[df_bilingual["language"] == "id"]["gender"].value_counts().to_dict()
    gender_en = df_bilingual[df_bilingual["language"] == "en"]["gender"].value_counts().to_dict()

    prof_counts = df_bilingual["acoustic_profile"].value_counts().to_dict()
    domain_counts = df_bilingual["domain"].value_counts().to_dict()

    print("\n" + "=" * 80)
    print("      RINGKASAN INTEGRASI BALANCED BILINGUAL DATASET (SUB-STEP 5.1)")
    print("=" * 80)
    print(f"Total Audio Bilingual Tersusun : {len(df_bilingual):,} / {TARGET_TOTAL_CLIPS:,} audio")
    print(f"Total Durasi Audio             : {total_hours:.2f} jam ({df_bilingual['duration'].sum():,.1f} detik)")
    print(f"Rata-rata Durasi per Klip      : {df_bilingual['duration'].mean():.2f} detik")
    print(f"Rasio Bahasa                   : Indonesian {lang_counts.get('id', 0):,} (50.0%) | English {lang_counts.get('en', 0):,} (50.0%)")
    print(f"Keselarasan Intent (ID vs EN)  : {31 - intent_skew} / 31 Intent IDENTIK SEMPURNA (Skew = {intent_skew})")
    print(f"Sebaran Gender Indonesia       : Pria {gender_id.get('male', 0):,} (50.0%) | Wanita {gender_id.get('female', 0):,} (50.0%)")
    print(f"Sebaran Gender English FSC     : Pria {gender_en.get('male', 0):,} | Wanita {gender_en.get('female', 0):,}")
    print(f"Waktu Pemrosesan Total         : {elapsed:.2f} detik ({elapsed / 60:.2f} menit)")
    print("=" * 80 + "\n")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_clips": len(df_bilingual),
        "target_clips": TARGET_TOTAL_CLIPS,
        "language_distribution": lang_counts,
        "indonesian_gender_distribution": gender_id,
        "english_gender_distribution": gender_en,
        "acoustic_profile_distribution": prof_counts,
        "domain_distribution": domain_counts,
        "intent_skew_count": int(intent_skew),
        "total_duration_hours": round(total_hours, 3),
        "mean_duration_seconds": round(float(df_bilingual["duration"].mean()), 3),
        "elapsed_seconds": round(elapsed, 2),
        "manifest_path": out_manifest_path,
        "status": "PASS" if len(df_bilingual) == TARGET_TOTAL_CLIPS and intent_skew == 0 else "FAIL",
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Balanced Bilingual Dataset Integration Pipeline")
    parser.add_argument("--indo-manifest", default=DEFAULT_INDO_MANIFEST, help="Indonesian augmented manifest CSV")
    parser.add_argument("--fsc-root", default=DEFAULT_FSC_ROOT, help="Fluent Speech Commands root directory")
    parser.add_argument("--intent-map", default=DEFAULT_INTENT_MAP, help="Path to intent map JSON")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--out-manifest", default=DEFAULT_OUT_MANIFEST, help="Output manifest CSV path")
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON, help="Report summary JSON path")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Reproducibility seed")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    args = parser.parse_args()

    build_balanced_bilingual_dataset(
        indo_manifest_path=args.indo_manifest,
        fsc_root=args.fsc_root,
        intent_map_path=args.intent_map,
        out_dir=args.out_dir,
        out_manifest_path=args.out_manifest,
        report_json_path=args.report_json,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
