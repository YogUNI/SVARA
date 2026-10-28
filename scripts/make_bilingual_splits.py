"""Script Pembuatan Stratified Split Ilmiah Bilingual Dataset (Sub-step 5.2).

Membagi 24.000 audio klip bilingual (12.000 Indonesia + 12.000 English FSC)
menjadi tiga partisi ilmiah standar deep learning:
- Train Split  : 70% = 16.800 audio (8.400 ID + 8.400 EN)
- Valid Split  : 15% =  3.600 audio (1.800 ID + 1.800 EN)
- Test Split   : 15% =  3.600 audio (1.800 ID + 1.800 EN)
- Test ID Slice:  1.800 audio (Evaluasi murni generalisasi Unseen-Utterance Bahasa Indonesia)
- Test EN Slice:  1.800 audio (Evaluasi murni generalisasi Speaker-Disjoint FSC English)

Prinsip Disiplin Deep Learning (docs/01 §4, docs/11 V-08):
1. Zero Utterance Leakage (Indonesia): Grouping strictly by `transcript`.
   set(train.transcript) ∩ set(test.transcript) == ∅
   set(train.transcript) ∩ set(valid.transcript) == ∅
   set(valid.transcript) ∩ set(test.transcript) == ∅
2. Zero Speaker Leakage (English FSC): Grouping strictly by `speaker_id`.
   set(train.speaker_id) ∩ set(test.speaker_id) == ∅
   set(train.speaker_id) ∩ set(valid.speaker_id) == ∅
   set(valid.speaker_id) ∩ set(test.speaker_id) == ∅
3. Perfect Intent Representation: All 31 intents present across all splits.
4. Exact Integer Balance: 50% ID and 50% EN in every single partition.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("svara.splits")

MANIFEST_24K_PATH = Path("data/bilingual_massive/bilingual_manifest_24k.csv")
OUTPUT_SPLITS_DIR = Path("data/bilingual_massive/splits")
REPORTS_DIR = Path("reports")


def compute_sha256(filepath: Path) -> str:
    """Hitung checksum SHA-256 dari suatu berkas."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def solve_subset_sum(items: Dict[str, int], target: int) -> List[str]:
    """Cari subset dari items yang jumlah bobotnya tepat sama dengan target menggunakan DP."""
    dp: Dict[int, List[str]] = {0: []}
    for item_key, count in items.items():
        new_dp: Dict[int, List[str]] = {}
        for s, seq in dp.items():
            new_sum = s + count
            if new_sum <= target and new_sum not in dp and new_sum not in new_dp:
                new_dp[new_sum] = seq + [item_key]
        dp.update(new_dp)
        if target in dp:
            break

    if target not in dp:
        raise ValueError(f"Tidak dapat menemukan kombinasi subset yang bernilai tepat {target}")
    return dp[target]


def split_indonesian_unseen_utterance(
    id_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bagi data Indonesia berdasarkan transkrip kalimat (Unseen Utterance).

    Target:
    - Train: 8.400 audio
    - Valid: 1.800 audio
    - Test : 1.800 audio
    Total  : 12.000 audio
    """
    logger.info("Memulai partisi data Indonesia berbasis Unseen Utterance...")

    # Kuota per intent (31 intent)
    # Total valid: 2*59 + 29*58 = 1.800
    # Total test : 2*59 + 29*58 = 1.800
    # Total train: 3*271 + 1*270 + 27*271 = 8.400
    val_targets = {i: 59 if i in [0, 1] else 58 for i in range(31)}
    test_targets = {i: 59 if i in [2, 3] else 58 for i in range(31)}

    val_transcripts: Set[str] = set()
    test_transcripts: Set[str] = set()
    train_transcripts: Set[str] = set()

    for i in range(31):
        grp = id_df[id_df["intent_id"] == i]
        t_counts = grp.groupby("transcript").size().sort_index().to_dict()

        # 1. Pilih transkrip untuk validasi
        v_selected = solve_subset_sum(t_counts, val_targets[i])
        val_transcripts.update(v_selected)

        # 2. Pilih transkrip untuk test dari sisa transkrip
        rem_counts = {t: cnt for t, cnt in t_counts.items() if t not in v_selected}
        t_selected = solve_subset_sum(rem_counts, test_targets[i])
        test_transcripts.update(t_selected)

        # 3. Sisa transkrip masuk ke train
        tr_selected = [t for t in rem_counts if t not in t_selected]
        train_transcripts.update(tr_selected)

    # Verifikasi isolasi transkrip (Unseen-Utterance assertion)
    assert len(val_transcripts & test_transcripts) == 0, "Kebocoran transkrip: Val dan Test beririsan!"
    assert len(train_transcripts & test_transcripts) == 0, "Kebocoran transkrip: Train dan Test beririsan!"
    assert len(train_transcripts & val_transcripts) == 0, "Kebocoran transkrip: Train dan Val beririsan!"

    id_val_df = id_df[id_df["transcript"].isin(val_transcripts)].copy()
    id_test_df = id_df[id_df["transcript"].isin(test_transcripts)].copy()
    id_train_df = id_df[id_df["transcript"].isin(train_transcripts)].copy()

    assert len(id_val_df) == 1800, f"Jumlah ID Val harus 1800, didapat {len(id_val_df)}"
    assert len(id_test_df) == 1800, f"Jumlah ID Test harus 1800, didapat {len(id_test_df)}"
    assert len(id_train_df) == 8400, f"Jumlah ID Train harus 8400, didapat {len(id_train_df)}"

    logger.info(
        f"Data Indonesia berhasil dipartisi: Train={len(id_train_df)}, Valid={len(id_val_df)}, Test={len(id_test_df)} | "
        f"Transkrip unik: Train={len(train_transcripts)}, Valid={len(val_transcripts)}, Test={len(test_transcripts)}"
    )
    return id_train_df, id_val_df, id_test_df


def split_english_speaker_disjoint(
    en_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bagi data English FSC berdasarkan penutur (Speaker-Disjoint).

    Target:
    - Train: 8.400 audio (53 speaker)
    - Valid: 1.800 audio (13 speaker)
    - Test : 1.800 audio (13 speaker)
    Total  : 12.000 audio (79 speaker)
    """
    logger.info("Memulai partisi data English FSC berbasis Speaker-Disjoint...")

    orig_test_spks = set(en_df[en_df["source_split_original"] == "test"]["speaker_id"])
    orig_val_spks = set(en_df[en_df["source_split_original"] == "valid"]["speaker_id"])
    orig_train_spks = set(en_df[en_df["source_split_original"] == "train"]["speaker_id"])

    # Pembagian speaker penyeimbang untuk mencapai tepat 1.800 / 1.800 / 8.400
    # Test asli: 1.654 audio (10 speaker) + 146 audio (3 speaker) = 1.800 audio (13 speaker)
    add_test_spks = {"5pa4DVyvN2fXpepb", "9mYN2zmq7aTw4Blo", "LR5vdbQgp3tlMBzB"}
    # Valid asli: 1.698 audio (9 speaker) + 102 audio (4 speaker) = 1.800 audio (13 speaker)
    add_val_spks = {"35v28XaVEns4WXOv", "W7LeKXje7QhZlLKe", "X4vEl3glp9urv4GN", "YbmvamEWQ8faDPx2"}

    final_test_spks = orig_test_spks | add_test_spks
    final_val_spks = orig_val_spks | add_val_spks
    final_train_spks = orig_train_spks - add_test_spks - add_val_spks

    # Verifikasi speaker disjoint (no speaker overlap)
    assert len(final_val_spks & final_test_spks) == 0, "Kebocoran speaker: Val dan Test beririsan!"
    assert len(final_train_spks & final_test_spks) == 0, "Kebocoran speaker: Train dan Test beririsan!"
    assert len(final_train_spks & final_val_spks) == 0, "Kebocoran speaker: Train dan Val beririsan!"
    assert len(final_train_spks) + len(final_val_spks) + len(final_test_spks) == 79, "Total speaker harus 79!"

    en_val_df = en_df[en_df["speaker_id"].isin(final_val_spks)].copy()
    en_test_df = en_df[en_df["speaker_id"].isin(final_test_spks)].copy()
    en_train_df = en_df[en_df["speaker_id"].isin(final_train_spks)].copy()

    assert len(en_val_df) == 1800, f"Jumlah EN Val harus 1800, didapat {len(en_val_df)}"
    assert len(en_test_df) == 1800, f"Jumlah EN Test harus 1800, didapat {len(en_test_df)}"
    assert len(en_train_df) == 8400, f"Jumlah EN Train harus 8400, didapat {len(en_train_df)}"

    logger.info(
        f"Data English FSC berhasil dipartisi: Train={len(en_train_df)}, Valid={len(en_val_df)}, Test={len(en_test_df)} | "
        f"Speaker unik: Train={len(final_train_spks)}, Valid={len(final_val_spks)}, Test={len(final_test_spks)}"
    )
    return en_train_df, en_val_df, en_test_df


def create_bilingual_splits() -> Dict[str, Any]:
    """Eksekusi pipeline lengkap pembagian data bilingual dan validasi integritas."""
    logger.info(f"Membaca dataset induk dari: {MANIFEST_24K_PATH}")
    if not MANIFEST_24K_PATH.exists():
        raise FileNotFoundError(f"Manifest {MANIFEST_24K_PATH} tidak ditemukan! Jalankan Sub-step 5.1 terlebih dahulu.")

    df_24k = pd.read_csv(MANIFEST_24K_PATH)
    if len(df_24k) != 24000:
        raise ValueError(f"Diharapkan tepat 24.000 baris data, namun ditemukan {len(df_24k)} baris.")

    id_df = df_24k[df_24k["language"] == "id"].copy()
    en_df = df_24k[df_24k["language"] == "en"].copy()

    # Partisi independen per bahasa sesuai kaidah ilmiahnya masing-masing
    id_train, id_val, id_test = split_indonesian_unseen_utterance(id_df)
    en_train, en_val, en_test = split_english_speaker_disjoint(en_df)

    # Gabungkan menjadi split bilingual terpadu
    # Seed 42 untuk pengacakan baris deterministik
    train_df = (
        pd.concat([id_train, en_train], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat([id_val, en_val], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat([id_test, en_test], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    # Slice murni per bahasa untuk keperluan evaluasi terpisah
    test_id_df = id_test.sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_en_df = en_test.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Pastikan direktori output tersedia
    OUTPUT_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Simpan berkas CSV
    train_path = OUTPUT_SPLITS_DIR / "train.csv"
    val_path = OUTPUT_SPLITS_DIR / "valid.csv"
    test_path = OUTPUT_SPLITS_DIR / "test.csv"
    test_id_path = OUTPUT_SPLITS_DIR / "test_id.csv"
    test_en_path = OUTPUT_SPLITS_DIR / "test_en.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    test_id_df.to_csv(test_id_path, index=False)
    test_en_df.to_csv(test_en_path, index=False)

    logger.info("Seluruh berkas CSV partisi berhasil disimpan:")
    logger.info(f"  - {train_path}: {len(train_df)} baris")
    logger.info(f"  - {val_path}: {len(val_df)} baris")
    logger.info(f"  - {test_path}: {len(test_df)} baris")
    logger.info(f"  - {test_id_path}: {len(test_id_df)} baris")
    logger.info(f"  - {test_en_path}: {len(test_en_df)} baris")

    # Hitung checksum SHA-256
    hashes = {
        "train_csv": compute_sha256(train_path),
        "valid_csv": compute_sha256(val_path),
        "test_csv": compute_sha256(test_path),
        "test_id_csv": compute_sha256(test_id_path),
        "test_en_csv": compute_sha256(test_en_path),
    }

    # Buat ringkasan manifes lengkap
    manifest_summary: Dict[str, Any] = {
        "status": "success",
        "sub_step": "5.2",
        "dataset_name": "SVARA Balanced Bilingual Dataset 24k",
        "total_clips": len(df_24k),
        "splits": {
            "train": {
                "total_rows": len(train_df),
                "percentage": round(len(train_df) / len(df_24k) * 100, 2),
                "language_breakdown": train_df["language"].value_counts().to_dict(),
                "gender_breakdown": train_df["gender"].value_counts().to_dict(),
                "intents_represented": int(train_df["intent_id"].nunique()),
                "total_duration_hours": round(float(train_df["duration"].sum() / 3600.0), 3),
                "file_path": str(train_path),
                "sha256": hashes["train_csv"],
            },
            "valid": {
                "total_rows": len(val_df),
                "percentage": round(len(val_df) / len(df_24k) * 100, 2),
                "language_breakdown": val_df["language"].value_counts().to_dict(),
                "gender_breakdown": val_df["gender"].value_counts().to_dict(),
                "intents_represented": int(val_df["intent_id"].nunique()),
                "total_duration_hours": round(float(val_df["duration"].sum() / 3600.0), 3),
                "file_path": str(val_path),
                "sha256": hashes["valid_csv"],
            },
            "test": {
                "total_rows": len(test_df),
                "percentage": round(len(test_df) / len(df_24k) * 100, 2),
                "language_breakdown": test_df["language"].value_counts().to_dict(),
                "gender_breakdown": test_df["gender"].value_counts().to_dict(),
                "intents_represented": int(test_df["intent_id"].nunique()),
                "total_duration_hours": round(float(test_df["duration"].sum() / 3600.0), 3),
                "file_path": str(test_path),
                "sha256": hashes["test_csv"],
            },
            "test_id": {
                "total_rows": len(test_id_df),
                "language_breakdown": test_id_df["language"].value_counts().to_dict(),
                "gender_breakdown": test_id_df["gender"].value_counts().to_dict(),
                "intents_represented": int(test_id_df["intent_id"].nunique()),
                "total_duration_hours": round(float(test_id_df["duration"].sum() / 3600.0), 3),
                "file_path": str(test_id_path),
                "sha256": hashes["test_id_csv"],
            },
            "test_en": {
                "total_rows": len(test_en_df),
                "language_breakdown": test_en_df["language"].value_counts().to_dict(),
                "gender_breakdown": test_en_df["gender"].value_counts().to_dict(),
                "intents_represented": int(test_en_df["intent_id"].nunique()),
                "total_duration_hours": round(float(test_en_df["duration"].sum() / 3600.0), 3),
                "file_path": str(test_en_path),
                "sha256": hashes["test_en_csv"],
            },
        },
        "leakage_verification": {
            "indonesian_unseen_utterance": {
                "train_test_transcript_overlap": len(
                    set(id_train["transcript"]) & set(id_test["transcript"])
                ),
                "train_valid_transcript_overlap": len(
                    set(id_train["transcript"]) & set(id_val["transcript"])
                ),
                "valid_test_transcript_overlap": len(
                    set(id_val["transcript"]) & set(id_test["transcript"])
                ),
                "is_zero_leakage": bool(
                    len(set(id_train["transcript"]) & set(id_test["transcript"])) == 0
                    and len(set(id_train["transcript"]) & set(id_val["transcript"])) == 0
                    and len(set(id_val["transcript"]) & set(id_test["transcript"])) == 0
                ),
            },
            "english_speaker_disjoint": {
                "train_test_speaker_overlap": len(
                    set(en_train["speaker_id"]) & set(en_test["speaker_id"])
                ),
                "train_valid_speaker_overlap": len(
                    set(en_train["speaker_id"]) & set(en_val["speaker_id"])
                ),
                "valid_test_speaker_overlap": len(
                    set(en_val["speaker_id"]) & set(en_test["speaker_id"])
                ),
                "is_zero_leakage": bool(
                    len(set(en_train["speaker_id"]) & set(en_test["speaker_id"])) == 0
                    and len(set(en_train["speaker_id"]) & set(en_val["speaker_id"])) == 0
                    and len(set(en_val["speaker_id"]) & set(en_test["speaker_id"])) == 0
                ),
            },
        },
    }

    manifest_json_path = OUTPUT_SPLITS_DIR / "split_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2, ensure_ascii=False)

    reports_summary_path = REPORTS_DIR / "bilingual_splits_summary.json"
    with open(reports_summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Ringkasan manifes disimpan di: {manifest_json_path} dan {reports_summary_path}")
    return manifest_summary


if __name__ == "__main__":
    create_bilingual_splits()
