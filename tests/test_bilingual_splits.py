"""Unit tests untuk Validasi Stratified Bilingual Splits 24k (Sub-step 5.2).

Memverifikasi integritas, distribusi, paritas bahasa, representasi 31 intent,
serta pembuktian matematis zero-leakage pada level transkrip dan penutur.
"""

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

SPLITS_DIR = Path("data/bilingual_massive/splits")
MANIFEST_JSON_PATH = SPLITS_DIR / "split_manifest.json"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VALID_CSV = SPLITS_DIR / "valid.csv"
TEST_CSV = SPLITS_DIR / "test.csv"
TEST_ID_CSV = SPLITS_DIR / "test_id.csv"
TEST_EN_CSV = SPLITS_DIR / "test_en.csv"


def get_file_sha256(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class TestBilingualSplitsIntegrity:
    """Pengujian keberadaan berkas dan kepatuhan kuota baris."""

    def test_split_files_exist(self):
        """Memastikan seluruh 5 CSV split dan 1 JSON manifes tersedia di disk."""
        assert TRAIN_CSV.exists(), f"File {TRAIN_CSV} tidak ditemukan"
        assert VALID_CSV.exists(), f"File {VALID_CSV} tidak ditemukan"
        assert TEST_CSV.exists(), f"File {TEST_CSV} tidak ditemukan"
        assert TEST_ID_CSV.exists(), f"File {TEST_ID_CSV} tidak ditemukan"
        assert TEST_EN_CSV.exists(), f"File {TEST_EN_CSV} tidak ditemukan"
        assert MANIFEST_JSON_PATH.exists(), f"File {MANIFEST_JSON_PATH} tidak ditemukan"

    def test_exact_row_counts_and_percentages(self):
        """Memastikan rasio 70:15:15 dan jumlah baris presisi bulat."""
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VALID_CSV)
        df_test = pd.read_csv(TEST_CSV)
        df_test_id = pd.read_csv(TEST_ID_CSV)
        df_test_en = pd.read_csv(TEST_EN_CSV)

        assert len(df_train) == 16800, f"Train harus 16.800 baris, didapat {len(df_train)}"
        assert len(df_val) == 3600, f"Valid harus 3.600 baris, didapat {len(df_val)}"
        assert len(df_test) == 3600, f"Test harus 3.600 baris, didapat {len(df_test)}"
        assert len(df_test_id) == 1800, f"Test ID harus 1.800 baris, didapat {len(df_test_id)}"
        assert len(df_test_en) == 1800, f"Test EN harus 1.800 baris, didapat {len(df_test_en)}"

        total_audio = len(df_train) + len(df_val) + len(df_test)
        assert total_audio == 24000, f"Total audio partisi harus tepat 24.000, didapat {total_audio}"

    def test_language_parity_50_50_across_splits(self):
        """Memastikan setiap partisi memiliki proporsi bahasa seimbang 50:50."""
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VALID_CSV)
        df_test = pd.read_csv(TEST_CSV)

        for split_name, df in [("train", df_train), ("valid", df_val), ("test", df_test)]:
            counts = df["language"].value_counts()
            assert "id" in counts and "en" in counts, f"{split_name} harus mengandung bahasa id dan en"
            assert counts["id"] == counts["en"], (
                f"Keseimbangan bahasa di {split_name} tidak 50:50! "
                f"id={counts['id']}, en={counts['en']}"
            )

        # Cek slice evaluasi independen
        df_test_id = pd.read_csv(TEST_ID_CSV)
        df_test_en = pd.read_csv(TEST_EN_CSV)
        assert (df_test_id["language"] == "id").all(), "Test ID slice harus 100% bahasa Indonesia"
        assert (df_test_en["language"] == "en").all(), "Test EN slice harus 100% bahasa Inggris"

    def test_full_intent_coverage_in_all_splits(self):
        """Memastikan seluruh 31 intent (0..30) hadir di seluruh split."""
        expected_intents = set(range(31))

        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VALID_CSV)
        df_test = pd.read_csv(TEST_CSV)
        df_test_id = pd.read_csv(TEST_ID_CSV)
        df_test_en = pd.read_csv(TEST_EN_CSV)

        for name, df in [
            ("Train", df_train),
            ("Valid", df_val),
            ("Test", df_test),
            ("Test ID", df_test_id),
            ("Test EN", df_test_en),
        ]:
            present_intents = set(df["intent_id"].unique())
            assert present_intents == expected_intents, (
                f"{name} split kehilangan intent: {expected_intents - present_intents}"
            )


class TestZeroDataLeakage:
    """Pengujian ketat pencegahan kebocoran data (Leakage Prevention)."""

    def test_zero_indonesian_utterance_leakage(self):
        """Verifikasi bahwa kalimat bahasa Indonesia pada Test tidak pernah muncul di Train (Unseen Utterance)."""
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VALID_CSV)
        df_test = pd.read_csv(TEST_CSV)

        train_id_transcripts = set(df_train[df_train["language"] == "id"]["transcript"])
        val_id_transcripts = set(df_val[df_val["language"] == "id"]["transcript"])
        test_id_transcripts = set(df_test[df_test["language"] == "id"]["transcript"])

        train_test_overlap = train_id_transcripts & test_id_transcripts
        train_val_overlap = train_id_transcripts & val_id_transcripts
        val_test_overlap = val_id_transcripts & test_id_transcripts

        assert len(train_test_overlap) == 0, f"Kebocoran transkrip Train-Test ID: {train_test_overlap}"
        assert len(train_val_overlap) == 0, f"Kebocoran transkrip Train-Valid ID: {train_val_overlap}"
        assert len(val_test_overlap) == 0, f"Kebocoran transkrip Valid-Test ID: {val_test_overlap}"

    def test_zero_english_speaker_leakage(self):
        """Verifikasi bahwa penutur FSC English pada Test tidak pernah muncul di Train (Speaker-Disjoint)."""
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VALID_CSV)
        df_test = pd.read_csv(TEST_CSV)

        train_en_spks = set(df_train[df_train["language"] == "en"]["speaker_id"])
        val_en_spks = set(df_val[df_val["language"] == "en"]["speaker_id"])
        test_en_spks = set(df_test[df_test["language"] == "en"]["speaker_id"])

        train_test_overlap = train_en_spks & test_en_spks
        train_val_overlap = train_en_spks & val_en_spks
        val_test_overlap = val_en_spks & test_en_spks

        assert len(train_test_overlap) == 0, f"Kebocoran speaker Train-Test EN: {train_test_overlap}"
        assert len(train_val_overlap) == 0, f"Kebocoran speaker Train-Valid EN: {train_val_overlap}"
        assert len(val_test_overlap) == 0, f"Kebocoran speaker Valid-Test EN: {val_test_overlap}"

        total_unique_spks = len(train_en_spks | val_en_spks | test_en_spks)
        assert total_unique_spks == 79, f"Total penutur FSC harus 79, didapat {total_unique_spks}"

    def test_zero_audio_file_path_overlap(self):
        """Verifikasi bahwa berkas audio pada masing-masing partisi bersifat saling lepas sempurna."""
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VALID_CSV)
        df_test = pd.read_csv(TEST_CSV)

        train_paths = set(df_train["audio_path"])
        val_paths = set(df_val["audio_path"])
        test_paths = set(df_test["audio_path"])

        assert len(train_paths & test_paths) == 0, "Ada tumpang tindih berkas audio antara Train dan Test"
        assert len(train_paths & val_paths) == 0, "Ada tumpang tindih berkas audio antara Train dan Valid"
        assert len(val_paths & test_paths) == 0, "Ada tumpang tindih berkas audio antara Valid dan Test"

        total_unique_paths = len(train_paths | val_paths | test_paths)
        assert total_unique_paths == 24000, f"Total jalur berkas harus 24.000, didapat {total_unique_paths}"


class TestManifestIntegrityAndHashes:
    """Pengujian verifikasi checksum sha256 dan sinkronisasi manifes json."""

    def test_manifest_sha256_matches_disk(self):
        """Memastikan SHA-256 yang tercantum pada manifes sesuai dengan kondisi riil berkas."""
        with open(MANIFEST_JSON_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        splits_meta = manifest["splits"]

        assert splits_meta["train"]["sha256"] == get_file_sha256(TRAIN_CSV)
        assert splits_meta["valid"]["sha256"] == get_file_sha256(VALID_CSV)
        assert splits_meta["test"]["sha256"] == get_file_sha256(TEST_CSV)
        assert splits_meta["test_id"]["sha256"] == get_file_sha256(TEST_ID_CSV)
        assert splits_meta["test_en"]["sha256"] == get_file_sha256(TEST_EN_CSV)

    def test_manifest_leakage_flags_true(self):
        """Memastikan status bebas kebocoran pada manifes bernilai True."""
        with open(MANIFEST_JSON_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        leakage_info = manifest["leakage_verification"]
        assert leakage_info["indonesian_unseen_utterance"]["is_zero_leakage"] is True
        assert leakage_info["english_speaker_disjoint"]["is_zero_leakage"] is True
