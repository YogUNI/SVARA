"""Unit tests untuk Validasi Zero Data Leakage Audit (Sub-step 5.3).

Memverifikasi hasil audit 8 dimensi ketiadaan kebocoran data, integritas baris,
paritas bahasa, keunikan biner waveform, dan konsistensi label.
"""

import json
from pathlib import Path
import pytest

REPORT_JSON_PATH = Path("reports/zero_data_leakage_audit.json")
REPORT_MD_PATH = Path("reports/zero_data_leakage_audit.md")


@pytest.fixture(scope="module")
def audit_data():
    """Load audit report JSON."""
    assert REPORT_JSON_PATH.exists(), f"Laporan {REPORT_JSON_PATH} belum dibuat"
    with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestZeroDataLeakageReport:
    """Pengujian verifikasi kepatuhan seluruh dimensi audit zero-leakage."""

    def test_report_files_exist(self):
        """Memastikan berkas laporan audit format JSON dan Markdown ada di reports/."""
        assert REPORT_JSON_PATH.exists(), "Berkas JSON audit tidak ditemukan"
        assert REPORT_MD_PATH.exists(), "Berkas Markdown audit tidak ditemukan"
        assert REPORT_MD_PATH.stat().st_size > 1000, "Berkas Markdown audit terlalu kecil/kosong"

    def test_global_audit_status(self, audit_data):
        """Memastikan status global audit bernilai PASS."""
        assert audit_data["global_status"] == "PASS"
        assert audit_data["sub_step"] == "5.3"

    def test_dim1_row_integrity_and_language_parity(self, audit_data):
        """Memastikan Dimensi 1 lolos (16.800 Train, 3.600 Valid, 3.600 Test, paritas 50:50)."""
        d1 = audit_data["audit_dimensions"]["dim1_row_integrity_and_language_parity"]
        assert d1["status"] == "PASS"
        assert d1["total_rows"] == 24000
        assert d1["train_rows"] == 16800
        assert d1["valid_rows"] == 3600
        assert d1["test_rows"] == 3600
        assert d1["test_id_rows"] == 1800
        assert d1["test_en_rows"] == 1800
        for split in ["train", "valid", "test"]:
            assert d1["language_parity"][split]["id"] == d1["language_parity"][split]["en"]

    def test_dim2_indonesian_unseen_utterance_zero_leakage(self, audit_data):
        """Memastikan Dimensi 2 lolos (Zero utterance leakage pada bahasa Indonesia)."""
        d2 = audit_data["audit_dimensions"]["dim2_indonesian_unseen_utterance_isolation"]
        assert d2["status"] == "PASS"
        assert d2["train_test_overlap_count"] == 0
        assert d2["train_valid_overlap_count"] == 0
        assert d2["valid_test_overlap_count"] == 0
        assert d2["is_unseen_utterance_strictly_isolated"] is True
        assert d2["total_unique_indonesian_phrases"] == 4358

    def test_dim3_english_speaker_disjoint_zero_leakage(self, audit_data):
        """Memastikan Dimensi 3 lolos (Zero speaker leakage pada English FSC)."""
        d3 = audit_data["audit_dimensions"]["dim3_english_speaker_disjoint_isolation"]
        assert d3["status"] == "PASS"
        assert d3["train_test_overlap_count"] == 0
        assert d3["train_valid_overlap_count"] == 0
        assert d3["valid_test_overlap_count"] == 0
        assert d3["is_speaker_disjoint_strictly_isolated"] is True
        assert d3["total_unique_english_speakers"] == 79

    def test_dim4_waveform_binary_uniqueness_zero_leakage(self, audit_data):
        """Memastikan Dimensi 4 lolos (Zero duplicate audio content across all splits)."""
        d4 = audit_data["audit_dimensions"]["dim4_audio_waveform_binary_uniqueness"]
        assert d4["status"] == "PASS"
        assert d4["total_audited_audio_files"] == 24000
        assert d4["train_test_hash_overlap"] == 0
        assert d4["train_valid_hash_overlap"] == 0
        assert d4["valid_test_hash_overlap"] == 0
        assert d4["internal_duplicates_in_train"] == 0
        assert d4["internal_duplicates_in_valid"] == 0
        assert d4["internal_duplicates_in_test"] == 0
        assert d4["is_binary_waveform_unique"] is True

    def test_dim5_label_consistency_and_phrasing_diversity(self, audit_data):
        """Memastikan Dimensi 5 lolos (0 konflik label dan memenuhi standar V-01 serta V-02)."""
        d5 = audit_data["audit_dimensions"]["dim5_label_consistency_and_phrasing_diversity"]
        assert d5["status"] == "PASS"
        assert d5["label_mapping_conflicts_v01"] == 0
        assert d5["v02_minimum_phrasing_requirement_met"] is True
        assert d5["indonesian_train_phrases_per_intent"]["min"] >= 2
        assert d5["english_train_phrases_per_intent"]["min"] >= 2

    def test_dim6_intent_class_balance_and_entropy(self, audit_data):
        """Memastikan Dimensi 6 lolos (31 intent legal dan entropi ternormalisasi >= 0.998)."""
        d6 = audit_data["audit_dimensions"]["dim6_intent_class_balance_and_entropy"]
        assert d6["status"] == "PASS"
        for split in ["train", "valid", "test"]:
            assert d6["split_entropy_metrics"][split]["intents_represented"] == 31
            assert d6["split_entropy_metrics"][split]["normalized_entropy"] >= 0.998

    def test_dim7_duration_and_signal_hygiene(self, audit_data):
        """Memastikan Dimensi 7 lolos (Frekuensi 16 kHz, Mono, PCM_16, durasi wajar)."""
        d7 = audit_data["audit_dimensions"]["dim7_acoustic_duration_and_signal_hygiene"]
        assert d7["status"] == "PASS"
        assert d7["sample_rate_hz"] == 16000
        assert d7["channels"] == 1
        assert d7["format"] == "PCM_16"
        for split in ["train", "valid", "test"]:
            assert d7["duration_statistics"][split]["min_sec"] >= 0.5
            assert d7["duration_statistics"][split]["max_sec"] <= 12.0

    def test_dim8_manifest_checksum_match(self, audit_data):
        """Memastikan Dimensi 8 lolos (100% kecocokan SHA-256 antara manifes dan disk)."""
        d8 = audit_data["audit_dimensions"]["dim8_manifest_checksum_and_disk_sync"]
        assert d8["status"] == "PASS"
        assert d8["all_checksums_match"] is True
        for k, v in d8["manifest_sha256_verifications"].items():
            assert v is True, f"Checksum {k} tidak cocok"
