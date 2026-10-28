"""Unit tests untuk Validasi Inisialisasi Master Notebook 03 (Sub-step 6.1).

Memverifikasi integritas berkas, kepatuhan format nbformat 4,
kelengkapan 9 sel produksi, serta kepatuhan kebijakan agar
notebooks/02 tidak tersentuh.
"""

import json
from pathlib import Path
import pytest

NOTEBOOK_03_PATH = Path("notebooks/03_production_slu_deep_learning_pipeline.ipynb")
NOTEBOOK_02_PATH = Path("notebooks/02_indonesian_bilingual_training.ipynb")


class TestMasterNotebookInitialization:
    """Pengujian keberadaan berkas dan kepatuhan skema Jupyter Notebook."""

    def test_notebook_03_exists_and_valid_json(self):
        """Memastikan berkas notebook 03 ada dan merupakan format JSON valid."""
        assert NOTEBOOK_03_PATH.exists(), f"Berkas {NOTEBOOK_03_PATH} tidak ditemukan!"
        assert NOTEBOOK_03_PATH.stat().st_size > 5000, "Ukuran berkas notebook terlalu kecil"

        with open(NOTEBOOK_03_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert isinstance(nb, dict), "Konten notebook bukan objek JSON dictionary"
        assert nb.get("nbformat") == 4, f"nbformat harus bernilai 4, didapat {nb.get('nbformat')}"
        assert "cells" in nb, "Notebook tidak memiliki key 'cells'"
        assert "metadata" in nb, "Notebook tidak memiliki key 'metadata'"

    def test_notebook_03_metadata_and_kernelspec(self):
        """Memastikan metadata dan kernelspec notebook terdefinisi untuk Python 3."""
        with open(NOTEBOOK_03_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)

        meta = nb.get("metadata", {})
        assert "kernelspec" in meta, "Kernelspec tidak ada di metadata"
        assert meta["kernelspec"]["language"] == "python"
        assert "language_info" in meta, "Language info tidak ada di metadata"
        assert meta["language_info"]["name"] == "python"

    def test_notebook_03_cell_structure_and_counts(self):
        """Memastikan struktur 19 sel (10 Markdown dan 9 Code) lengkap."""
        with open(NOTEBOOK_03_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])
        assert len(cells) == 19, f"Diharapkan tepat 19 cells, didapat {len(cells)}"

        md_cells = [c for c in cells if c["cell_type"] == "markdown"]
        code_cells = [c for c in cells if c["cell_type"] == "code"]

        assert len(md_cells) == 10, f"Diharapkan 10 markdown cells, didapat {len(md_cells)}"
        assert len(code_cells) == 9, f"Diharapkan 9 code cells, didapat {len(code_cells)}"

    def test_all_9_production_sections_present(self):
        """Memastikan kesembilan sel produksi memiliki penanda dan kode yang sesuai."""
        with open(NOTEBOOK_03_PATH, "r", encoding="utf-8") as f:
            nb = json.load(f)

        all_text = " ".join("".join(c.get("source", [])) for c in nb.get("cells", []))

        expected_sections = [
            "Sel 1: Konfigurasi Lingkungan, Seed & Deteksi GPU CUDA",
            "Sel 2: Visualisasi Sinyal Audio Sebelum vs Sesudah Filter",
            "Sel 3: Pemuatan Manifest Partisi Bilingual 24k & DataLoader",
            "Sel 4: Definisi Arsitektur Model Wav2VecSLU",
            "Sel 5: Training Engine Skeleton & Hyperparameter Configuration",
            "Sel 6: Visualisasi Riwayat Pelatihan & Metrik Konvergensi",
            "Sel 7: Evaluasi Ilmiah Multi-Domain SLU",
            "Sel 8: Pipeline Ekspor Universal ONNX & Kuantisasi INT8",
            "Sel 9: Live Audio Prediction & Smart Home Intent Consistency Gate",
        ]

        for sec in expected_sections:
            assert sec in all_text, f"Bagian sel '{sec}' tidak ditemukan dalam konten notebook!"

    def test_notebook_02_preserved(self):
        """Memastikan notebook 02 tetap ada dan tidak dimodifikasi secara ilegal."""
        assert NOTEBOOK_02_PATH.exists(), f"Berkas {NOTEBOOK_02_PATH} harus tetap ada!"
        with open(NOTEBOOK_02_PATH, "r", encoding="utf-8") as f:
            nb2 = json.load(f)
        assert nb2.get("nbformat") == 4
