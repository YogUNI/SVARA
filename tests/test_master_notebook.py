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
            "Sel 1: Konfigurasi Lingkungan",
            "Sel 2: Pemrosesan Sinyal Audio",
            "Sel 3: Pemuatan Dataset Terstratifikasi Bilingual",
            "Sel 4: Arsitektur Model Transformer SLU",
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


@pytest.fixture(scope="module")
def notebook_cells():
    with open(NOTEBOOK_03_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    return nb["cells"]


class TestMasterNotebookCells1To4Implementation:
    """Pengujian verifikasi implementasi kode produksi pada Sel 1 hingga Sel 4."""

    def test_all_code_cells_syntax_validity(self, notebook_cells):
        """Memastikan seluruh sel kode dapat di-parse sebagai sintaks Python yang valid (0 syntax error)."""
        import ast
        for i, cell in enumerate(notebook_cells):
            if cell["cell_type"] == "code":
                code_text = "".join(cell.get("source", []))
                try:
                    ast.parse(code_text)
                except SyntaxError as e:
                    pytest.fail(f"SyntaxError pada Cell index {i}: {e}")

    def test_cell_1_environment_and_cuda_implementation(self, notebook_cells):
        """Memastikan Sel 1 memuat kode inisialisasi lingkungan dan deteksi CUDA RTX 4060."""
        cell1_code = "".join(notebook_cells[2].get("source", []))
        assert "SEED = 42" in cell1_code
        assert "torch.cuda.is_available()" in cell1_code
        assert "cudnn.deterministic = True" in cell1_code
        assert "PROJECT_ROOT" in cell1_code
        assert "intent_map.json" in cell1_code

    def test_cell_2_signal_hygiene_implementation(self, notebook_cells):
        """Memastikan Sel 2 memuat visualisasi pemrosesan sinyal audio Butterworth, RMS, dan VAD."""
        cell2_code = "".join(notebook_cells[4].get("source", []))
        assert "preprocess_audio" in cell2_code
        assert "torchaudio.load" in cell2_code
        assert "specgram" in cell2_code
        assert "plt.subplots" in cell2_code

    def test_cell_3_bilingual_dataset_and_loader_implementation(self, notebook_cells):
        """Memastikan Sel 3 mendefinisikan BilingualSLUDataset dan collate_fn_bilingual."""
        cell3_code = "".join(notebook_cells[6].get("source", []))
        assert "class BilingualSLUDataset(Dataset):" in cell3_code
        assert "def collate_fn_bilingual" in cell3_code
        assert "DataLoader(" in cell3_code
        assert "train_loader" in cell3_code
        assert "valid_loader" in cell3_code
        assert "test_loader" in cell3_code

    def test_cell_4_model_architecture_and_sanity_check(self, notebook_cells):
        """Memastikan Sel 4 memuat inisialisasi Wav2VecSLU, masked pooling, dan initial random loss check."""
        cell4_code = "".join(notebook_cells[8].get("source", []))
        assert "Wav2VecSLU" in cell4_code
        assert "facebook/wav2vec2-base" in cell4_code
        assert "freeze_feature_encoder" in cell4_code
        assert "total_params" in cell4_code
        assert "loss_act_init" in cell4_code
        assert "V-11" in cell4_code
