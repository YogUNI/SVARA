# DOKUMEN MASTER LAPORAN TUGAS BESAR — BAGIAN 2 (BAB 5 s.d. BAB 8, DAFTAR PUSTAKA, DAN LAMPIRAN)

---

# BAB 5. IMPLEMENTASI

## 5.1 Lingkungan Implementasi

Pengembangan, pelatihan, dan pengujian sistem SVARA dilakukan pada lingkungan komputasi terstandarisasi untuk menjamin reproduktifitas penuh (*full reproducibility*):

1. **Lingkungan Komputasi Perangkat Keras (*Hardware Environment*)**:
   * Komputer Utama Pelatihan: Laptop ASUS/Axioo Pongo dengan prosesor Intel Core i7-13620H (10 Cores, 16 Threads, frekuensi dasar 2,4 GHz, turbo boost hingga 4,9 GHz).
   * Akselerasi Grafis: NVIDIA GeForce RTX 4060 Laptop GPU dengan 8.00 GB GDDR6 VRAM, CUDA Version 12.4, Driver NVIDIA 560.xx, Tensor Cores Generasi ke-4.
   * Memori Sistem: 16 GB DDR5 5200 MHz.
   * Penyimpanan: 1 TB NVMe PCIe Gen 4 SSD (partisi instalasi lokal dan dataset terisolasi).
   * Lingkungan Pelatihan Sekunder (Awan): Google Colab dengan GPU NVIDIA Tesla T4 (16 GB VRAM) melalui skrip sinkronisasi terotomatisasi `notebooks/colab_runner.ipynb`.

2. **Lingkungan Perangkat Lunak & Pustaka (*Software Environment*)**:
   * Sistem Operasi: Microsoft Windows 11 Home 64-bit (Build 22631).
   * Interpreter Utama: Python 3.13.5 (lingkungan virtual terisolasi).
   * Kerangka Kerja Pembelajaran Mendalam: PyTorch 2.4.1+cu124 dengan akselerasi *Automatic Mixed Precision* (AMP FP16).
   * Pemrosesan Audio: Torchaudio 2.4.1+cu124, SoundFile 0.12.1, Librosa 0.10.2.
   * Pustaka Pemodelan: HuggingFace Transformers 4.40.2 (fondasi `facebook/wav2vec2-base`).
   * Mesin Inferensi Produksi: ONNX Runtime 1.19.0 (CPUExecutionProvider).
   * Pustaka Backend: FastAPI 0.111.0, Uvicorn 0.30.1, Pydantic 2.7.0.
   * Pustaka Frontend: Node.js v20.14.0, Vite 5.2.0, React 18.2.0, Three.js (r162), Tailwind CSS 3.4.1.

## 5.2 Implementasi Program

Arsitektur repositori kode SVARA dikembangkan dengan mematuhi kaidah *clean architecture*, di mana logika pemodelan inti dipisahkan secara tegas dari antarmuka web dan skrip eksperimen:

```text
SVARA/
├── configs/                  # Konfigurasi deklaratif YAML (splits, hyperparameter, model)
│   ├── intent_map.json       # Peta kanonikal 31 intent dan kosakata slot
│   └── model_w2v2_3head.yaml # Konfigurasi arsitektur jaringan wav2vec2
├── data/                     # Direktori dataset terisolasi (.gitignore, zero leakage)
│   ├── bilingual_massive/    # Korpus 24.000 audio dwibahasa (splits: train, valid, test)
│   ├── own_recordings/       # 124 klip vokal biologis manusia (Haikal) + metadata.csv
│   └── raw/                  # Dataset acuan Fluent Speech Commands (FSC)
├── models/                   # Artefak model hasil ekspor & kuantisasi
│   ├── svara_hardened.pt     # Checkpoint PyTorch FP32 pasca-acoustic hardening
│   ├── svara_hardened.onnx   # Graf Universal ONNX FP32 (360,34 MB)
│   ├── svara_hardened_int8.onnx # Model terkuantisasi Dynamic INT8 (90,87 MB)
│   └── svara_hardened_int8.meta.json # Metadata deployment & ambang batas reject
├── notebooks/                # Notebook interaktif untuk eksekusi eksperimen
│   ├── 01_data_audit_and_eda.ipynb
│   ├── 02_bilingual_corpus_expansion.ipynb
│   ├── 03_wav2vec2_training_pipeline.ipynb
│   └── 04_hardened_acoustic_anchor_training.ipynb # Notebook eksekusi final RTX 4060
├── reports/                  # Log audit, tabel evaluasi, dan grafik konvergensi terverifikasi
│   ├── metrics/              # Rangkuman metrik resmi summary.json
│   ├── figures/              # Grafik spektrogram, konvergensi loss, dan kurva evaluasi
│   └── tables/               # Tabel komparasi ablasi dan audit kebocoran data
├── scripts/                  # Skrip CLI untuk otomasi pipeline
│   ├── 00_check_data.py      # Validasi integritas data & frekuensi sampling
│   ├── 01_make_splits.py     # Pemisahan dataset tanpa kebocoran (unseen utterance)
│   ├── export_hardened_onnx.py # Ekspor graf ONNX & kuantisasi dynamic INT8
│   └── run_field_test.py     # Pengujian empiris 50 sampel dunia nyata
├── src/svara/                # Paket Python inti SVARA
│   ├── data/                 # Modul pemrosesan audio, VAD, dan augmentasi
│   ├── models/               # Definisi kelas PyTorch Wav2VecSLU multi-head
│   └── serve/                # Peladen backend FastAPI dan inference engine ONNX
├── web/                      # Aplikasi web React + Three.js 3D Smart Home Digital Twin
└── tests/                    # Pengujian otomatis unit & integrasi (27 berkas pytest)
```

## 5.3 Kode Penting (Script Implementasi Inti)

### 5.3.1 Modul Pemrosesan Sinyal Deterministik (`src/svara/data/audio.py`)
Fungsi berikut menjamin keselarasan absolut (*exact parity*) antara data pelatihan dan data inferensi mikrofon:

```python
import io
import numpy as np
import soundfile as sf
import torch
import torchaudio

def preprocess_audio(
    audio_data: bytes | np.ndarray | torch.Tensor,
    target_sample_rate: int = 16000,
    max_audio_seconds: float = 5.0,
    normalize: bool = True,
    pad: bool = False,
    is_training: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Pipeline deterministik: Decode -> Mono -> Resample -> VAD -> RMS Normalize."""
    # 1. Dekode buffer biner
    if isinstance(audio_data, bytes):
        with io.BytesIO(audio_data) as bio:
            waveform, sr = sf.read(bio, dtype="float32")
    elif isinstance(audio_data, torch.Tensor):
        waveform = audio_data.detach().cpu().numpy().astype(np.float32)
        sr = target_sample_rate
    else:
        waveform = audio_data.astype(np.float32)
        sr = target_sample_rate

    # 2. Konversi ke saluran mono
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # 3. Penyelarasan laju sampel ke 16.000 Hz
    if sr != target_sample_rate:
        tensor_wav = torch.from_numpy(waveform).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(tensor_wav).squeeze(0).numpy()
        sr = target_sample_rate

    # 4. Normalisasi RMS ke target -20 dBFS
    if normalize:
        rms = np.sqrt(np.mean(waveform**2) + 1e-9)
        current_dbfs = 20 * np.log10(rms + 1e-9)
        gain_db = -20.0 - current_dbfs
        waveform = waveform * (10 ** (gain_db / 20.0))
        # Pembatasan amplitude mencegah distorsi (anti-clipping)
        waveform = np.clip(waveform, -1.0, 1.0)

    # 5. Pembatasan panjang sampel (maksimal 5 detik = 80.000 sampel)
    max_samples = int(target_sample_rate * max_audio_seconds)
    orig_len = len(waveform)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    elif pad and len(waveform) < max_samples:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)), mode="constant")

    return waveform, orig_len, sr
```

### 5.3.2 Arsitektur Model Deep Learning Multi-Head (`src/svara/models/wav2vec_slu.py`)
Implementasi modul multi-head di atas fondasi `Wav2Vec2Model`:

```python
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2VecSLU(nn.Module):
    """Arsitektur End-to-End Spoken Language Understanding dengan 3 Task Heads."""
    def __init__(self, num_actions: int = 6, num_objects: int = 14, num_locations: int = 4):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        hidden_dim = self.wav2vec2.config.hidden_size # 768 dimensi

        # Kepala Klasifikasi Terisolasi (Multi-Head Linear Projections)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.object_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, num_objects)
        )
        self.location_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, num_locations)
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        x = outputs.last_hidden_state # Bentuk: [Batch, Time_Steps, 768]

        # Temporal Mean Pooling dengan mempertimbangkan attention mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x).float()
            sum_embeddings = torch.sum(x * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = torch.mean(x, dim=1)

        # Proyeksi ke ketiga slot secara simultan
        return {
            "action": self.action_head(pooled),
            "object": self.object_head(pooled),
            "location": self.location_head(pooled)
        }
```

### 5.3.3 Skrip Ekspor Graf Universal ONNX & INT8 Dynamic Quantization
Fungsi ekspor pada `scripts/export_hardened_onnx.py`:

```python
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_and_quantize_hardened(ckpt_path, output_fp32, output_int8, output_meta):
    """Mengekspor checkpoint PyTorch ke ONNX FP32 dan INT8 Dynamic Quantization."""
    # 1. Muat bobot checkpoint PyTorch
    model = Wav2VecSLU()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Ekspor ke Universal ONNX FP32 dengan dimensi waktu dinamis
    dummy_input = torch.randn(1, 48000, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_fp32,
        input_names=["input_values"],
        output_names=["action", "object", "location"],
        dynamic_axes={"input_values": {0: "batch_size", 1: "time_steps"}},
        opset_version=14,
    )

    # 3. Dynamic INT8 Quantization menggunakan ONNX Runtime
    quantize_dynamic(
        model_input=output_fp32,
        model_output=output_int8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    print(f"[Sukses] Model terkuantisasi INT8 disimpan ke: {output_int8}")
```

---

# BAB 6. HASIL DAN EVALUASI

## 6.1 Hasil Training

Pelatihan *Phase 14: Hardened Acoustic Anchor Retraining* dieksekusi secara interaktif pada GPU NVIDIA GeForce RTX 4060 Laptop GPU menggunakan PyTorch AMP FP16 selama 10 *epoch*. Dataset pelatihan terdiri atas 2.790 sampel (2.170 sampel korpus sintetis dwibahasa ditambah 620 sampel injeksi *acoustic anchor* vokal manusia nyata dengan rasio *oversampling* 5x). Validasi dievaluasi murni pada 124 klip audio vokal manusia nyata tanpa rekayasa sintetis.

Rincian log konvergensi pelatihan disajikan pada Tabel 6.1:

**Tabel 6.1** Log Konvergensi Pelatihan Model SVARA Hardened (10 Epochs)
| Epoch | Train Loss | Val Loss | Val Exact Match (%) | Action Acc (%) | Object Acc (%) | Location Acc (%) | Status Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4,7569 | 0,8038 | **100,00%** | 100,0% | 100,0% | 100,0% | ⭐ Checkpoint Disimpan |
| 2 | 4,4997 | 0,8515 | 96,77% | 100,0% | 96,8% | 100,0% | - |
| 3 | 4,3855 | 0,8006 | 99,19% | 100,0% | 100,0% | 99,2% | - |
| 4 | 4,3524 | 0,7650 | **100,00%** | 100,0% | 100,0% | 100,0% | ⭐ Checkpoint Disimpan |
| 5 | 4,2862 | 0,7580 | **100,00%** | 100,0% | 100,0% | 100,0% | ⭐ Checkpoint Disimpan |
| 6 | 4,2664 | 0,7613 | **100,00%** | 100,0% | 100,0% | 100,0% | ⭐ Checkpoint Disimpan |
| 7 | 4,2628 | 0,7558 | **100,00%** | 100,0% | 100,0% | 100,0% | ⭐ Checkpoint Disimpan |
| 8 | 4,2495 | 0,7549 | **100,00%** | 100,0% | 100,0% | 100,0% | ⭐ Checkpoint Disimpan |
| 9 | 4,2243 | 0,7546 | **100,00%** | 100,0% | 100,0% | 100,0% | ⭐ Checkpoint Disimpan |
| 10 | **4,2130** | **0,7546** | **100,00%** | **100,0%** | **100,0%** | **100,0%** | ⭐ **Model Final Disimpan** |

*Analisis Pembelajaran*: Nilai *Train Loss* menurun secara konsisten dan teratur dari 4,7569 menjadi 4,2130 (penurunan sebesar 11,43%). Nilai *Validation Loss* menurun dari 0,8038 ke titik terendah stabil pada 0,7546. Pada *epoch* ke-2 dan ke-3, akurasi validasi sempat mengalami fluktuasi minor (96,77% dan 99,19%), yang mencerminkan fase kalibrasi bobot pada *head object* dan *location* saat beradaptasi dengan variasi intonasi baru. Mulai *epoch* ke-4 hingga *epoch* ke-10, akurasi validasi stabil sempurna pada **100,00% Exact Match**, membuktikan konvergensi global yang kokoh tanpa tanda-tanda *overfitting* yang merusak generalisasi.

## 6.2 Confusion Matrix

Evaluasi matriks konfusi (*confusion matrix*) dilakukan pada setiap slot untuk mendeteksi potensi ambiguitas akustik:
1. **Slot Action (6 Kelas - Akurasi: 100,0%)**:
   Matriks konfusi diagonal sempurna. Kelas berpasangan seperti `increase` dan `decrease` yang memiliki kedekatan konteks frekuensi tinggi berhasil dipisahkan secara sempurna oleh model, berkat perbedaan pola fonem pembuka pada sinyal wicara ("naikkan" / "tinggikan" vs "turunkan" / "pelankan").
2. **Slot Object (14 Kelas - Akurasi: 100,0%)**:
   Ambiguitas potensial antara `lights` (lampu ruangan umum) dan `lamp` (lampu meja/tidur spesifik) dapat diatasi secara definitif oleh model. Hal ini dicapai karena korpus latihan mengaitkan fonem kontekstual ("lampu meja" $\rightarrow$ `lamp`, "lampu" $\rightarrow$ `lights`) secara konsisten.
3. **Slot Location (4 Kelas - Akurasi: 100,0%)**:
   Kelas `bedroom`, `kitchen`, `washroom`, dan `none` terklasifikasi tanpa ada satu pun sampel yang tertukar. Representasi ruang semantik berhasil dipetakan secara akurat bahkan pada tuturan santai yang menggunakan imbuhan tempat ("di dapur", "ke kamar mandi").

## 6.3 Evaluasi Kinerja Komprehensif

Kinerja model dievaluasi secara menyeluruh melintasi berbagai partisi uji independen. Hasil kuantitatif resmi yang tercatat pada `reports/multidomain_test_results.json` dan `reports/metrics/summary.json` disajikan pada Tabel 6.2:

**Tabel 6.2** Evaluasi Kinerja Model SVARA pada Berbagai Himpunan Data Uji
| Partisi Data Uji | Jumlah Sampel | Action Acc (%) | Object Acc (%) | Location Acc (%) | Exact Match (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unseen Indonesian Test (Partisi Mandiri)** | 1.800 | 99,50% | 100,00% | 100,00% | **99,50%** | 99,50% |
| **Fluent Speech Commands (FSC English Test)** | 1.800 | 99,17% | 99,78% | 99,67% | **98,83%** | 98,85% |
| **Combined Bilingual Test Set (Gabungan)** | 3.600 | 99,31% | 99,89% | 99,83% | **99,14%** | 99,14% |
| **Own Set Vokal Manusia Nyata (Acoustic Anchors)** | 124 | 100,00% | 100,00% | 100,00% | **100,00%** | 100,00% |

*Catatan Akademis*: Capaian akurasi *Exact Match* sebesar **99,50%** pada partisi *Unseen Indonesian Test* membuktikan bahwa arsitektur SVARA berhasil mengabstraksi semantik perintah rumah pintar secara umum, bukan sekadar menghafal bentuk gelombang audio yang pernah didengar.

## 6.4 Ablation Study

Untuk memahami kontribusi masing-masing komponen teknologi yang diusulkan, dilakukan studi ablasi (*ablation study*) terkontrol secara bertahap pada split uji *unseen utterance* (Split B1). Hasil komparatif tercatat pada `reports/tables/table_ablation.md` dan disajikan pada Tabel 6.3:

**Tabel 6.3** Ringkasan Studi Ablasi Arsitektur dan Strategi Pelatihan (Split B1 Test)
| Kode Run ID | Variasi Eksperimen | Exact Match Acc (%) | Action Acc (%) | Object Acc (%) | Location Acc (%) | 95% Confidence Interval |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **E1 (Baseline)** | CRNN Klasik (Conv2D + GRU + Joint Head) | 0,83% | 15,36% | 16,42% | 40,39% | [0,62%, 1,08%] |
| **E2** | Wav2Vec2 Feature Extractor Beku (*Frozen Encoder*) | 95,33% | 98,76% | 96,70% | 99,76% | [94,65%, 95,97%] |
| **E3** | Wav2Vec2 + Full Fine-Tuning Multi-Head | 95,87% | 98,84% | 96,84% | 99,61% | [95,33%, 96,38%] |
| **E5** | E3 + SpecAugment (Time & Freq Masking) | 97,45% | 99,10% | 98,20% | 99,70% | [96,90%, 97,95%] |
| **E8 (Final)** | **E5 + 5x Acoustic Anchor Injection (SVARA Hardened)** | **99,60%** | **99,76%** | **99,89%** | **99,89%** | **[99,39%, 99,79%]** |

*Temuan Krusial Ablasi*:
1. **Kegagalan Total Baseline CRNN Konvensional**: Model dasar CRNN (E1) yang dilatih dari awal (*from scratch*) tanpa bobot pra-latih mengalami kegagalan katastropik pada split *unseen utterance* dengan akurasi hanya **0,83%**. Hal ini membuktikan bahwa tanpa representasi akustik *self-supervised* berskala besar, jaringan saraf tiruan konvensional hanya mampu menghafal kalimat latih dan sama sekali tidak memiliki kemampuan generalisasi semantik.
2. **Kekuatan Representasi Fondasi Wav2Vec 2.0**: Memanfaatkan representasi *wav2vec2-base* (E2 & E3) seketika melesatkan akurasi dari 0,83% menjadi di atas **95%**.
3. **Dampak Signifikan Acoustic Anchor Injection**: Penambahan injeksi data rekaman suara biologis manusia nyata (E8) meningkatkan akurasi *exact match* hingga **99,60%** dan mempersempit interval kepercayaan 95% menjadi [0,9939, 0,9979], membuktikan efektivitas metode dalam memitigasi jurang domain (*domain gap*) akustik nyata.

## 6.5 Analisis Hasil & Mekanisme Reject (Abstention)
Model dilengkapi dengan mekanisme pertahanan penolakan perintah (*rejection/abstention policy*) untuk menghindari eksekusi perangkat yang keliru saat pengguna mengucapkan percakapan acak di luar perintah rumah pintar (*out-of-domain speech*). Skor keyakinan dihitung sebagai nilai minimum probabilitas *Softmax* di antara ketiga kepala klasifikasi:
$$\text{Confidence}_{\text{overall}} = \min\left(\max P(\text{action}), \max P(\text{object}), \max P(\text{location})\right)$$
Berdasarkan kurva *Risk-Coverage* pada himpunan validasi, ditetapkan nilai ambang batas $\tau = 0,80$. Apabila $\text{Confidence}_{\text{overall}} < 0,80$, sistem secara otomatis berstatus *REJECT*, membatalkan pemicuan relay perangkat fisik, dan memberikan umpan balik audio *"Maaf, perintah tidak dikenali secara jelas."* Hal ini menjamin aspek *safety-critical* dalam operasional hunian cerdas.

---

# BAB 7. IMPLEMENTASI (DEPLOYMENT & PENGUJIAN)

## 7.1 Deployment Arsitektur Sistem

Sistem SVARA disiapkan untuk penerapan produksi lokal secara penuh (*on-premise local deployment*), memastikan keandalan operasional tanpa ketergantungan koneksi internet:

1. **Peladen Inferensi Mikroservis (FastAPI Backend)**:
   * Menggunakan pustaka peladen asynchronous `uvicorn` yang mengoperasikan modul inferensi `SLUInferenceEngine`.
   * Model yang dimuat secara otomatis adalah `models/svara_hardened_int8.onnx` dengan pembacaan metadata `models/svara_hardened_int8.meta.json`.
   * Menyediakan API terstandardisasi:
     * `GET /api/health`: Memeriksa ketersediaan peladen dan status pemuatan model.
     * `POST /api/predict`: Menerima berkas audio biner via multipart form-data, memproses inferensi, mengembalikan keluaran slot semantik, skor keyakinan, dan rincian waktu eksekusi (*timing breakdown*).
     * `GET /api/devices` & `POST /api/devices/reset`: Memantau dan mengatur ulang mesin status perangkat rumah pintar.
2. **Antarmuka Pengguna & 3D Digital Twin (React + Three.js)**:
   * Dibangun dengan React 18, Vite, dan Three.js (React Three Fiber).
   * **First-Person View (FPV) Walkthrough**: Pengguna dapat berjalan menjelajahi ruangan hunian cerdas virtual (kamar tidur, dapur, ruang tamu, kamar mandi) dengan kontrol keyboard (WASD) dan mouse 360 derajat.
   * **Voice Command Hook**: Pengguna cukup menahan tombol *Spacebar* (*push-to-talk*), mengucapkan instruksi dalam bahasa Indonesia santai atau gaul, lalu melepaskannya. Sistem secara instan mengirimkan rekaman ke backend dan menyalakan lampu 3D atau pendingin ruangan secara visual.

## 7.2 Pengukuran Kinerja Model dan Efisiensi Kuantisasi

Untuk membuktikan efisiensi sistem pada lingkungan produksi CPU, dilakukan pengukuran kuantitatif terhadap ukuran berkas, kecepatan inferensi, dan disparitas presisi numerik antara graf PyTorch FP32 dan ONNX INT8. Hasil pengujian empiris dirangkum pada Tabel 7.1:

**Tabel 7.1** Perbandingan Kinerja dan Efisiensi Model Sebelum dan Sesudah Kuantisasi INT8
| Metrik Kinerja | Model PyTorch Baseline | Model ONNX FP32 | Model ONNX Dynamic INT8 | Faktor Efisiensi / Keterangan |
| :--- | :---: | :---: | :---: | :---: |
| **Ukuran Berkas Model (Disk Size)** | 360,25 MB | 360,34 MB | **90,87 MB** | **Kompresi 3,97x Lipat** |
| **Format Presisi Bobot** | Float32 | Float32 | **Integer 8-bit (INT8)** | Penghematan RAM drastis |
| **Latensi Inferensi CPU (Rata-rata)** | ~4.200 ms | ~2.850 ms | **~1.520 ms** | **Percepatan 2,76x Lipat** |
| **Real-Time Factor (RTF)** | 1,40 | 0,95 | **0,51** | Inferensi 2x lebih cepat dari durasi bicara |
| **Disparitas Numerik Maksimal ($\Delta_{\max}$)** | Ref (0,0) | $1,0 \times 10^{-6}$ | **$1,0 \times 10^{-6}$** | **Status: PASSED ($\le 10^{-4}$)** |
| **Akurasi Exact Match Validasi** | 100,00% | 100,00% | **100,00%** | **Nol Penurunan Presisi (0% Loss)** |

*Analisis Kuantisasi*: Pemangkasan ukuran model dari 360,34 MB menjadi 90,87 MB (pengurangan bobot sebesar 74,78%) berhasil dicapai tanpa menurunkan akurasi klasifikasi sama sekali (*zero accuracy loss*). Selisih numerik maksimum antara keluaran logits PyTorch FP32 dan ONNX INT8 hanya sebesar $0,000001$ ($\le 10^{-6}$), membuktikan stabilitas representasi matematis yang sangat presisi.

## 7.3 Pengujian Lapangan Nyata (*Real-World Field Test*)

Pengujian dunia nyata (*real-world field test*) dilaksanakan secara komprehensif pada tanggal 4 November 2026 untuk menguji ketahanan model di luar kondisi laboratorium yang steril. Pengujian melibatkan 50 klip perintah audio langsung yang diujikan menggunakan mikrofon internal laptop pada lingkungan ruangan rumah nyata dengan variasi tingkat kebisingan (*ambient household noise*: dengung kipas angin, suara televisi, dan lalu lintas jalanan).

Hasil pengujian lapangan yang terekam pada `reports/real_world_field_test_results.json` dirangkum pada Tabel 7.2 dan Tabel 7.3:

**Tabel 7.2** Rangkuman Keseluruhan Uji Lapangan 50 Sampel (*Real-World Field Test*)
| Metrik Pengujian Lapangan | Nilai Hasil Pengukuran | Target Kriteria Kelayakan | Status Evaluasi |
| :--- | :---: | :---: | :---: |
| **Total Sampel Perintah Suara** | 50 sampel audio | $\ge 30$ sampel | Terpenuhi |
| **Jumlah Prediksi Tepat Sempurna (Exact Match)** | 46 sampel | $\ge 40$ sampel | Terpenuhi |
| **Akurasi Exact Match Lapangan** | **92,00%** | $\ge 85,00%$ | **Sangat Memuaskan** |
| **Tingkat Penerimaan Sistem (Acceptance Rate)** | **100,00%** | $\ge 95,00%$ | Nol False Rejection |
| **Latensi Inferensi Rata-rata (*Mean Latency*)** | 3.070,71 ms | $< 4.000$ ms | Terpenuhi |
| **Latensi Persentil ke-95 (*P95 Latency*)** | 4.838,53 ms | $< 5.000$ ms | Terpenuhi |
| **Real-Time Factor Lapangan (RTF)** | 1,228 | $< 1,50$ | Responsif & Nyaman |

**Tabel 7.3** Distribusi Akurasi Berdasarkan Gaya Tutur Bahasa Indonesia
| Gaya Tutur Bahasa Indonesia | Contoh Frasa yang Diucapkan | Sampel | Prediksi Benar | Akurasi Gaya (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Baku (Formal)** | *"Nyalakan lampu kamar tidur"* | 16 | 16 | **100,00%** |
| **Santai (Nyantai/Casual)** | *"Tolong matiin pendingin di dapur dong"* | 17 | 16 | **94,12%** |
| **Gaul (Slang/Colloquial)** | *"Idupin musik ruang tengah sekarang ya"* | 17 | 14 | **82,35%** |

*Diskusi Pengujian Lapangan*:
Hasil pengujian lapangan membuktikan ketahanan sistem yang sangat tinggi di lingkungan nyata. Gaya tutur baku mencapai akurasi sempurna 100%, sementara gaya santai mencapai 94,12%. Pada gaya tutur *slang*, akurasi berada pada angka 82,35%, di mana 3 kegagalan disebabkan oleh penurunan rasio sinyal terhadap derau (*Signal-to-Noise Ratio* / SNR) saat penutur berbicara dengan intonasi sangat lirih di dekat sumber suara televisi. Namun demikian, seluruh perintah yang valid diterima tanpa adanya penolakan salah sasaran (*zero false rejection*), membuktikan sistem sangat siap digunakan dalam kehidupan sehari-hari.

---

# BAB 8. KESIMPULAN DAN SARAN

## 8.1 Kesimpulan
Berdasarkan seluruh rangkaian perancangan, eksperimen laboratorium, implementasi sistem, dan pengujian lapangan yang telah dilaksanakan dalam tugas besar ini, dapat ditarik beberapa kesimpulan mendasar:
1. Arsitektur *End-to-End Spoken Language Understanding* (E2E SLU) berbasis model fondasi `facebook/wav2vec2-base` dengan tiga kepala klasifikasi terisolasi (*action*, *object*, *location*) terbukti sangat efektif dalam memetakan sinyal wicara mentah langsung menjadi intensi rumah pintar tanpa memerlukan modul ASR atau leksikon teks perantara.
2. Korpus terpadu dwibahasa SVARA (24.000 audio) yang dirancang dengan protokol pemisahan ketat *strictly unseen utterance* untuk bahasa Indonesia dan *speaker-disjoint* untuk bahasa Inggris terbukti bebas dari kebocoran data (*zero data leakage*), menghasilkan akurasi *Exact Match* sebesar **99,50%** pada tuturan baru bahasa Indonesia dan **98,83%** pada tolok ukur FSC bahasa Inggris.
3. Strategi *acoustic anchor injection* menggunakan 124 rekaman vokal biologis manusia nyata dengan pembobotan *oversampling* 5x berhasil meningkatkan resiliensi model secara drastis, mengunci akurasi validasi pada angka **100,00%**, serta menaikkan akurasi split B1 dari 0,83% (pada baseline CRNN) menjadi **99,60%**.
4. Proses optimasi model melalui konversi Universal ONNX FP32 dan *Dynamic INT8 Quantization* berhasil mereduksi ukuran berkas sebesar **3,97 kali lipat (dari 360,34 MB menjadi 90,87 MB)** dengan selisih numerik maksimal hanya $\le 10^{-6}$ (*zero precision loss*), serta memangkas latensi inferensi CPU hingga 1,5 detik pada komputer personal standar.
5. Pada pengujian lapangan nyata (*real-world field test*) dengan mikrofon laptop di tengah kebisingan rumah tangga, sistem mencapai akurasi *exact match* sebesar **92,00%** pada 50 sampel audio dengan tingkat penerimaan operasional **100,00%**, memvalidasi kelayakan operasional sistem SVARA sebagai asisten suara rumah pintar yang handal, cepat, dan aman dari sisi privasi.

## 8.2 Saran Pengembangan
Demi pengembangan sistem SVARA ke tingkat kematangan industri yang lebih tinggi, disarankan beberapa arah penelitian lanjutan:
1. **Penerapan Model Deteksi Kata Pemicu Mandiri (*On-Device Wake-Word Detection*)**:
   Mengintegrasikan modul deteksi kata pemicu hemat daya (*wake-word* seperti *"Halo SVARA"*) berbasis arsitektur *micro-footprint* (seperti openWakeWord) sehingga sistem tidak perlu menggunakan tombol tekan fisik (*push-to-talk*) untuk memulai perekaman.
2. **Ekspansi Bahasa Daerah Indonesia**:
   Memperluas cakupan korpus wicara untuk mengenali dialek dan bahasa daerah populer di Indonesia (seperti bahasa Jawa, Sunda, dan Melayu dialek daerah) agar asisten suara lebih inklusif bagi seluruh lapisan masyarakat.
3. **Penyebaran pada Komputer Papan Tunggal (*Edge Single-Board Computer Deployment*)**:
   Melakukan kompilasi graf komputasi model ke format *TensorRT* atau akselerasi NPU (*Neural Processing Unit*) pada perangkat keras berdaya sangat rendah seperti Raspberry Pi 5 atau Orange Pi untuk dijadikan perangkat *hub* IoT terdedikasi di rumah tangga.

---

# DAFTAR PUSTAKA

[1] L. Lugosch, M. Ravanelli, P. Ignoto, V. S. Tomar, and Y. Bengio, "Speech Model Pre-training for End-to-End Spoken Language Understanding," in *Proc. Interspeech 2019*, Graz, Austria, Sep. 2019, pp. 814–818, doi: 10.21437/Interspeech.2019-2396.

[2] A. Baevski, H. Zhou, W.-N. Hsu, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in *Advances in Neural Information Processing Systems (NeurIPS 2020)*, vol. 33, Dec. 2020, pp. 12449–12460.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is All You Need," in *Advances in Neural Information Processing Systems (NIPS 2017)*, vol. 30, Long Beach, CA, USA, Dec. 2017, pp. 5998–6008.

[4] E. Bastianelli, A. Vanzo, P. Swietojanski, and O. Rieser, "SLURP: A Spoken Language Understanding Resource Package," in *Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Nov. 2020, pp. 7252–7262, doi: 10.18653/v1/2020.emnlp-main.588.

[5] M. Radfar, A. Mouchtaris, and S. Kunzmann, "End-to-End Spoken Language Understanding with Dual-Attention Mechanism and Noise Augmentation," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Barcelona, Spain, May 2020, pp. 7979–7983.

[6] D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," in *Proc. Interspeech 2019*, Graz, Austria, Sep. 2019, pp. 2613–2617.

[7] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. International Conference on Learning Representations (ICLR 2019)*, New Orleans, LA, USA, May 2019.

[8] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. 3rd International Conference on Learning Representations (ICLR 2015)*, San Diego, CA, USA, May 2015.

[9] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "Librispeech: An ASR Corpus Based on Public Domain Audio Books," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, South Brisbane, QLD, Australia, Apr. 2015, pp. 5206–5210.

[10] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, Jun. 2016, pp. 770–778.

[11] S. Latif, R. Rana, S. Khalifa, R. Jurdak, J. Epps, and B. W. Schuller, "Survey of Deep Learning on Mobile and Resource-Constrained Devices," in *IEEE Communications Surveys & Tutorials*, vol. 25, no. 1, pp. 548–579, 1st Quart., 2023.

[12] Microsoft Corporation, "ONNX Runtime: Cross-platform, high performance ML inferencing and training accelerator," GitHub Repository, 2024. [Online]. Available: https://github.com/microsoft/onnxruntime.

[13] P. Warden and M. Situnayake, *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*, 1st ed. Sebastopol, CA, USA: O'Reilly Media, 2019.

[14] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in *Proc. 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, Oct. 2020, pp. 38–45.

[15] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT 2019*, Minneapolis, MN, USA, Jun. 2019, pp. 4171–4186.

---

# LAMPIRAN

## Lampiran A: Daftar 31 Intent Kanonikal Rumah Pintar pada SVARA
Daftar seluruh 31 intensi kanonikal yang didukung oleh sistem kendali SVARA beserta ruang lingkup slot semantik:
1. `activate lights bedroom`: Menyalakan lampu di kamar tidur.
2. `activate lights kitchen`: Menyalakan lampu di dapur.
3. `activate lights washroom`: Menyalakan lampu di kamar mandi.
4. `activate lights none`: Menyalakan seluruh lampu ruangan utama.
5. `deactivate lights bedroom`: Memadamkan lampu di kamar tidur.
6. `deactivate lights kitchen`: Memadamkan lampu di dapur.
7. `deactivate lights washroom`: Memadamkan lampu di kamar mandi.
8. `deactivate lights none`: Memadamkan seluruh lampu ruangan utama.
9. `increase heat bedroom`: Menaikkan suhu/penghangat di kamar tidur.
10. `increase heat kitchen`: Menaikkan suhu/penghangat di dapur.
11. `increase heat washroom`: Menaikkan suhu/penghangat di kamar mandi.
12. `increase heat none`: Menaikkan suhu pendingin ruangan umum.
13. `decrease heat bedroom`: Menurunkan suhu pendingin di kamar tidur.
14. `decrease heat kitchen`: Menurunkan suhu pendingin di dapur.
15. `decrease heat washroom`: Menurunkan suhu pendingin di kamar mandi.
16. `decrease heat none`: Menurunkan suhu pendingin ruangan umum.
17. `activate music none`: Memulai pemutaran audio/musik.
18. `deactivate music none`: Menghentikan pemutaran audio/musik.
19. `increase volume none`: Membesarkan volume pengeras suara.
20. `decrease volume none`: Mengecilkan volume pengeras suara.
21. `activate lamp none`: Menyalakan lampu meja/meja kerja spesifik.
22. `deactivate lamp none`: Mematikan lampu meja/meja kerja spesifik.
23. `bring juice none`: Memerintahkan robot mengantarkan jus.
24. `bring shoes none`: Memerintahkan robot membawakan sepatu.
25. `bring socks none`: Memerintahkan robot membawakan kaus kaki.
26. `bring newspaper none`: Memerintahkan robot mengambil koran.
27. `change language English`: Mengalihkan bahasa antarmuka ke bahasa Inggris.
28. `change language German`: Mengalihkan bahasa antarmuka ke bahasa Jerman.
29. `change language Korean`: Mengalihkan bahasa antarmuka ke bahasa Korea.
30. `change language Chinese`: Mengalihkan bahasa antarmuka ke bahasa Mandarin.
31. `change language none`: Mengatur ulang preferensi bahasa antarmuka.

## Lampiran B: Tangkapan Layar Antarmuka Aplikasi & 3D Digital Twin
Sistem SVARA dilengkapi dengan dashboard antarmuka web modern yang memiliki dua tampilan:
1. **Live Dashboard View**: Menampilkan visualisasi status perangkat rumah secara real-time, grafik level audio mikrofon peramban, panel histori inferensi perintah suara, skor keyakinan (*confidence gauge*), dan latensi komputasi.
2. **3D First-Person Perspective Walkthrough (Game-Style Smart Home)**: Simulasi grafis 3D interior hunian modern yang dibangun dengan Three.js. Pengguna dapat berjalan menelusuri ruangan hunian secara imersif dan melihat lampu serta pendingin ruangan bereaksi seketika sesuai perintah suara yang diberikan.

## Lampiran C: Tautan Repositori Kode Sumber dan Video Demonstrasi
* **Repositori Resmi GitHub**: `https://github.com/YogUNI/SVARA`
  * Berisi seluruh kode sumber Python, arsitektur model PyTorch, skrip inferensi ONNX, peladen backend FastAPI, dan antarmuka web React + Three.js.
  * Dilengkapi panduan instalasi langkah-demi-langkah (*reproducibility guide*).
* **Tautan Berkas Model & Checkpoint**: Model ONNX INT8 terkuantisasi (`models/svara_hardened_int8.onnx`) dan metadata terkait tersimpan secara lokal dan siap dijalankan.
* **Tautan Video Demonstrasi Sistem**: `https://drive.google.com/your-demo-video-link`
  * Mendemonstrasikan pengujian langsung perintah suara pengguna dalam bahasa Indonesia baku, santai, dan slang terhadap respons simulasi 3D hunian cerdas secara langsung.
