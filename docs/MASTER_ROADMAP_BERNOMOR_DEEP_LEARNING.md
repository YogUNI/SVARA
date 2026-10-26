# MASTER ROADMAP: Eksekusi Ilmiah Pematangan Deep Learning SLU SVARA (Terstruktur & Bernomor)

Dokumen ini adalah panduan kontrol proyek resmi untuk pematangan sistem Deep Learning **SVARA (Smart Voice Assistant for Residential Automation)**. Setiap fase diberi kode hierarkis bernomor (`Fase X.Y`) agar eksekusi dapat dipantau satu per satu secara tuntas tanpa ada tahapan yang terlewat.

> **Kebijakan Eksekusi**:
> - Eksekusi dilakukan **1 sub-step demi 1 sub-step** secara fokus dan maksimal.
> - Selesai mengerjakan 1 sub-step, hasil dan artefak **wajib langsung di-commit dan di-push ke branch `main` GitHub milik Yoga (`YogUNI/SVARA`)**.
> - File notebook `notebooks/02_indonesian_bilingual_training.ipynb` **tetap dipertahankan (tidak diubah)**. Master notebook baru akan dibuat pada **Fase 5.1** di `notebooks/03_production_slu_deep_learning_pipeline.ipynb`.

---

## DAFTAR HIERARKI TAHAPAN EKSEKUSI (MASTER CHECKLIST)

```
├── FASE 1: REKAYASA KORPUS NLP BAHASA INDONESIA SEHARI-HARI (TARGET 4.000 - 5.000 FRASA)
│   ├── Sub-step 1.1: Pembuatan Generator Matriks Leksikon, Slang, & Partikel Spontan (Python Script)
│   ├── Sub-step 1.2: Eksekusi Generasi Korpus Teks 31 Intent (130-150 Frasa per Intent)
│   └── Sub-step 1.3: Audit & Validasi Integritas Linguistik (Deduplikasi, Cek Sebaran, Ekspor CSV)
│
├── FASE 2: SINTESIS SUARA MULTI-SPEAKER MASSAL (NEURAL TTS TARGET 8.000 - 10.000 AUDIO BERSIH)
│   ├── Sub-step 2.1: Pembuatan Engine Sintesis Paralel Multi-Worker (Edge Neural TTS Python)
│   ├── Sub-step 2.2: Eksekusi Sintesis Suara Pria (id-ID-ArdiNeural) & Wanita (id-ID-GadisNeural)
│   └── Sub-step 2.3: Verifikasi Kualitas Sinyal Audio Bersih (16 kHz 16-bit PCM Mono, Zero-Corruption Check)
│
├── FASE 3: ENVIRONMENTAL ACOUSTIC AUGMENTATION (SIMULASI DUNIA NYATA TARGET 24.000 AUDIO)
│   ├── Sub-step 3.1: Implementasi Modul Schroeder Reverb (Akustik Kamar Tidur & Ruang Tamu)
│   ├── Sub-step 3.2: Implementasi Modul Hardware Mic Filter & Injeksi Noise Kipas/AC (SNR 16-26 dB)
│   ├── Sub-step 3.3: Implementasi Variasi Tempo Bicara (Speed Perturbation 0.92x - 1.08x)
│   └── Sub-step 3.4: Eksekusi Batch Pembuatan 24.000 Audio Robust Indonesia
│
├── FASE 4: INTEGRASI BALANCED BILINGUAL DATASET & SPLITTING ILMIAH (24.000 AUDIO SEIMBANG)
│   ├── Sub-step 4.1: Penggabungan Seimbang 50:50 (12.000 Audio Indonesia + 12.000 Audio FSC English)
│   ├── Sub-step 4.2: Pembuatan Stratified Split (Train 70%, Valid 15%, Unseen Test 15%)
│   └── Sub-step 4.3: Audit Verifikasi Bebas Kebocoran Data (Zero-Data-Leakage Validation)
│
├── FASE 5: PEMBUATAN MASTER NOTEBOOK BARU & ARSITEKTUR MODEL
│   ├── Sub-step 5.1: Inisialisasi Master Notebook `notebooks/03_production_slu_deep_learning_pipeline.ipynb`
│   ├── Sub-step 5.2: Desain Sel 1-4 (CUDA GPU 4060, Waveform Spectrogram, DataLoader, Model Head)
│   └── Sub-step 5.3: Desain Sel 5-9 (Interactive Training, Kurva Loss, Evaluasi Ilmiah, Widget Live Mic)
│
├── FASE 6: TRAINING & OPTIMASI TERKALIBRASI PADA GPU NVIDIA RTX 4060
│   ├── Sub-step 6.1: Konfigurasi Loss Multi-Task dengan Label Smoothing (ε = 0.05) & Optimizer AdamW
│   ├── Sub-step 6.2: Eksekusi Fine-Tuning FP16 AMP pada GPU RTX 4060 (Monitoring Loss & Exact-Match)
│   └── Sub-step 6.3: Simpan Checkpoint Terbaik (Best Model Checkpoint) di `reports/runs/`
│
├── FASE 7: UNCERTAINTY QUANTIFICATION, TEMPERATURE SCALING & CONSISTENCY GATE
│   ├── Sub-step 7.1: Kalibrasi Probabilitas Softmax dengan Temperature Scaling (T-Scaling)
│   ├── Sub-step 7.2: Implementasi Geometric Mean Confidence Scoring
│   └── Sub-step 7.3: Validasi Tata Bahasa Cerdas (31 Intent Legal Gate / Abstention Filter)
│
├── FASE 8: EXPORT GRAPH ONNX, DYNAMIC INT8 QUANTIZATION & PARITY CHECK
│   ├── Sub-step 8.1: Serialisasi Graf PyTorch ke Format Universal ONNX
│   ├── Sub-step 8.2: Dynamic INT8 Quantization (Reduksi Ukuran 360 MB -> 90 MB)
│   └── Sub-step 8.3: Verifikasi Paritas Numerik PyTorch vs ONNX (Selisih Max Diff < 1e-5)
│
└── FASE 9: INTEGRASI SISTEM WEB SERVING & LIVE TEST MIKROFON FISIK
    ├── Sub-step 9.1: Konfigurasi Port Bebas Konflik (FastAPI Port 8001 & Proxy Vite Port 5180)
    ├── Sub-step 9.2: Pengujian Otomatis Seluruh Intent pada Server Aktif
    └── Sub-step 9.3: Uji Coba Langsung Suara Fisik di Visualizer 2D/3D Smart Home
```

---

## DETAIL OPERASIONAL PER SUB-STEP

### FASE 1: REKAYASA KORPUS NLP BAHASA INDONESIA (TARGET 4.000 - 5.000 FRASA)
* **Sub-step 1.1**: Buat script `scripts/generate_massive_indonesian_corpus.py` yang memuat kamus leksikon lengkap (sinonim aksi gaul, kata benda, nama ruangan, partikel spontan, dan 6 pola kalimat terbalik).
* **Sub-step 1.2**: Eksekusi script untuk menghasilkan 130 - 150 variasi frasa unik per intent untuk ke-31 intent ($31 \times 135 \approx 4.200\text{ frasa}$).
* **Sub-step 1.3**: Audit duplikasi teks, pastikan semua intent memiliki jumlah sampel merata, simpan di `data/corpus/indonesian_smart_home_corpus_4k.csv`.

### FASE 2: SINTESIS SUARA MULTI-SPEAKER MASSAL (NEURAL TTS 8.000 - 10.000 AUDIO)
* **Sub-step 2.1**: Buat script `scripts/synthesize_massive_indonesian.py` dengan arsitektur async paralel (`asyncio.Semaphore(15)`).
* **Sub-step 2.2**: Generate 4.200 frasa $\times$ 2 suara (`id-ID-ArdiNeural` Pria & `id-ID-GadisNeural` Wanita) $\rightarrow 8.400$ audio klip.
* **Sub-step 2.3**: Verifikasi seluruh audio tersimpan dalam format 16 kHz 16-bit Mono WAV di `data/synthetic_indonesian_massive/wavs_16k/`.

### FASE 3: ENVIRONMENTAL ACOUSTIC AUGMENTATION (24.000 AUDIO ROBUST)
* **Sub-step 3.1**: Terapkan Schroeder Reverberator untuk kamar tidur dan ruang keluarga.
* **Sub-step 3.2**: Terapkan filter mic laptop internal (Highpass 100 Hz) dan injeksi noise AC/kipas (SNR 16-26 dB).
* **Sub-step 3.3**: Terapkan variasi tempo bicara ($0.92\times, 1.0\times, 1.08\times$).
* **Sub-step 3.4**: Generate 3 variasi per audio bersih $\rightarrow 8.400 \times 3 = 25.200$ audio klip robust di `data/synthetic_indonesian_massive/wavs_augmented/`.

### FASE 4: INTEGRASI BALANCED BILINGUAL DATASET & SPLITTING ILMIAH
* **Sub-step 4.1**: Gabungkan 12.000 audio Indonesia robust + 12.000 audio FSC English native (Total 24.000 audio).
* **Sub-step 4.2**: Buat split stratified: Train (70% = 16.800 audio), Valid (15% = 3.600 audio), Unseen Test (15% = 3.600 audio).
* **Sub-step 4.3**: Simpan di `data/bilingual_massive/splits/` dan verifikasi hash duplikasi.

### FASE 5: PEMBUATAN MASTER NOTEBOOK BARU & ARSITEKTUR MODEL
* **Sub-step 5.1**: Buat `notebooks/03_production_slu_deep_learning_pipeline.ipynb` dari nol.
* **Sub-step 5.2**: Lengkapi Sel 1-4 untuk inspeksi GPU, spectrogram audio, dan arsitektur model.
* **Sub-step 5.3**: Lengkapi Sel 5-9 untuk live training, kurva loss interaktif, evaluasi, dan widget prediksi suara.

### FASE 6: TRAINING & OPTIMASI TERKALIBRASI PADA GPU NVIDIA RTX 4060
* **Sub-step 6.1**: Konfigurasi loss multi-task dengan label smoothing $\epsilon=0.05$ dan learning rate schedule.
* **Sub-step 6.2**: Jalankan training loop dengan PyTorch AMP FP16 di RTX 4060 sampai loss konvergen optimal.
* **Sub-step 6.3**: Simpan model bobot terbaik di `reports/runs/massive_bilingual_w2v2/best.ckpt`.

### FASE 7: UNCERTAINTY QUANTIFICATION, TEMPERATURE SCALING & CONSISTENCY GATE
* **Sub-step 7.1**: Hitung skala suhu $T$ untuk kalibrasi probabilitas softmax.
* **Sub-step 7.2**: Terapkan skor kepercayaan geometric mean ($C_{\text{overall}}$).
* **Sub-step 7.3**: Pasang validator konsistensi 31 intent agar model menolak perintah yang tidak masuk akal.

### FASE 8: EXPORT GRAPH ONNX, DYNAMIC INT8 QUANTIZATION & PARITY CHECK
* **Sub-step 8.1**: Export model ke format ONNX (`models/svara_massive_bilingual.onnx`).
* **Sub-step 8.2**: Terapkan Dynamic INT8 Quantization (`models/svara_massive_bilingual_int8.onnx`).
* **Sub-step 8.3**: Uji paritas numerik PyTorch vs ONNX ($\max \Delta < 10^{-5}$).

### FASE 9: INTEGRASI SISTEM WEB SERVING & LIVE TEST MIKROFON FISIK
* **Sub-step 9.1**: Setting backend FastAPI di port 8001 dan Vite proxy di port 5180.
* **Sub-step 9.2**: Uji otomatis ke-31 intent melalui API.
* **Sub-step 9.3**: Uji live langsung lewat mic laptop di browser pada visualizer 2D dan 3D smart home.
