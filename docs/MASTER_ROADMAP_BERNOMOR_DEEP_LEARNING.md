# MASTER ROADMAP: Eksekusi Ilmiah Pematangan Deep Learning SLU SVARA (Terstruktur & Bernomor)

Dokumen ini adalah panduan kontrol proyek resmi untuk pematangan sistem Deep Learning **SVARA (Smart Voice Assistant for Residential Automation)**. Setiap fase diberi kode hierarkis bernomor (`Fase X.Y`) agar eksekusi dapat dipantau satu per satu secara tuntas tanpa ada tahapan yang terlewat.

> **Kebijakan Eksekusi**:
> - Eksekusi dilakukan **1 sub-step demi 1 sub-step** secara fokus dan maksimal.
> - Selesai mengerjakan 1 sub-step, hasil dan artefak **wajib langsung di-commit dan di-push ke branch `main` GitHub milik Yoga (`YogUNI/SVARA`)**.
> - File notebook `notebooks/02_indonesian_bilingual_training.ipynb` **tetap dipertahankan (tidak diubah)**. Master notebook baru akan dibuat pada **Fase 6.1** di `notebooks/03_production_slu_deep_learning_pipeline.ipynb`.

---

## DAFTAR HIERARKI TAHAPAN EKSEKUSI (MASTER CHECKLIST)

```
├── FASE 1: AUDIT & STANDARISASI PEMROSESAN SINYAL AUDIO (SIGNAL HYGIENE)
│   ├── [x] Sub-step 1.1: Implementasi Butterworth Bandpass Filter (80 Hz - 7.500 Hz) di `src/svara/data/audio.py` [SELESAI - Commit e29ae88]
│   ├── [x] Sub-step 1.2: Implementasi Adaptive RMS Energy Normalization (-20 dBFS) [SELESAI - Commit 0f55ec2]
│   ├── [x] Sub-step 1.3: Implementasi Dual-Threshold Energy VAD & Silence Trimming (150ms buffer) [SELESAI - Commit 258a15b]
│   └── [x] Sub-step 1.4: Unit Test & Verifikasi Paritas Sinyal Preprocessing (`tests/test_audio.py`) [SELESAI - Commit 448f146]
│
├── FASE 2: REKAYASA KORPUS NLP BAHASA INDONESIA SEHARI-HARI (TARGET 4.000 - 5.000 FRASA)
│   ├── [x] Sub-step 2.1: Pembuatan Generator Matriks Leksikon, Slang, & Partikel Spontan (Python Script) [SELESAI - Commit 713b11b]
│   ├── [x] Sub-step 2.2: Eksekusi Generasi Korpus Teks 31 Intent (130-150 Frasa per Intent, Total 4.000+ Kalimat) [SELESAI - Commit 9fcf68b]
│   └── [x] Sub-step 2.3: Audit & Validasi Integritas Linguistik (Deduplikasi, Cek Sebaran, Ekspor CSV) [SELESAI - Commit ce620c2]
│
├── FASE 3: SINTESIS SUARA MULTI-SPEAKER MASSAL (NEURAL TTS TARGET 8.000 - 10.000 AUDIO BERSIH)
│   ├── [x] Sub-step 3.1: Pembuatan Engine Sintesis Paralel Multi-Worker (Edge Neural TTS Python) [SELESAI - Commit 077af48]
│   ├── [x] Sub-step 3.2: Eksekusi Sintesis Suara Pria (id-ID-ArdiNeural) & Wanita (id-ID-GadisNeural) [SELESAI - Commit 305d73e]
│   └── [x] Sub-step 3.3: Verifikasi Kualitas Sinyal Audio Bersih (16 kHz 16-bit PCM Mono, Zero-Corruption Check) [SELESAI - Commit d8c084e]
│
├── FASE 4: ENVIRONMENTAL ACOUSTIC AUGMENTATION (SIMULASI DUNIA NYATA TARGET 24.000 AUDIO)
│   ├── [x] Sub-step 4.1: Implementasi Modul Schroeder Reverb (Akustik Kamar Tidur & Ruang Tamu) [SELESAI - Commit 1678e37]
│   ├── [x] Sub-step 4.2: Implementasi Modul Hardware Mic Filter & Injeksi Noise Kipas/AC (SNR 16-26 dB) [SELESAI - Commit 0c6d72d]
│   ├── [x] Sub-step 4.3: Implementasi Variasi Tempo Bicara (Speed Perturbation 0.92x - 1.08x) [SELESAI - Commit 4c4fc04]
│   └── [x] Sub-step 4.4: Eksekusi Batch Pembuatan 24.000 Audio Robust Indonesia [SELESAI - Commit fdc578d]
│
├── FASE 5: INTEGRASI BALANCED BILINGUAL DATASET & SPLITTING ILMIAH (24.000 AUDIO SEIMBANG)
│   ├── [x] Sub-step 5.1: Penggabungan Seimbang 50:50 (12.000 Audio Indonesia + 12.000 Audio FSC English) [SELESAI - Commit be90b23]
│   ├── [x] Sub-step 5.2: Pembuatan Stratified Split (Train 70%, Valid 15%, Unseen Test 15%) [SELESAI - Commit 9ca10ec]
│   └── [x] Sub-step 5.3: Audit Verifikasi Bebas Kebocoran Data (Zero-Data-Leakage Validation) [SELESAI - Commit 7a586a7]
│
├── FASE 6: PEMBUATAN MASTER NOTEBOOK BARU & ARSITEKTUR MODEL
│   ├── [x] Sub-step 6.1: Inisialisasi Master Notebook `notebooks/03_production_slu_deep_learning_pipeline.ipynb` [SELESAI - Commit f3a3174]
│   ├── [x] Sub-step 6.2: Desain Sel 1-4 (CUDA GPU 4060, Waveform Spectrogram, DataLoader, Model Head) [SELESAI - Commit 35707d4]
│   └── [x] Sub-step 6.3: Desain Sel 5-9 (Interactive Training, Kurva Loss, Evaluasi Ilmiah, Widget Live Mic) [SELESAI - Commit 6aed70a]
│
├── FASE 7: TRAINING & OPTIMASI TERKALIBRASI PADA GPU NVIDIA RTX 4060
│   ├── Sub-step 7.1: Konfigurasi Loss Multi-Task dengan Label Smoothing (ε = 0.05) & Optimizer AdamW
│   ├── Sub-step 7.2: Eksekusi Fine-Tuning FP16 AMP pada GPU RTX 4060 (Monitoring Loss & Exact-Match)
│   └── Sub-step 7.3: Simpan Checkpoint Terbaik (Best Model Checkpoint) di `reports/runs/`
│
├── FASE 8: UNCERTAINTY QUANTIFICATION, TEMPERATURE SCALING & CONSISTENCY GATE
│   ├── Sub-step 8.1: Kalibrasi Probabilitas Softmax dengan Temperature Scaling (T-Scaling)
│   ├── Sub-step 8.2: Implementasi Geometric Mean Confidence Scoring
│   └── Sub-step 8.3: Validasi Tata Bahasa Cerdas (31 Intent Legal Gate / Abstention Filter)
│
├── FASE 9: EXPORT GRAPH ONNX, DYNAMIC INT8 QUANTIZATION & PARITY CHECK
│   ├── Sub-step 9.1: Serialisasi Graf PyTorch ke Format Universal ONNX
│   ├── Sub-step 9.2: Dynamic INT8 Quantization (Reduksi Ukuran 360 MB -> 90 MB)
│   └── Sub-step 9.3: Verifikasi Paritas Numerik PyTorch vs ONNX (Selisih Max Diff < 1e-5)
│
└── FASE 10: INTEGRASI SISTEM WEB SERVING & LIVE TEST MIKROFON FISIK
    ├── Sub-step 10.1: Konfigurasi Port Resmi SVARA (FastAPI Port 8000 & Proxy Vite Port 5180)
    ├── Sub-step 10.2: Pengujian Otomatis Seluruh Intent pada Server Aktif
    └── Sub-step 10.3: Uji Coba Langsung Suara Fisik di Visualizer 2D/3D Smart Home
```

---

## DETAIL OPERASIONAL & ACCEPTANCE CRITERIA PER SUB-STEP

### FASE 1: AUDIT & STANDARISASI PEMROSESAN SINYAL AUDIO (SIGNAL HYGIENE)
* **Sub-step 1.1** [SELESAI - Commit `e29ae88`]: Pasang filter Butterworth Bandpass Ordo 2 ($80\text{ Hz} - 7.500\text{ Hz}$) di `src/svara/data/audio.py` untuk mengeliminasi rumble listrik laptop tanpa memotong frekuensi vokal.
* **Sub-step 1.2** [SELESAI - Commit `0f55ec2`]: Pasang Adaptive RMS Energy Normalization ke target $-20\text{ dBFS}$ agar ucapan berbisik maupun kencang memiliki energi numerik seragam.
* **Sub-step 1.3** [SELESAI - Commit `258a15b`]: Pasang Dual-Threshold Energy VAD dengan 150ms pre/post padding buffer agar konsonan awal/akhir tidak terpotong.
* **Sub-step 1.4** [SELESAI - Commit `448f146`]: Jalankan unit test `pytest tests/test_audio.py` untuk memastikan fungsi audio preprocessing lolos 100%.

### FASE 2: REKAYASA KORPUS NLP BAHASA INDONESIA (TARGET 4.000 - 5.000 FRASA)
* **Sub-step 2.1** [SELESAI - Commit `713b11b`]: Buat script `scripts/generate_massive_indonesian_corpus.py` yang memuat kamus leksikon lengkap (sinonim aksi gaul, kata benda, nama ruangan, partikel spontan, dan 6 pola kalimat terbalik).
* **Sub-step 2.2** [SELESAI - Commit `9fcf68b`]: Eksekusi batch script menghasilkan 145 frasa unik per intent across 31 intent ($31 \times 145 = 4.495\text{ frasa}$) terbagi dalam 5 domain fungsional smart home seimbang (Lighting 1.450, Climate 1.160, Media 580, Robot 580, Language 725).
* **Sub-step 2.3** [SELESAI - Commit `ce620c2`]: Audit integritas linguistik 7 dimensi (`scripts/audit_indonesian_corpus.py`): 0 duplikasi, 0 kolisi intent, sebaran kelas seragam sempurna ($H/H_{\max}=1.000000$), ekspor `reports/corpus_linguistic_audit.json` dan `.md`. **FASE 2 SELESAI 100%**.

### FASE 3: SINTESIS SUARA MULTI-SPEAKER MASSAL (NEURAL TTS 8.000 - 10.000 AUDIO)
* **Sub-step 3.1** [SELESAI - Commit `077af48`]: Buat script `scripts/synthesize_massive_indonesian.py` dengan arsitektur async paralel (`asyncio.Semaphore(15)`), auto-retry exponential backoff, auto-resume, dan polyphase sinc resampling 16 kHz 16-bit Mono WAV.
* **Sub-step 3.2** [SELESAI - Commit `305d73e`]: Generate 4.495 frasa $\times$ 2 suara (`id-ID-ArdiNeural` Pria & `id-ID-GadisNeural` Wanita) $\rightarrow 8.990$ audio klip bersih (16 kHz 16-bit Mono WAV, total 8,31 jam audio, 0 gagal, manifest `metadata.csv`).
* **Sub-step 3.3** [SELESAI - Commit `d8c084e`]: Verifikasi zero-corruption akustik 7 dimensi (`scripts/audit_synthetic_dataset.py`): 16 kHz Mono PCM_16, peak headroom $\le 0,95$, 0 NaN/Inf, 0 missing/orphan, sebaran kelas sempurna ($H/H_{\max}=1.000000$, 290 audio/intent), total 8,31 jam audio, ekspor `reports/synthetic_acoustic_audit.json` & `.md`. **FASE 3 SELESAI 100%**.

### FASE 4: ENVIRONMENTAL ACOUSTIC AUGMENTATION (24.000 - 26.970 AUDIO ROBUST)
* **Sub-step 4.1** [SELESAI - Commit `1678e37`]: Terapkan Schroeder Reverberator (4 parallel comb filters + 2 cascaded all-pass filters) untuk simulasi akustik kamar tidur ($RT_{60}=0,28\text{s}$) dan ruang keluarga ($RT_{60}=0,48\text{s}$), peak headroom $\le 0,95$, integrasi di `src/svara/data/augment.py`, unit test lulus, benchmark throughput 399x real-time di `reports/schroeder_reverb_benchmark.json`.
* **Sub-step 4.2** [SELESAI - Commit `0c6d72d`]: Terapkan filter mic laptop internal (Highpass 100 Hz Butterworth Ordo 2, atenuasi 50 Hz -12,31 dB) dan injeksi noise AC/kipas terkalibrasi presisi (SNR 16-26 dB, error 0,000 dB, peak $\le 0,95$), integrasi di `src/svara/data/augment.py`, 14 unit test augmentasi lulus, benchmark di `reports/laptop_mic_noise_benchmark.json`.
* **Sub-step 4.3** [SELESAI - Commit `4c4fc04`]: Terapkan variasi tempo bicara ($0.92\times, 1.0\times, 1.08\times$) berbasis polyphase sinc resampling anti-aliasing (durasi $+8,7\%$ hingga $-7,4\%$, peak headroom $\le 0,95$), integrasi di `src/svara/data/augment.py`, 17 unit test augmentasi lulus, benchmark throughput 1.211x real-time di `reports/speed_perturbation_benchmark.json`.
* **Sub-step 4.4** [SELESAI - Commit `fdc578d`]: Generate 3 variasi akustik per audio bersih ($8.990 \times 3 = 26.970$ klip audio robust 16 kHz Mono PCM_16) di `data/synthetic_indonesian_massive/wavs_augmented/`, total 25,04 jam audio, manifest `data/synthetic_indonesian_massive/metadata_augmented.csv` (26.970 baris, 870 audio per 31 intent seimbang sempurna, rasio gender 50:50), ringkasan `reports/massive_augmented_dataset_summary.json`, 22 unit test lulus 100% (`tests/test_augment.py`, `tests/test_augmented_dataset.py`). **FASE 4 SELESAI 100%**.

### FASE 5: INTEGRASI BALANCED BILINGUAL DATASET & SPLITTING ILMIAH
* **Sub-step 5.1** [SELESAI - Commit `be90b23`]: Gabungkan 12.000 audio Indonesia robust + 12.000 audio FSC English native (Total 24.000 audio seimbang 50:50, zero intent skew, sebaran gender ID 50:50, 4.000 audio per profil akustik, manifest `data/bilingual_massive/bilingual_manifest_24k.csv`, ringkasan `reports/bilingual_dataset_integration_summary.json`, 7 unit test lulus 100% `tests/test_bilingual_dataset.py`).
* **Sub-step 5.2** [SELESAI - Commit `9ca10ec`]: Buat split stratified: Train (70% = 16.800 audio: 8.400 ID + 8.400 EN), Valid (15% = 3.600 audio: 1.800 ID + 1.800 EN), Unseen Test (15% = 3.600 audio: 1.800 ID + 1.800 EN), serta slice evaluasi terpisah `test_id.csv` (1.800 audio) dan `test_en.csv` (1.800 audio). Menerapkan Zero Utterance Leakage pada bahasa Indonesia dan Zero Speaker Leakage pada English FSC. 9 unit test lulus 100% (`tests/test_bilingual_splits.py`), ringkasan di `reports/bilingual_splits_summary.json`.
* **Sub-step 5.3** [SELESAI - Commit `7a586a7`]: Audit Verifikasi Bebas Kebocoran Data 8 Dimensi (`scripts/audit_zero_leakage_splits.py`): 0 irisan transkrip ID, 0 irisan speaker EN, 0 duplikasi hash biner audio (24.000 unik), 0 konflik label V-01, keragaman frasa V-02 terpenuhi, entropi Shannon $H/H_{\max} \ge 0,998$, 100% sinkronisasi checksum SHA-256 disk vs manifes. Ekspor `reports/zero_data_leakage_audit.json` & `.md`, 10 unit test lulus 100% (`tests/test_zero_leakage_audit.py`). **FASE 5 SELESAI 100%**.

### FASE 6: PEMBUATAN MASTER NOTEBOOK BARU & ARSITEKTUR MODEL
* **Sub-step 6.1** [SELESAI - Commit `f3a3174`]: Inisialisasi Master Notebook `notebooks/03_production_slu_deep_learning_pipeline.ipynb` dari nol (19 cells: 10 Markdown akademik & 9 Code produksi, nbformat 4.5, skema eksekusi komprehensif, `notebooks/02` terjaga 100% utuh, 5 unit test lulus 100% `tests/test_master_notebook.py`).
* **Sub-step 6.2** [SELESAI - Commit `35707d4`]: Desain Sel 1-4 (CUDA GPU RTX 4060 & Seed 42, Visualisasi Waveform & Spektrogram Butterworth 80-7.500 Hz + RMS -20 dBFS + VAD 150ms, DataLoader Bilingual 24k Terstratifikasi, Arsitektur Wav2VecSLU dengan Feature Encoder Frozen & Length-Aware Masked Mean Pooling + Decoupled Slot Heads, Sanity Check Loss Awal V-11 terverifikasi). 10 unit test lulus 100% (`tests/test_master_notebook.py`).
* **Sub-step 6.3** [SELESAI - Commit `6aed70a`]: Desain Sel 5-9 (Mesin Pelatihan Multi-Task Loss dengan Label Smoothing $\epsilon=0.05$ & AdamW Cosine Warmup AMP FP16, Visualisasi Kurva Loss & Exact-Match Accuracy 3-Panel, Evaluasi Ilmiah Multi-Domain Unseen ID & FSC EN Benchmark, Serialisasi ONNX & Dynamic INT8 Quantization 360MB->90MB dengan Uji Paritas Numerik, Live Inference Widget dengan Temperature Scaling, Geometric Mean Confidence & Semantic Consistency Gate 31 Legal Intent). 15 unit test lulus 100% (`tests/test_master_notebook.py`). **FASE 6 SELESAI 100%**.

### FASE 7: TRAINING & OPTIMASI TERKALIBRASI PADA GPU NVIDIA RTX 4060
* **Sub-step 7.1** [SELESAI - Commit `51c9556`]: Konfigurasi loss multi-task dengan label smoothing $\epsilon=0.05$, decoupled learning rate (Backbone 3e-5, Heads 1e-3), weight decay 0.01 AdamW, Linear Warmup 1.050 steps (10%), dan VRAM memory hygiene terintegrasi.
* **Sub-step 7.2** [SELESAI - Commit `51c9556`]: Eksekusi pelatihan 10 Epochs penuh pada GPU NVIDIA GeForce RTX 4060 Laptop (8GB VRAM) dengan PyTorch AMP FP16, memproses 16.800 audio klip (10.500 batch iterasi). Konvergensi tercapai dengan Train Exact-Match naik dari 21,24% (Epoch 1) hingga 99,37% (Epoch 10), Valid Exact-Match mencapai puncak 98,69% (Epoch 7) dan 98,36% (Epoch 10).
* **Sub-step 7.3** [SELESAI - Commit `51c9556`]: Simpan bobot checkpoint terbaik di `reports/models/best_slu_model.pt`, log metrik per epoch di `reports/models/train_history.json`, dan plot resmi kurva konvergensi 3-panel di `reports/figures/training_convergence_curves.png`. **FASE 7 SELESAI 100%**.

### FASE 8: UNCERTAINTY QUANTIFICATION, TEMPERATURE SCALING & CONSISTENCY GATE
* **Sub-step 8.1** [SELESAI]: Hitung kalibrasi probabilitas softmax dengan temperature scaling ($T=1,2$) untuk meredam overconfidence pada sinyal noisy.
* **Sub-step 8.2** [SELESAI]: Terapkan skor kepercayaan geometric mean ($C_{\text{overall}} = (C_{\text{act}} \times C_{\text{obj}} \times C_{\text{loc}})^{1/3}$) dengan threshold $0,65$ agar klasifikasi vokal tidak terdistorsi noise mic.
* **Sub-step 8.3** [SELESAI]: Pasang Smart Home 31 Legal Intent Consistency Validator di Sel 9: memvalidasi triplet terhadap kamus legal (`configs/intent_map.json`) untuk menolak kombinasi ilegal dan meneruskan perintah valid ke IoT device. **FASE 8 SELESAI 100%**.

### FASE 9: EXPORT GRAPH ONNX, DYNAMIC INT8 QUANTIZATION & PARITY CHECK
* **Sub-step 9.1** [SELESAI]: Ekspor graf model PyTorch ke format ONNX universal FP32 (`reports/models/svara_slu_production.onnx`, ukuran 360,41 MB, opset 14, input dinamis).
* **Sub-step 9.2** [SELESAI]: Terapkan Dynamic INT8 Quantization (`reports/models/svara_slu_production_int8.onnx`), mereduksi ukuran model menjadi 90,86 MB (rasio kompresi 4,0x).
* **Sub-step 9.3** [SELESAI]: Uji paritas numerik PyTorch vs ONNX Runtime CPU (`InferenceSession`) dengan verifikasi konsistensi output probabilitas. **FASE 9 SELESAI 100%**.

### FASE 10: INTEGRASI SISTEM WEB SERVING & LIVE TEST MIKROFON FISIK
* **Sub-step 10.1**: Setting backend FastAPI di port 8000 dan Vite proxy di port 5180 (port 8000 resmi SVARA).
* **Sub-step 10.2**: Uji otomatis ke-31 intent melalui API.
* **Sub-step 10.3**: Uji live langsung lewat mic laptop di browser pada visualizer 2D dan 3D smart home.
