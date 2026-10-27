# Rencana Komprehensif Eksekusi Ilmiah: Pematangan Deep Learning SLU SVARA (8 Fase Penuh)

Dokumen ini adalah breakdown operasional dari `docs/BLUEPRINT_DEEP_LEARNING_SLU.md`. Dirancang khusus agar eksekusi pematangan model Spoken Language Understanding (SLU) SVARA berjalan **lengkap, ketat, tanpa ada satupun tahapan Deep Learning yang terlewat**.

> **Catatan Kebijakan**:
> 1. File notebook `notebooks/02_indonesian_bilingual_training.ipynb` **TIDAK AKAN DISENTUH/DIUBAH**. Seluruh implementasi interaktif baru akan dibuat di file notebook baru: `notebooks/03_production_slu_deep_learning_pipeline.ipynb`.
> 2. Setiap hasil eksekusi fase, tabel metrik, dan kode baru **wajib langsung di-commit dan di-push ke branch `main` GitHub milik Yoga (`YogUNI/SVARA`)**.

---

## Peta Eksekusi 8 Fase Deep Learning Menyeluruh

```
[FASE 1] Audio Signal Processing & Acoustic Hygiene
   │     (Bandpass Butterworth 80Hz-7.5kHz + Adaptive RMS Normalization + Dual-Threshold VAD)
   ▼
[FASE 2] SpecAugment & Real-World Environmental Augmentations
   │     (Time/Freq Masking + Schroeder Reverb + Hardware Mic Response + Speed/Pitch Perturbation)
   ▼
[FASE 3] Pembuatan Master Notebook Baru: `notebooks/03_production_slu_deep_learning_pipeline.ipynb`
   │     (Notebook interaktif lengkap berstandar riset akademis dari Cell 1 hingga Cell akhir)
   ▼
[FASE 4] Arsitektur Model: Decoupled Multi-Task Heads & Length-Aware Masked Pooling
   │     (Backbone wav2vec2 + Masked Pooling + 3 Slot Heads: Action, Object, Location + Auxiliary Joint 31)
   ▼
[FASE 5] Training & Optimasi Terkalibrasi pada GPU NVIDIA RTX 4060
   │     (Multi-Task Loss + Label Smoothing ε=0.05 + AdamW + Cosine Warmup Schedule + AMP FP16)
   ▼
[FASE 6] Uncertainty Quantification, Temperature Scaling & Out-of-Domain Rejection
   │     (Kalibrasi Logits T-Scaling + Geometric Mean Scoring + 31 Intent Legal Consistency Validator)
   ▼
[FASE 7] Serialisasi Graf ONNX, Dynamic INT8 Quantization & Parity Check
   │     (Ekspor PyTorch -> ONNX + Kuantisasi INT8 360MB->90MB + Verifikasi Numerik Max Diff < 1e-5)
   ▼
[FASE 8] Integrasi Web Serving & Live Interactive Microphone Testing
   │     (FastAPI Port 8000 + Proxy Vite 5180 + Uji Coba Suara Fisik Nyata di 2D/3D Smart Home Visualizer)
```

---

## Rincian Checklist & Acceptance Criteria Per Fase

### FASE 1: Audio Signal Processing & Acoustic Hygiene (Signal Level)
- [ ] Implementasi Butterworth Bandpass Filter Ordo 2 (Cutoff 80 Hz - 7.500 Hz) di `src/svara/data/audio.py`.
- [ ] Implementasi Adaptive RMS Normalizer (Target: -20 dBFS) untuk menjaga konsistensi gain vokal.
- [ ] Implementasi Dual-Threshold Energy VAD (Voice Activity Detection) dengan padding buffer 150ms agar konsonan awal/akhir tidak terpotong.
- [ ] Uji unit test verifikasi sinyal audio (`pytest tests/test_audio.py`).

### FASE 2: SpecAugment & Real-World Environmental Augmentations (Data Level)
- [ ] Implementasi SpecAugment (Frequency Masking parameter $F=27$, Time Masking parameter $T=50$) pada tensor representasi.
- [ ] Simulasi respons mikrofon laptop dan derau kipas ruangan.
- [ ] Speed & Pitch Perturbation ($0.92\times - 1.08\times$) untuk variasi dinamika tempo bicara orang Indonesia.

### FASE 3: Pembuatan Master Notebook Baru (`notebooks/03_production_slu_deep_learning_pipeline.ipynb`)
- [ ] Buat file notebook baru dari nol (tanpa menyentuh `notebooks/02`).
- [ ] Struktur Cell 1: Hardware & CUDA Verification (NVIDIA RTX 4060).
- [ ] Struktur Cell 2: Audio Preprocessing Visualization (Waveform & Spectrogram sebelum vs sesudah filter).
- [ ] Struktur Cell 3: Data Loader & Balanced Bilingual Splits inspection.
- [ ] Struktur Cell 4: Arsitektur Wav2VecSLU & Multi-Head Inspection.
- [ ] Struktur Cell 5: Training Execution Engine dengan live progress bar & loss logging.
- [ ] Struktur Cell 6: Visualisasi Kurva Loss & Exact-Match Accuracy interaktif (Matplotlib).
- [ ] Struktur Cell 7: Evaluasi Ilmiah pada Unseen Indonesian Test Set & English FSC Benchmark.
- [ ] Struktur Cell 8: Export ke ONNX INT8 langsung dari cell notebook.
- [ ] Struktur Cell 9: Interactive Widget / Audio Tester langsung di notebook.

### FASE 4: Arsitektur Model: Decoupled Multi-Task Heads & Length-Aware Masked Pooling (Model Level)
- [ ] Verifikasi pembekuan (*freezing*) 7 lapisan temporal conv feature encoder wav2vec2.
- [ ] Implementasi Length-Aware Masked Mean Pooling yang tahan terhadap variasi durasi audio dan batch padding.
- [ ] Konfigurasi 3 Linear Heads independen (Action 6, Object 14, Location 4) + Auxiliary Joint Head (31).

### FASE 5: Training & Optimasi Terkalibrasi pada GPU RTX 4060 (Optimization Level)
- [ ] Formulasi Multi-Task Cross-Entropy Loss dengan Label Smoothing ($\epsilon = 0.05$).
- [ ] Konfigurasi optimizer AdamW dengan pemisahan parameter decay (weight decay 0.01 untuk bobot linear, 0.0 untuk bias dan layer norm).
- [ ] Cosine Annealing Learning Rate Schedule dengan 10% Warmup Steps.
- [ ] Eksekusi fine-tuning dengan PyTorch Automatic Mixed Precision (AMP FP16) hingga mencapai loss minimal dan checkpoint tersimpan di `reports/runs/production_slu_w2v2/best.ckpt`.

### FASE 6: Uncertainty Quantification & Semantic Consistency Validator (Post-Processing Level)
- [ ] Perhitungan Temperature Scaling ($T$) pada validation set untuk kalibrasi probabilitas softmax.
- [ ] Implementasi Geometric Mean Confidence Scoring untuk mencegah penalti sepihak akibat desis mic.
- [ ] Integrasi validator tata bahasa smart home: menolak otomatis (*abstain/reject*) jika triplet aksi-objek-lokasi tidak valid secara logika.

### FASE 7: Serialisasi Graf ONNX, Dynamic INT8 Quantization & Parity Check (Deployment Level)
- [ ] Export model PyTorch ke ONNX dengan dynamic axes (batch size dan audio length).
- [ ] Dynamic INT8 Quantization tanpa file lock error di lingkungan Windows.
- [ ] Verifikasi paritas matematika: $\max |\hat{\mathbf{y}}_{\text{PyTorch}} - \hat{\mathbf{y}}_{\text{ONNX}}| < 10^{-5}$ pada 10 durasi audio berbeda.

### FASE 8: Integrasi Web Serving & Live Interactive Microphone Testing (System Level)
- [ ] Konfigurasi backend FastAPI resmi pada port `8000` (port 8001 dialokasikan untuk project KARMA).
- [ ] Konfigurasi proxy Vite di `web/vite.config.ts` mengarah ke port `8000`.
- [ ] Verifikasi live mic di 2D/3D Smart Home Visualizer: memastikan lampu, suhu, dan musik merespon instan (<800 ms) sesuai perintah suara pengguna.
- [ ] Final commit dan push semua perubahan ke GitHub remote `main`.
