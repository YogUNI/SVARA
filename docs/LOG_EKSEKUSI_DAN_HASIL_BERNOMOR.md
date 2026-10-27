# LOGBOOK RESMI EKSEKUSI & HASIL DEEP LEARNING SVARA (BERNOMOR)

Dokumen ini adalah **catatan laboratorium resmi (*Scientific Experimentation & Execution Logbook*)** proyek Spoken Language Understanding (SLU) **SVARA (Smart Voice Assistant for Residential Automation)**.
Setiap sub-tahap yang dieksekusi dicatat secara lengkap mulai dari formulasi matematika, konfigurasi parameter numerik, hasil evaluasi/unit test riil, artefak file yang dihasilkan, hingga nomor *commit* GitHub.

> **Tujuan untuk Tim (Yoga & Haikal)**:
> Dokumen ini disusun agar saat penulisan **Laporan Tugas Akhir / Skripsi (khususnya Bab 4: Perancangan & Implementasi, serta Bab 5: Hasil dan Pembahasan)**, seluruh data empiris, tabel parameter, dan grafik luaran dapat disalin langsung tanpa ada data teknis yang terlewat.

---

## DAFTAR RINGKASAN PROGRESS SUB-STEP

| Kode Sub-step | Nama Tahapan | Status | Commit Git | File Utama | Rujukan Bab Laporan |
| :---: | :--- | :---: | :---: | :--- | :---: |
| **1.1** | Butterworth Bandpass Filter (80 Hz - 7.5 kHz) | **SELESAI** | `e29ae88` | `src/svara/data/audio.py` | Bab 4 §4.2 |
| **1.2** | Adaptive RMS Energy Normalization (-20 dBFS) | **SELESAI** | `0f55ec2` | `src/svara/data/audio.py` | Bab 4 §4.2 |
| **1.3** | Dual-Threshold Energy VAD & Silence Trimming | **SELESAI** | `258a15b` | `src/svara/data/audio.py` | Bab 4 §4.2 |
| **1.4** | Unit Test Paritas Sinyal Preprocessing | *MENUNGGU* | - | `tests/test_audio.py` | Bab 4 §4.2, Bab 5 |
| **2.1 - 2.3** | Rekayasa Korpus NLP Bahasa Indonesia (4.000+ Frasa) | *MENUNGGU* | - | `scripts/generate_...py` | Bab 3 §3.2, Bab 4 §4.1 |
| **3.1 - 3.3** | Sintesis Suara Multi-Speaker Massal (TTS) | *MENUNGGU* | - | `scripts/synthesize_...py` | Bab 4 §4.1 |
| **4.1 - 4.4** | Environmental Acoustic Augmentation (24k Audio) | *MENUNGGU* | - | `scripts/augment_...py` | Bab 4 §4.1 |
| **5.1 - 5.3** | Balanced Bilingual Dataset Splitting (Zero-Leakage) | *MENUNGGU* | - | `scripts/make_splits_...py`| Bab 4 §4.1 |
| **6.1 - 6.3** | Pembuatan Master Notebook `03_production_pipeline` | *MENUNGGU* | - | `notebooks/03_...ipynb` | Bab 4 §4.3 |
| **7.1 - 7.3** | Training & Optimasi Terkalibrasi GPU RTX 4060 | *MENUNGGU* | - | `reports/runs/...` | Bab 5 §5.1, §5.2 |
| **8.1 - 8.3** | Uncertainty Quantification, T-Scaling & Legal Gate | *MENUNGGU* | - | `src/svara/serve/...` | Bab 4 §4.4, Bab 5 |
| **9.1 - 9.3** | Export Graph ONNX, Dynamic INT8 & Parity Check | *MENUNGGU* | - | `models/svara_...onnx` | Bab 5 §5.3 |
| **10.1 - 10.3** | Integrasi Web Serving & Live Test Mikrofon Fisik | *MENUNGGU* | - | `svara.serve.app` | Bab 5 §5.4 |

---

## DETIL CATATAN EKSPERIMEN & OUTPUT HASIL PER SUB-STEP

---

### FASE 1: AUDIT & STANDARISASI PEMROSESAN SINYAL AUDIO (SIGNAL HYGIENE)

#### Sub-step 1.1: Implementasi Butterworth Bandpass Filter (80 Hz - 7.500 Hz)
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi & Verifikasi Kode)
* **Latar Belakang & Masalah**:
  Mikrofon internal laptop rentan merekam getaran fisik meja/keyboard (*mechanical body rumble*) di frekuensi infrasonik ($< 80\text{ Hz}$) serta desis kebocoran listrik komponen elektronik (*electrical hiss/coil whine*) di frekuensi tinggi mendekati frekuensi Nyquist ($> 7.500\text{ Hz}$). Noise ini mengotori fitur spektrogram akustik CNN pada lapisan bawah `wav2vec2`.
* **Formulasi Matematika & Algoritma**:
  - Menggunakan filter lolos-pita (*bandpass*) IIR Butterworth Ordo 2 dengan normalisasi digital frekuensi ternormalisasi terhadap frekuensi Nyquist ($f_{\text{Nyq}} = 8.000\text{ Hz}$ pada sampling rate $16.000\text{ Hz}$):
    $$\omega_{\text{low}} = \frac{80}{8000} = 0.010, \quad \omega_{\text{high}} = \frac{7500}{8000} = 0.9375$$
  - Implementasi *zero-phase digital filtering* menggunakan fungsi `scipy.signal.filtfilt` agar tidak ada pergeseran fasa sinyal (*zero phase distortion*):
    $$y[n] = \text{filtfilt}(b, a, x[n])$$
* **File yang Dimodifikasi**:
  - `src/svara/data/audio.py` (Fungsi `apply_butterworth_bandpass` dan integrasi ke `preprocess_audio`).
  - `tests/test_audio.py` (Fungsi pengujian `test_butterworth_bandpass_attenuation`).
* **Hasil Pengujian Unit Test Riil**:
  - Nada vokal manusia ($500\text{ Hz}$): Nilai korelasi retensi pasca-filter $= 0.9412 > 0.90$ (Lolos/Preserved).
  - Derau getaran mekanis meja ($30\text{ Hz}$): Nilai korelasi pasca-filter $= 0.0821 < 0.20$ (Teredam drastis/Attenuated).
  - Status pytest: `5 passed in 8.02s`.
* **Nomor Commit Git**:
  - `commit e29ae88` (*feat(audio): implement 80Hz-7.5kHz Butterworth bandpass filter in preprocess_audio (Sub-step 1.1)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

#### Sub-step 1.2: Implementasi Adaptive RMS Energy Normalization (-20 dBFS)
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi & Verifikasi Kode)
* **Latar Belakang & Masalah**:
  Dalam lingkungan nyata, volume ucapan pengguna sangat bervariasi: bisikan berjarak jauh menghasilkan energi $-35\text{ dBFS}$ sampai $-40\text{ dBFS}$, sementara ucapan kencang di depan laptop mencapai $-6\text{ dBFS}$. Jika tidak distandarkan energinya, threshold VAD akan salah potong dan representasi transformer akan bias pada suara keras saja.
* **Formulasi Matematika & Algoritma**:
  - Nilai RMS dihitung dari seluruh sampel $N$:
    $$\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$
  - Target RMS pada $-20\text{ dBFS}$ dihitung dalam skala linier:
    $$\text{target\_rms} = 10^{\frac{-20.0}{20}} = 10^{-1} = 0.100$$
  - Faktor pengali linier (*gain*):
    $$g_{\text{raw}} = \frac{\text{target\_rms}}{\text{RMS}(x) + \epsilon} = \frac{0.100}{\text{RMS}(x) + 10^{-7}}$$
  - Pembatasan amplifikasi maksimum (*gain clamp*) $+30\text{ dB}$ ($g_{\max} = 10^{30/20} = 31.6228$) dan *threshold noise floor* ($10^{-4}$) agar desis hening tidak diamplifikasi berlebihan:
    $$g = \min(g_{\text{raw}}, 31.6228) \quad \text{jika } \text{RMS}(x) \ge 10^{-4}$$
  - Pengamanan puncak gelombang (*peak limiting guard*) mencegah *digital clipping* $> 1.0$:
    $$x_{\text{norm}} = \frac{x \times g}{\max(1.0, \max(|x \times g|))}$$
* **File yang Dimodifikasi**:
  - `src/svara/data/audio.py` (Konstanta `DEFAULT_TARGET_RMS_DBFS`, `DEFAULT_MAX_RMS_GAIN_DB`, fungsi `adaptive_rms_normalize`, dan integrasi parameter `apply_rms_norm=True` di `preprocess_audio`).
  - `tests/test_audio.py` (Test `test_adaptive_rms_normalize_quiet_and_loud` dan `test_adaptive_rms_normalize_silence_and_gain_clamp`).
* **Hasil Pengujian Numerik Riil**:
  - Sinyal berbisik (RMS awal $= 0.0200$, $-34.0\text{ dBFS}$) $\rightarrow$ RMS akhir $= 0.1000$ ($-20.00\text{ dBFS}$, tepat 100%).
  - Sinyal lantang (RMS awal $= 0.5000$, $-6.0\text{ dBFS}$) $\rightarrow$ RMS akhir $= 0.1000$ ($-20.00\text{ dBFS}$, tepat 100%).
  - Hening total ($x = 0$): Output tetap bernilai 0.0 mutlak, bebas NaN dan bebas Inf.
  - Sinyal sangat redup yang butuh $+40\text{ dB}$ gain: Dibatasi tepat pada batas aman penguatan $+30.00\text{ dB}$ ($g = 31.6228$).
  - Status pytest: `7 passed in 36.88s`.
* **Nomor Commit Git**:
  - `commit 0f55ec2` (*feat(audio): implement adaptive RMS energy normalization to -20 dBFS (Sub-step 1.2)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

#### Sub-step 1.3: Implementasi Dual-Threshold Energy VAD & Silence Trimming (Buffer 150ms)
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi & Verifikasi Kode)
* **Latar Belakang & Masalah**:
  Rekaman mikrofon mentah (baik dari dataset maupun web UI browser) sering menyisakan hening 0.5s - 1.5s di awal sebelum pengguna berbicara dan di akhir setelah perintah selesai. Hening panjang ini membuang siklus komputasi inferensi (latensi lambat), mengaburkan representasi *self-attention* transformer, dan menurunkan probabilitas softmax pada pooling akhir. Namun, algoritma pemangkas hening konvensional dengan *single hard threshold* sangat berbahaya karena memotong letupan konsonan tak bersuara (*unvoiced plosives*) di awal kata seperti *"k"* (*"kamar"*, *"kunci"*), *"t"* (*"tolong"*, *"tutup"*), dan *"p"* (*"puter"*), atau ekor kata (*"hidup"*, *"musik"*).
* **Formulasi Matematika & Algoritma**:
  - Menggunakan komputasi energi per-frame bervektor (*vectorized short-term energy*):
    - Frame duration $K = 20\text{ ms} = 320\text{ samples}$ pada $16\text{ kHz}$.
    - Hop duration $H = 10\text{ ms} = 160\text{ samples}$.
    - Energi frame ke-$m$:
      $$\text{RMS}_m = \sqrt{\frac{1}{K}\sum_{k=0}^{K-1} x[m \cdot H + k]^2}, \quad E_m = 20 \log_{10}(\text{RMS}_m + 10^{-12}) \text{ dBFS}$$
  - **Mekanisme Dual-Threshold (Histeresis)**:
    - $\theta_{\text{start}} = -32.0\text{ dBFS}$: Threshold onset ucapan aktif (mencegah desis latar memicu false positive).
    - $\theta_{\text{stop}} = -42.0\text{ dBFS}$: Threshold pelacakan fonem lembut (menjaga fonem frikatif/plosif bernada rendah tidak terpotong sebelum/sesudah puncak vokal).
  - **Acoustic Safety Buffer ($150\text{ ms}$)**:
    - Margin perlindungan akustik: $150\text{ ms} = 2.400\text{ sampel}$ pada $16\text{ kHz}$.
    - Batas awal pemotongan: $S_{\text{start}} = \max(0, m_{\text{first}} \cdot H - 2400)$
    - Batas akhir pemotongan: $S_{\text{end}} = \min(N, m_{\text{last}} \cdot H + K + 2400)$
  - **Anti-Click / Transient Noise Filter**:
    - Jika durasi aktif unpadded ucapan $< 200\text{ ms}$ ($3.200\text{ sampel}$), sinyal dianggap derau klik/ketukan mouse dan sistem secara aman mempertahankan sinyal asli tanpa pemotongan parsial.
* **File yang Dimodifikasi**:
  - `src/svara/data/audio.py` (Konstanta `DEFAULT_VAD_START_DBFS`, `DEFAULT_VAD_STOP_DBFS`, `DEFAULT_VAD_PAD_MS`, `DEFAULT_VAD_MIN_SPEECH_MS`, fungsi `trim_silence_vad`, dan integrasi parameter `apply_vad_trim=True` di `preprocess_audio`).
  - `tests/test_audio.py` (Test `test_trim_silence_vad_speech_padding`, `test_trim_silence_vad_silence_and_click_fallback`, `test_trim_silence_vad_continuous`).
* **Hasil Pengujian Unit Test Riil**:
  - **Uji Pemangkasan Hening 3.0s (1.0s hening + 1.0s nada + 1.0s hening)**:
    - Titik potong onset: Terpangkas tepat di sampel $13.440$ ($0.840\text{s}$, menyisakan buffer aman $160\text{ ms}$ sebelum nada).
    - Titik potong offset: Terpangkas tepat di sampel $34.560$ ($2.160\text{s}$, menyisakan buffer aman $160\text{ ms}$ setelah nada).
    - Total durasi terpangkas efisien dari $3.00\text{s}$ menjadi $1.32\text{s}$ (menghemat komputasi inferensi hingga $> 55\%$).
  - **Uji Hening Total & Desis**: Sinyal nol mutlak tetap aman dan utuh tanpa runtime error atau array kosong.
  - **Uji Klik Transien (10 ms click)**: Ditolak dari pemotongan dan kembali ke durasi aman.
  - **Uji Sinyal Kontinu**: 100% sampel dipertahankan tanpa satu pun sampel terbuang.
  - Status pytest: `10 passed in 33.93s`.
* **Nomor Commit Git**:
  - `commit 258a15b` (*feat(audio): implement dual-threshold energy VAD & silence trimming with 150ms buffer (Sub-step 1.3)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---
*(Logbook ini akan diperbarui secara otomatis setiap kali sub-step berikutnya selesai dieksekusi)*
