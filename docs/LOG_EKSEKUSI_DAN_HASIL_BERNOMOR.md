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
| **1.4** | Unit Test Paritas Sinyal Preprocessing | **SELESAI** | `448f146` | `tests/test_audio.py` | Bab 4 §4.2, Bab 5 |
| **2.1** | Generator Matriks Leksikon & Permutasi Sintaksis | **SELESAI** | `713b11b` | `scripts/generate_massive_indonesian_corpus.py` | Bab 3 §3.2, Bab 4 §4.1 |
| **2.2** | Eksekusi Generasi Korpus Teks 31 Intent (4.000+ Frasa) | *MENUNGGU* | - | `data/corpus/indonesian_smart_home_corpus_4k.csv` | Bab 4 §4.1 |
| **2.3** | Audit & Validasi Integritas Linguistik Korpus | *MENUNGGU* | - | `tests/test_corpus.py` | Bab 4 §4.1, Bab 5 |
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

#### Sub-step 1.4: Unit Test & Verifikasi Paritas Sinyal Preprocessing
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi & Verifikasi Kode)
* **Latar Belakang & Masalah**:
  Salah satu jebakan terbesar dalam Machine Learning audio adalah ketidakcocokan (*skew / distribution drift*) antara input data saat training dengan input data saat inferensi di production. Sesuai **Rule 7 AGENTS.md**, model SLU harus menerima perlakuan sinyal yang 100% identik baik ketika data dibaca dari file WAV di harddisk (oleh PyTorch DataLoader), dikirim berupa *in-memory bytes buffer* via HTTP multipart/form-data oleh browser microphone di FastAPI backend, maupun diolah sebagai NumPy array atau PyTorch Tensor.
* **Cakupan Pengujian Paritas Komprehensif**:
  1. **Multi-Format Input Parity**:
     - Membandingkan hasil pra-pemrosesan sinyal dari 4 format input: `filepath` string, `bytes` buffer, `np.ndarray`, dan `torch.Tensor`.
     - *File vs Bytes*: Selisih numerik $\Delta = 0.00\text{e}+00$ (identik bit-for-bit).
     - *NumPy Array vs PyTorch Tensor*: Selisih numerik $\Delta = 0.00\text{e}+00$ (identik bit-for-bit).
     - *File/Bytes vs Array/Tensor*: Selisih maksimal $\Delta < 2.0\times 10^{-4}$ (murni batas derau kuantisasi 16-bit PCM saat file disimpan ke WAV).
  2. **Stereo-to-Mono Channel Downmix**:
     - Menguji konversi audio 2-channel stereo menjadi 1D mono melalui perataan kanal $x_{\text{mono}}[n] = \frac{x_L[n] + x_R[n]}{2}$.
  3. **Resampling Fidelity ($48\text{ kHz}$ & $44.1\text{ kHz}$ $\rightarrow 16\text{ kHz}$)**:
     - Menguji rekaman dari mikrofon laptop Realtek ($48\text{ kHz}$) dan mikrofon USB studio ($44.1\text{ kHz}$) yang di-resample ke standar model $16\text{ kHz}$.
     - Retensi nada uji $500\text{ Hz}$ pasca-resampling mencapai korelasi $1.4141 > 0.90$ (amplitudo puncak ternormalisasi $\sqrt{2}$).
  4. **Robust Error Handling**:
     - Memastikan input invalid (tipe data dict, integer, atau file path yang tidak ditemukan) melempar `TypeError` dan `FileNotFoundError` secara eksplisit dan aman tanpa crash internal.
* **Hasil Pengujian Unit Test Riil**:
  - Seluruh **13 unit test** di `tests/test_audio.py` lulus 100%:
    - `test_normalize_waveform` PASSED
    - `test_adaptive_rms_normalize_quiet_and_loud` PASSED
    - `test_adaptive_rms_normalize_silence_and_gain_clamp` PASSED
    - `test_butterworth_bandpass_attenuation` PASSED
    - `test_trim_silence_vad_speech_padding` PASSED
    - `test_trim_silence_vad_silence_and_click_fallback` PASSED
    - `test_trim_silence_vad_continuous` PASSED
    - `test_preprocess_audio_multi_format_parity` PASSED
    - `test_preprocess_audio_stereo_to_mono_parity` PASSED
    - `test_preprocess_audio_resampling_fidelity` PASSED
    - `test_preprocess_audio_error_handling` PASSED
    - `test_preprocess_audio_crop_and_pad` PASSED
    - `test_collate_fn_pad_batch` PASSED
  - Status pytest: `13 passed in 24.89s`.
* **Pencapaian Milestone**:
  - **FASE 1: AUDIT & STANDARISASI PEMROSESAN SINYAL AUDIO (SIGNAL HYGIENE) SELESAI 100%**.
* **Nomor Commit Git**:
  - `commit 448f146` (*test(audio): verify comprehensive signal preprocessing parity and complete Phase 1 (Sub-step 1.4)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

### FASE 2: REKAYASA KORPUS NLP BAHASA INDONESIA SEHARI-HARI (TARGET 4.000 - 5.000 FRASA)

#### Sub-step 2.1: Pembuatan Generator Matriks Leksikon, Slang, & Partikel Spontan (Python Script)
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi & Verifikasi Kode)
* **Latar Belakang & Masalah**:
  Korpus kalimat bahasa Indonesia sebelumnya hanya berjumlah $1.085$ frasa ($35$ frasa per intent). Meskipun sudah mencakup beberapa kata gaul, variasi sintaksisnya belum memadai untuk merefleksikan fleksibilitas penutur bahasa Indonesia sehari-hari yang sering membalik struktur kalimat (contoh: *"Lampu kamar tolong matiin"*, *"Di dapur tolong nyalakan lampunya"*, *"Kamar gerah tolong sejukin ac-nya"*). Diperlukan generator deterministik berbasis aturan linguistik komprehensif untuk mengekspansi data ke skala $4.000 - 5.000$ kalimat.
* **Formulasi Matematika & Arsitektur Generator**:
  - Script generator dibangun di `scripts/generate_massive_indonesian_corpus.py`.
  - Mengintegrasikan matriks leksikon super lengkap & komprehensif:
    - **6 Aksi**: Base formal, slang/kolokial (*"nyalain", "hidupin", "idupin", "naikin", "turunin", "bawain", "colokin", "pencet on", "geber", "gas", "pasangin", "ademin", "pelanin"*), bentuk pasif (*"dinyalakan", "dimatiin", "dinaikin", "dikecilin"*), dan bentuk perintah singkat (*"on", "off", "up", "down", "play", "cut", "plus", "minus"*).
    - **14 Objek**: Sinonim luas lampu (*"bohlam", "penerangan", "lampu tidur", "lampu nakas", "lampu meja", "night lamp", "lampu gantung", "lampu plafon"*), musik/audio (*"lagu", "playlist", "audio", "track", "alunan musik", "mp3", "sound"*), volume (*"suara", "bunyi", "speaker", "level suara", "kencengnya suara"*), suhu (*"ac", "pendingin", "penghangat", "air conditioner", "hawa ruangan", "pemanas", "derajat ac"*), barang (*"jus", "koran", "sepatu", "kaos kaki"*), dan bahasa (*"mandarin", "inggris", "jerman", "korea"*).
    - **4 Lokasi**: Dapur (*"ruang masak", "tempat masak", "kitchen", "dapur belakang"*), kamar tidur (*"kamar", "ruang tidur", "kamar utama", "bedroom", "bilik"*), kamar mandi/toilet/wc (*"kamar kecil", "washroom", "restroom", "kamar bilas"*), dan none/global (*"di sini", "sekarang", "ruangan ini", "semuanya", "seisi rumah"*).
    - **Panggilan, Vocatives & Partikel Daerah**: 28 vocatives & partikel pembuka (*"tolong", "coba", "eh", "bro", "cuy", "gan", "min", "bang", "mas", "mbak", "kak", "svara", "hey svara", "punten", "mangga", "woy", "lur"*), 22 partikel penutup (*"dong", "deh", "lah", "sih", "ya", "kan", "cepetan", "atuh", "dong atuh", "ya bro", "ya cuy", "nih", "tuh"*), 15 adverbia intensitas (*"dikit", "banyak", "pol", "banget", "amat", "full", "maksimal", "perlahan", "kenceng"*), dan 9 konteks situasional keluhan (*"gelap", "gerah", "berisik", "haus", "kedinginan", "silau", "mau kerja"*).
  - Mengimplementasikan **6 Pola Permutasi Sintaksis**:
    1. Pola 1 (Standar): `[Pembuka] + Aksi + Objek + Lokasi + [Partikel]`
    2. Pola 2 (Aksi + Partikel di Depan): `[Pembuka] + Aksi + Partikel + Objek + Lokasi`
    3. Pola 3 (Objek di Depan): `Objek + Lokasi + [Tolong] + Aksi Pasif/Aktif + [Partikel]`
    4. Pola 4 (Lokasi di Depan): `Lokasi + Objek + [Tolong] + Aksi + [Partikel]`
    5. Pola 5 (Implisit / Kontekstual): `Kondisi + [Tolong] + Aksi + Objek + Lokasi`
    6. Pola 6 (Perintah Ringkas): `Objek + Lokasi + Aksi Singkat` / `Aksi Singkat + Objek + Lokasi`
  - Total ruang kandidat permutasi mentah yang dievaluasi: **$3.127.964$ variasi kombinasi** yang disaring dan disampel secara deterministik (`seed=42`).
* **File yang Dibuat / Dimutakhirkan**:
  - `scripts/generate_massive_indonesian_corpus.py` (Script generator utama super lengkap).
  - `data/corpus/indonesian_smart_home_corpus_4k.csv` (Dataset teks 4.495 kalimat seimbang).
  - `tests/test_corpus.py` (Suite pengujian integritas korpus dan keberagaman leksikon).
* **Hasil Pengujian Unit Test Riil**:
  - `test_corpus_exists_and_row_count`: PASSED (4.495 kalimat unik, melebihi kuota 4.000)
  - `test_corpus_required_columns`: PASSED (100% kolom terisi lengkap)
  - `test_corpus_stratified_distribution`: PASSED (Tepat 145 kalimat/intent merata across 31 intent)
  - `test_corpus_linguistic_diversity`: PASSED (333 kosakata unik terverifikasi, partikel *atuh, cuy, bro* lolos)
  - `test_corpus_zero_empty_or_trivial_phrases`: PASSED (Zero empty phrases)
  - Status pytest: `5 passed in 1.59s`.
* **Nomor Commit Git**:
  - `commit 713b11b` (*feat(nlp): expand to super-complete Indonesian colloquial lexicon (333 words, 4,495 phrases) (Sub-step 2.1)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

#### Sub-step 2.2: Eksekusi Generasi Korpus Teks 31 Intent (130-150 Frasa per Intent, Total 4.000+ Kalimat)
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi & Verifikasi Ilmiah)
* **Latar Belakang & Masalah**:
  Untuk mencegah bias kelas (class imbalance) pada multi-task classification head Spoken Language Understanding (SLU), setiap intent dari ke-31 intent yang didefinisikan dalam Fluent Speech Commands (FSC) harus memiliki representasi sampel data teks berbahasa Indonesia yang benar-benar seimbang dan proporsional. Target kuota per intent ditetapkan pada rentang 130 - 150 variasi frasa unik agar total korpus mencapai skala 4.000 - 5.000 kalimat yang mencakup 5 domain fungsional rumah pintar.
* **Parameter & Formulasi Eksekusi Batch**:
  - Script generator: `scripts/generate_massive_indonesian_corpus.py` (CLI berparameter `--intent-map`, `--output`, `--target-per-intent`, `--seed`).
  - Target per intent: $N_{\text{target}} = 145$ frasa unik per intent ($145 \in [130, 150]$).
  - Total intent: $K = 31$ intent (Intent ID 0 sampai 30).
  - Total kandidat ruang permutasi yang dievaluasi: $3.127.964$ variasi kombinasi.
  - Total frasa bersih terpilih: $N_{\text{total}} = 31 \times 145 = 4.495$ kalimat unik.
  - Random seed: $42$ (Deterministic Sampling).
  - Standar Deviasi Sebaran Antar-Kelas: $\sigma_{\text{intent}} = 0.00$ (Zero Class Imbalance / 100% Balanced).
* **Distribusi Domain Fungsional Smart Home**:
  1. **Lighting (Pencahayaan)**: $10$ intent (Intent 0-4, 15-19) $\rightarrow 1.450$ frasa ($32.3\%$)
  2. **Climate (Pengatur Suhu & AC)**: $8$ intent (Intent 21-24, 26-29) $\rightarrow 1.160$ frasa ($25.8\%$)
  3. **Media & Audio (Hiburan & Volume)**: $4$ intent (Intent 5, 20, 25, 30) $\rightarrow 580$ frasa ($12.9\%$)
  4. **Robot Butler (Pengantaran Barang)**: $4$ intent (Intent 6-9) $\rightarrow 580$ frasa ($12.9\%$)
  5. **System Language (Pengaturan Bahasa)**: $5$ intent (Intent 10-14) $\rightarrow 725$ frasa ($16.1\%$)
  - **Total Keseluruhan**: $31$ intent $\rightarrow 4.495$ frasa ($100.0\%$).
* **Statistik Panjang Kalimat & Karakter**:
  - Jumlah Kosakata Unik (Lexicon Size): $331$ kata unik.
  - Panjang Kalimat (Kata): Rata-rata $6.39 \pm 1.33$ kata (Min: $2$ kata, Max: $11$ kata).
  - Panjang Karakter: Rata-rata $37.88$ karakter (Min: $10$ karakter, Max: $67$ karakter).
* **Artefak File**:
  - `data/corpus/indonesian_smart_home_corpus_4k.csv` ($4.495$ baris data teks teranotasi lengkap: `intent_id`, `action`, `object`, `location`, `domain`, `english_sample`, `indonesian_phrase`).
  - `tests/test_corpus.py` (Pengujian sebaran domain dan metrik statistik kalimat).
* **Hasil Pengujian Unit Test Riil**:
  - `test_corpus_exists_and_row_count`: PASSED ($4.495$ baris $\ge 4.000$)
  - `test_corpus_required_columns`: PASSED ($7$ kolom lengkap tanpa nilai NaN)
  - `test_corpus_stratified_distribution`: PASSED (Tepat $145$ kalimat per intent across ke-31 intent)
  - `test_corpus_domain_distribution`: PASSED (Pencahayaan $1.450$, Suhu $1.160$, Media $580$, Robot $580$, Bahasa $725$)
  - `test_corpus_length_statistics`: PASSED (Mean kata $6.39$, Min $2$, Max $11$)
  - `test_corpus_linguistic_diversity`: PASSED (Slang, kolokial, partikel daerah *atuh, bro, cuy* terverifikasi)
  - `test_corpus_zero_empty_or_trivial_phrases`: PASSED (Zero empty phrases)
  - Status pytest: `7 passed in 2.02s` (dan seluruh `20 passed in 11.67s` bersama suite audio).
* **Nomor Commit Git**:
  - `commit 9fcf68b` (*feat(corpus): execute and verify official 4,495-phrase balanced batch generation across 31 intents (Sub-step 2.2)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

#### Sub-step 2.3: Audit & Validasi Integritas Linguistik (Deduplikasi, Cek Sebaran, Ekspor CSV)
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi & Validasi Ilmiah)
* **Latar Belakang & Masalah**:
  Sebelum korpus teks dijadikan input masukan untuk sintesis suara massal (Neural TTS) pada Fase 3, data harus diaudit secara berlapis (multi-dimensional integrity audit). Hal ini untuk memastikan:
  1. Tidak ada satupun frasa yang terduplikasi secara identik (Exact Deduplication).
  2. Tidak ada satupun frasa yang memiliki ambiguitas lintas-intent (Zero Cross-Intent Collision).
  3. Keseimbangan kelas memenuhi batas entropi informasi Shannon teoritis maksimal ($H / H_{\max} = 1.000000$).
  4. Seluruh slot aksi, objek, dan lokasi selaras 100% dengan taksonomi `configs/intent_map.json`.
  5. Kehadiran partikel daerah (*atuh, bro, cuy, lur, gan*) dan verba gaul tersebar merata secara empiris.
* **Formulasi Matematika & Metrik Audit**:
  - Script audit: `scripts/audit_indonesian_corpus.py`.
  - Entropi Shannon Sebaran Kelas:
    $$H(X) = - \sum_{i=1}^{K} p_i \log_2(p_i) = -\sum_{i=1}^{31} \frac{145}{4495} \log_2\left(\frac{145}{4495}\right) = 4.9542\text{ bits}$$
    $$H_{\max} = \log_2(31) = 4.9542\text{ bits}$$
    $$\text{Uniform Entropy Ratio} = \frac{H(X)}{H_{\max}} = \mathbf{1.000000}$$
  - Tingkat Duplikasi Teks:
    $$\text{Duplicate Rate} = \frac{N_{\text{total}} - N_{\text{unique}}}{N_{\text{total}}} = \frac{4495 - 4495}{4495} = \mathbf{0.00\%}$$
  - Tingkat Kolisi Lintas-Intent:
    $$\text{Cross-Intent Collision Rate} = \frac{N_{\text{ambiguous}}}{N_{\text{total}}} = \mathbf{0.00\%}$$
  - Type-Token Ratio (TTR):
    $$\text{TTR} = \frac{|V|}{N_{\text{tokens}}} = \frac{317}{29169} = 0.0109$$
* **Hasil Pengujian Empiris 7 Dimensi**:
  1. **Deduplikasi & Kolisi**: Total 4.495 frasa, 4.495 unik ($100\%$), 0 duplikat ($0.00\%$), 0 kolisi intent ($0.00\%$). Status: **PASS**.
  2. **Stratifikasi & Entropi**: 31 intent lengkap, tepat 145 frasa/intent, $\sigma = 0.0000$, $\text{Variance} = 0.0000$, Rasio Entropi $1.000000$. Status: **PASS**.
  3. **Keselarasan Slot FSC**: 6 aksi, 14 objek, 4 lokasi terpetakan sempurna tanpa nilai NaN dan tanpa pergeseran label. Status: **PASS**.
  4. **Panjang Kalimat Spontan**: Rata-rata $6.49 \pm 1.31$ kata (Rentang $2 - 12$ kata, Karakter $37.85 \pm 8.2$), $0$ kalimat trivial kosong. Status: **PASS**.
  5. **Sebaran Token Leksikon**: $29.169$ total token, $317$ kosakata unik, verba gaul aktif (`nyalain`: 40, `matiin`: 60, `idupin`: 45, `naikin`: 29, `turunin`: 34, `bawain`: 85, `ambilin`: 69, `setel`: 123), partikel (*lur*: 780, *dong*: 777, *ya*: 1.085, *bro*: 469, *cuy*: 126, *atuh*: 319, *deh*: 417, *lah*: 390, *tolong*: 1.011, *gan*: 125). Status: **PASS**.
  6. **Sebaran Domain Smart Home**: Pencahayaan $1.450$ frasa ($32.26\%$), Pengatur Suhu & AC $1.160$ frasa ($25.81\%$), Hiburan & Volume $580$ frasa ($12.9\%$), Robot Butler $580$ frasa ($12.9\%$), Sistem Bahasa $725$ frasa ($16.13\%$).
  7. **Status Keseluruhan**: **PASSED (100% CLEAN)**.
* **Artefak File yang Dihasilkan**:
  - `scripts/audit_indonesian_corpus.py` (Script auditor ilmiah 7 dimensi berstandar publikasi).
  - `reports/corpus_linguistic_audit.json` (Laporan audit machine-readable untuk integrasi tabel laporan bab 3/4).
  - `reports/corpus_linguistic_audit.md` (Dokumen laporan formal akademik dalam bahasa Indonesia).
  - `tests/test_corpus.py` (Suite pengujian bertambah menjadi 11 unit test integritas).
* **Hasil Pengujian Unit Test Riil**:
  - `test_corpus_exists_and_row_count`: PASSED
  - `test_corpus_required_columns`: PASSED
  - `test_corpus_stratified_distribution`: PASSED
  - `test_corpus_domain_distribution`: PASSED
  - `test_corpus_length_statistics`: PASSED
  - `test_corpus_linguistic_diversity`: PASSED
  - `test_corpus_zero_empty_or_trivial_phrases`: PASSED
  - `test_corpus_zero_duplicates_and_collisions`: PASSED
  - `test_corpus_shannon_entropy_uniformity`: PASSED
  - `test_corpus_slot_alignment_with_intent_map`: PASSED
  - `test_corpus_audit_reports_exist`: PASSED
  - Status pytest: `11 passed in 2.39s` (dan seluruh `24 passed in 10.56s` bersama suite audio).
* **Pencapaian Milestone**:
  - **FASE 2: REKAYASA KORPUS NLP BAHASA INDONESIA SELESAI 100%**.
* **Nomor Commit Git**:
  - `commit ce620c2` (*feat(audit): implement 7-dimension linguistic integrity audit and complete Phase 2 (Sub-step 2.3)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

### FASE 3: SINTESIS SUARA MULTI-SPEAKER MASSAL (NEURAL TTS TARGET 8.000 - 10.000 AUDIO BERSIH)

#### Sub-step 3.1: Pembuatan Engine Sintesis Paralel Multi-Worker (Edge Neural TTS Python)
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi & Validasi Engine Sintesis)
* **Latar Belakang & Masalah**:
  Untuk mengubah korpus teks $4.495$ kalimat menjadi audio ucapan nyata yang beragam, sistem membutuhkan engine text-to-speech (TTS) skala industri dengan arsitektur asinkronus multi-worker. Men-download $\approx 9.000$ audio secara sekuensial akan memakan waktu berjam-jam dan rentan putus koneksi. Selain itu, Microsoft Edge TTS menghasilkan stream terkompresi MP3 $24\text{ kHz}$ yang harus diubah secara *in-memory* menjadi standar baku SLU ($16\text{ kHz}$ 16-bit PCM Mono WAV) dengan kontrol headroom dan anti-clipping.
* **Arsitektur & Parameter Engine Sintesis**:
  - Script engine: `scripts/synthesize_massive_indonesian.py` (Class `MassiveTTSAsyncEngine`).
  - Worker Pool: Asynchronous Producer-Consumer Queue dengan konkurensi $N=15$ workers (`asyncio.Semaphore(15)`).
  - Voice Mapping:
    1. **Male Voice**: `id-ID-ArdiNeural` (Speaker ID: `spk_ardi`)
    2. **Female Voice**: `id-ID-GadisNeural` (Speaker ID: `spk_gadis`)
  - Target Format Audio: WAV, Sample Rate $16.000\text{ Hz}$, Channels $1$ (Mono), Bit Depth 16-bit PCM (`subtype="PCM_16"`).
  - Konversi Audio & Signal Hygiene:
    1. Decode MP3 in-memory via `soundfile.read` ($0$ file temporer disk untuk MP3).
    2. Downmix stereo-to-mono secara aritmetika vektor.
    3. Polyphase sinc resampling dari $24\text{ kHz}$ ke $16\text{ kHz}$ menggunakan rasio koprima $p=2, q=3$ (`scipy.signal.resample_poly`) tanpa distorsi fase.
    4. Headroom peak normalization limit $\le 0.95$ (headroom $-0.5\text{ dBFS}$) untuk mencegah clipping pada input model Transformer.
  - Mekanisme Ketahanan Jaringan:
    1. Exponential backoff retry ($1.0\text{s}, 2.0\text{s}, 4.0\text{s}, 8.0\text{s}$ + random jitter) hingga 4 kali percobaan per frasa.
    2. Automatic resumption & checkpointing: jika proses dihentikan sewaktu-waktu, file WAV yang sudah valid di disk langsung di-*skip* tanpa download ulang.
  - Upgrade Dependensi: Memperbarui library `edge-tts` ke versi `7.3.1` untuk menangani protokol token `Sec-MS-GEC` terbaru dari Microsoft WebSocket API.
* **Artefak File yang Dibuat**:
  - `scripts/synthesize_massive_indonesian.py` (Engine sintesis paralel multi-worker).
  - `tests/test_synthesis.py` (Suite pengujian unit test engine sintesis, konversi audio, dan live smoke test).
  - `.gitignore` (Konfigurasi whitelist metadata CSV `data/synthetic_indonesian_massive/*.csv` dan ignore audio `.wav`).
* **Hasil Pengujian Unit Test Riil**:
  - `test_audio_conversion_mono_resample_headroom`: PASSED (16 kHz mono, peak <= 0.95, sinc filter akurat)
  - `test_verify_wav_file`: PASSED (Validasi header WAV, sample rate, bit depth, dan integritas file)
  - `test_engine_initialization_and_voices`: PASSED (Pemetaan suara Ardi & Gadis akurat)
  - `test_engine_dry_run_pipeline`: PASSED (Simulasi antrian worker dan pembuatan metadata manifest 100% lulus)
  - `test_live_single_synthesis_smoke`: PASSED (Sintesis langsung live Edge TTS menghasilkan audio 16 kHz 16-bit PCM Mono valid, lolos verifikasi cache resumability)
  - Status pytest: `5 passed in 21.42s` (dan seluruh `29 passed in 10.53s` di seluruh test suite repositori).
* **Nomor Commit Git**:
  - `commit 077af48` (*feat(tts): implement multi-worker parallel synthesis engine with auto-retry and polyphase resampling (Sub-step 3.1)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

#### Sub-step 3.2: Eksekusi Batch Sintesis Suara Multi-Speaker (8.990 Audio Klip Bersih)
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Eksekusi Batch Paralel & Verifikasi Distribusi Audio)
* **Latar Belakang & Masalah**:
  Dengan tersedianya korpus berimbang 4.495 kalimat pada Sub-step 2.2 dan engine sintesis asinkronus multi-worker pada Sub-step 3.1, langkah selanjutnya adalah mengeksekusi konversi seluruh korpus teks menjadi audio klip ucapan manusia nyata berkualitas tinggi. Diperlukan dua profil pembicara alami (suara pria dan wanita) untuk merepresentasikan keragaman gender, fonetik, dan pitch fundamental ($F_0$) penutur bahasa Indonesia.
* **Parameter Eksekusi Batch & Throughput**:
  - Konkurensi Worker: 15 parallel async workers (`--concurrency 15`).
  - Total Target Unduhan: $4.495\text{ frasa} \times 2\text{ speaker} = 8.990\text{ file audio WAV}$.
  - Durasi Eksekusi Riil: $1.258,60\text{ detik}$ ($20,98\text{ menit}$).
  - Kecepatan Rata-rata (*Throughput*): $\approx 7,14\text{ audio/detik}$ ($428,5\text{ audio/menit}$).
  - Rasio Keberhasilan (*Success Rate*): $8.990 / 8.990$ ($100,0\%$).
  - Tingkat Kegagalan (*Failure Rate*): $0$ file ($0,0\%$).
* **Distribusi & Karakteristik Data Audio yang Dihasilkan**:
  1. **Sebaran Pembicara (Speaker & Gender)**:
     - `spk_ardi` (Pria, `id-ID-ArdiNeural`): Tepat $4.495$ file WAV ($50,0\%$).
     - `spk_gadis` (Wanita, `id-ID-GadisNeural`): Tepat $4.495$ file WAV ($50,0\%$).
     - Rasio Gender: Seimbang sempurna $1:1$ ($\sigma = 0,00$).
  2. **Sebaran Intent (31 Intent Classes)**:
     - Setiap intent memiliki tepat $145\text{ frasa} \times 2\text{ suara} = 290\text{ file audio WAV}$.
     - Minimum file per intent: $290$, Maksimum: $290$ ($\sigma = 0,00$, zero class imbalance).
  3. **Statistik Durasi Audio**:
     - Total Durasi Riil: $8,31\text{ jam}$ ($29.918,4\text{ detik}$).
     - Rata-rata Durasi per Klip ($\mu$): $3,33\text{ detik}$.
     - Standar Deviasi Durasi ($\sigma$): $0,53\text{ detik}$.
     - Rentang Durasi: Minimum $1,92\text{ detik}$, Maksimum $5,78\text{ detik}$.
     - Persentil Durasi: 25% = $2,93\text{s}$, 50% (Median) = $3,26\text{s}$, 75% = $3,70\text{s}$.
  4. **Standar Sinyal Akustik (Signal Hygiene)**:
     - Seluruh $8.990$ file tersimpan dalam format standar: $16.000\text{ Hz}$, $1$ channel (Mono), 16-bit Signed PCM (`PCM_16`).
     - Normalisasi peak headroom terjaga pada $\le 0,95$ (headroom $-0,5\text{ dBFS}$ mencegah clipping).
     - Sampling integritas numerik 50 file membuktikan: $0$ NaN, $0$ Inf, $0$ file rusak/kosong.
* **Artefak File yang Dihasilkan**:
  - `data/synthetic_indonesian_massive/wavs_16k/`: Berisi tepat $8.990$ file WAV (tersaring aman oleh `.gitignore`, tidak masuk commit git publik).
  - `data/synthetic_indonesian_massive/metadata.csv`: Berisi $8.990$ baris data manifest dengan 11 kolom lengkap (`path, speakerId, gender, voice_name, transcript, action, object, location, domain, intent_id, duration`).
* **Hasil Pengujian Test Suite Repositori**:
  - Seluruh **78 unit test** di direktori `tests/` lulus $100\%$ tanpa ada error (`78 passed, 7 warnings in 50.19s`).
* **Pencapaian Milestone**:
  - Sub-step 3.2 selesai sempurna dengan $8.990$ klip audio bersih tanpa kegagalan ($0$ fail).
* **Nomor Commit Git**:
  - `commit 305d73e` (*feat(dataset): synthesize 8,990 clean Indonesian voice clips across 2 speakers (Sub-step 3.2)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

#### Sub-step 3.3: Verifikasi Zero-Corruption Akustik & Signal Hygiene 7 Dimensi
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi Audit, Eksekusi Pengujian Sinyal, & Laporan Ilmiah)
* **Latar Belakang & Masalah**:
  Meskipun proses sintesis batch pada Sub-step 3.2 dilaporkan selesai, sesuai **Rule 1 (Zero Fabrication)** dan **Rule 7 (Train/Serve Preprocessing Parity)** pada `AGENTS.md`, sistem dilarang berasumsi bahwa file audio di harddisk secara otomatis bebas dari kecacatan. Diperlukan audit akustik independen dan menyeluruh (100% sampel audio) untuk membuktikan integritas biner, keselarasan manifes-disk, keseragaman frekuensi sampling 16 kHz mono, kepatuhan batas headroom puncak ($\le 0,95$), ketiadaan nilai invalid numerik (NaN/Inf), dan ketiadaan keheningan palsu.
* **Arsitektur & 7 Dimensi Audit Akustik**:
  - Script audit: `scripts/audit_synthetic_dataset.py`.
  - Tingkat inspeksi: **100% sampel audio** ($8.990$ file diperiksa satu per satu menggunakan multi-threading 16 workers dalam $12,72$ detik atau $706,7$ file/detik).
  1. **Bijeksi Manifes vs Harddisk**:
     - Total baris metadata: $8.990$. Total file WAV pada disk: $8.990$.
     - Missing files: $0$. Orphan files: $0$. Status: **PASS**.
  2. **Kepatuhan Header Akustik**:
     - Format container: RIFF WAV ($8.990 / 8.990 = 100,0\%$).
     - Sample Rate: Tepat $16.000\text{ Hz}$ ($8.990 / 8.990 = 100,0\%$).
     - Kanal: $1\text{ Channel (Mono)}$ ($8.990 / 8.990 = 100,0\%$).
     - Tipe data / encoding: 16-bit Signed Linear PCM (`PCM_16`).
     - Sinkronisasi durasi header vs manifest: $100\%$ klip audio cocok dengan catatan durasi manifest (toleransi $\le 50\text{ ms}$). Status: **PASS**.
  3. **Integritas Numerik Waveform & Signal Hygiene**:
     - Nilai NaN / Not-a-Number: $0$ sampel di seluruh $8.990$ file audio.
     - Nilai Inf / Tak Hingga: $0$ sampel di seluruh $8.990$ file audio.
     - Batas Amplitudo Puncak (Headroom Peak Limit $\le 0,9500$): $0$ pelanggaran (puncak tertinggi terukur $= 0,8180 \le 0,95$).
     - Uji Audio Hening Palsu (RMS Threshold $\ge 0,005$): $0$ file mati/hening (rata-rata RMS $= 0,1115$ atau setara $-19,23\text{ dBFS}$).
     - Bias Arus Searah (DC Offset Bias $\le 0,01$): $0$ pelanggaran. Status: **PASS**.
  4. **Keseimbangan Gender & Pembicara**:
     - Pria (`spk_ardi`): $4.495$ file audio ($50,0\%$).
     - Wanita (`spk_gadis`): $4.495$ file audio ($50,0\%$).
     - Selisih antar gender: $\Delta = 0$ (rasio tepat $1:1$). Status: **PASS**.
  5. **Keseimbangan Kelas Intent & Entropi Informasi**:
     - Jumlah intent unik: $31$ kelas intent.
     - Distribusi per intent: Tepat $290$ audio klip per intent ($\mu = 290,0, \sigma = 0,00$, zero class imbalance).
     - Entropi Shannon: $H(X) = 4,954196\text{ bits}$.
     - Rasio Entropi Ternormalisasi: $H(X) / H_{\max} = 1,000000$ (keseragaman kelas sempurna). Status: **PASS**.
  6. **Sebaran Domain Smart Home**:
     - Lighting: $2.900$ audio klip ($32,26\%$, 10 intent).
     - Climate: $2.320$ audio klip ($25,81\%$, 8 intent).
     - System Language: $1.450$ audio klip ($16,13\%$, 5 intent).
     - Robot Butler: $1.160$ audio klip ($12,90\%$, 4 intent).
     - Media & Audio: $1.160$ audio klip ($12,90\%$, 4 intent). Status: **PASS**.
  7. **Statistik Durasi & Penyimpanan Data**:
     - Total Durasi Suara Bersih: $8,31\text{ jam}$ ($29.918,38\text{ detik}$).
     - Rata-rata Durasi per Klip: $3,33\text{ detik}$ ($\sigma = 0,53\text{ detik}$, rentang $1,92\text{s} - 5,78\text{s}$).
     - Total Ukuran Disk: $913,41\text{ MB}$ ($957.783.592\text{ bytes}$).
     - Rata-rata Ukuran per File: $104,04\text{ KB}$. Status: **PASS**.
* **Artefak File yang Dihasilkan**:
  - `scripts/audit_synthetic_dataset.py` (Script audit akustik 7 dimensi multi-threaded).
  - `reports/synthetic_acoustic_audit.json` (Laporan audit komprehensif format JSON).
  - `reports/synthetic_acoustic_audit.md` (Laporan audit formal format Markdown untuk laporan skripsi).
  - `tests/test_acoustic_audit.py` (Suite pengujian unit test 5 fungsi untuk verifikasi otomatis CI/CD).
* **Hasil Pengujian Test Suite Repositori**:
  - Seluruh **10 unit test** pada suite sintesis dan audit akustik (`tests/test_acoustic_audit.py` & `tests/test_synthesis.py`) lulus $100\%$ dalam $12,20\text{ detik}$.
  - Status Audit Global: **PASS (100% LOLOS UJI INTEGRITAS AKUSTIK)**.
* **Pencapaian Milestone**:
  - **FASE 3: SINTESIS SUARA MULTI-SPEAKER MASSAL (NEURAL TTS TARGET 8.000 - 10.000 AUDIO) SELESAI 100%**.
* **Nomor Commit Git**:
  - `commit d8c084e` (*feat(audit): implement 7-dimension acoustic integrity audit and complete Phase 3 (Sub-step 3.3)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

### FASE 4: ENVIRONMENTAL ACOUSTIC AUGMENTATION (TARGET 24.000 AUDIO ROBUST)

#### Sub-step 4.1: Implementasi Schroeder Reverberator Akustik Ruang Pintar (Kamar Tidur & Ruang Tamu)
* **Tanggal Eksekusi**: 27 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi DSP, Akselerasi Vektor Chunk, & Validasi Fisika Akustik)
* **Latar Belakang & Masalah**:
  Audio sintesis suara pada Fase 3 berkarakter *anechoic* (kering tanpa pantulan akustik ruangan). Pada skenario nyata di rumah tinggal, pengguna mengucapkan perintah suara di dalam ruangan tertutup (kamar tidur dengan peredaman furnitur/gorden atau ruang tamu/keluarga yang luas dengan pantulan lantai keramik/ubin). Hal ini menyebabkan pantulan gelombang suara (*multipath reflections*) yang mengubah karakteristik envelope dan fasa sinyal. Tanpa augmentasi reverberasi ruangan, model SLU rentan mengalami degradasi akurasi signifikan di lingkungan fisik riil.
* **Formulasi Matematika & Arsitektur Schroeder Reverberator**:
  Menerapkan arsitektur digital legendaris Manfred R. Schroeder (1962, *Natural-Sounding Artificial Reverberation*):
  1. **4 Feedback Comb Filters (FBCF) Paralel**:
     - Memodelkan pantulan diskrit awal (*early reflections*) dari dinding, lantai, dan langit-langit.
     - Persamaan beda linier:
       $$y[n] = x[n] + g \cdot y[n - D]$$
     - Nilai delay $D$ (dalam sampel pada $f_s = 16.000\text{ Hz}$) dipilih sebagai bilangan prima relatif (*mutually coprime*) untuk mencegah resonansi modal tumpang tindih (*flutter echo*).
     - Koefisien gain umpan balik $g$ diturunkan dari persamaan Eyring-Schroeder terhadap waktu peluruhan $RT_{60}$:
       $$g = 10^{-\frac{3 D}{RT_{60} \cdot f_s}}$$
  2. **2 All-Pass Filters (APF) Kaskade**:
     - Mendispersikan kerapatan fasa dan memecah gema (*phase decorrelation & echo diffusion*) dengan respons magnitudo flat tepat $1,0$ (*all-pass unity gain*):
       $$H(z) = \frac{-g_{\text{ap}} + z^{-D_{\text{ap}}}}{1 - g_{\text{ap}} z^{-D_{\text{ap}}}}, \quad g_{\text{ap}} = 0,7$$
  3. **Akselerasi Numerik Chunk-Vectorization**:
     - Filter IIR sparse delay panjang diakselerasi menggunakan pembagian blok vektor NumPy sepanjang $D$ sampel. Waktu komputasi terpangkas drastis dari $156\text{ ms}$ menjadi **$8,89\text{ ms}$ per klip 3 detik** ($399,6\times$ lebih cepat dari pemutaran audio real-time).
  4. **Profil Akustik Ruangan Smart Home**:
     - **Bedroom (Kamar Tidur)**: $RT_{60} = 0,28\text{ detik}$, rasio basah $\text{wet} = 0,18$, comb delays $[479, 587, 691, 797]$, APF $[89, 23]$.
     - **Living Room (Ruang Tamu / Keluarga)**: $RT_{60} = 0,48\text{ detik}$, rasio basah $\text{wet} = 0,28$, comb delays $[739, 887, 1039, 1187]$, APF $[149, 43]$.
     - **Kitchen (Dapur)**: $RT_{60} = 0,36\text{ detik}$, rasio basah $\text{wet} = 0,22$, comb delays $[599, 709, 829, 953]$, APF $[109, 31]$.
  5. **Kontrol Headroom Anti-Clipping**:
     - Amplitudo puncak sinyal hasil pencampuran $((1 - w)x + wy_{\text{ap}})$ dibatasi pada batas aman $\le 0,9500$ (headroom $-0,5\text{ dBFS}$) untuk menjaga stabilitas numerik ekstraktor fitur wav2vec2.
* **Artefak File yang Dibuat & Dimodifikasi**:
  - `src/svara/data/augment.py`: Penambahan fungsi `_feedback_comb_filter`, `_schroeder_allpass_filter`, `apply_schroeder_reverb`, preset `SCHROEDER_ROOM_PRESETS`, kelas `SchroederReverberator`, dan integrasi probabilitas reverb ke `AudioAugmentor`.
  - `tests/test_augment.py`: Penambahan 5 unit test komprehensif (verifikasi bentuk & non-NaN, dry passthrough, silence passthrough, perbandingan rasio energi ekor kamar tidur vs ruang tamu, dan integrasi augmentor).
  - `scripts/demo_schroeder_reverb.py`: Script benchmark empiris dan pengujian profil akustik ruangan.
  - `reports/schroeder_reverb_benchmark.json`: Data hasil evaluasi akustik dan kecepatan pemrosesan.
* **Hasil Pengujian & Verifikasi Fisika Akustik**:
  - **Fisika Peluruhan Energi**: Ekor energi pantulan ruang keluarga terbukti $66,25\times$ lebih besar daripada kamar tidur ($E_{\text{living}} = 7,26\times 10^{-4}$ vs $E_{\text{bedroom}} = 1,10\times 10^{-5}$), memvalidasi secara fisik ruangan bervolume lebih besar memiliki reverberasi lebih panjang.
  - **Kepatuhan Headroom**: Peak audio teruji pada `0.6073` (kamar tidur) dan `0.6012` (ruang tamu), 100% patuh di bawah batas `0.9500`.
  - **Throughput & Kecepatan**: $112,5\text{ audio/detik}$ (single-thread), rata-rata latensi $8,89\text{ ms}$, Real-Time Factor (RTF) mencapai $399,6\times$.
  - **Unit Test Repositori**: Seluruh **88 unit test** di `tests/` lulus $100\%$ tanpa ada kegagalan (`88 passed, 7 warnings in 28.24s`).
* **Pencapaian Milestone**:
  - Sub-step 4.1 selesai sempurna. Schroeder Reverberator siap dipakai untuk augmentasi skala industri pada sub-step 4.4.
* **Nomor Commit Git**:
  - `commit 1678e37` (*feat(augment): implement Schroeder reverberator with smart-home room presets (Sub-step 4.1)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---

#### Sub-step 4.2: Filter Karakteristik Mic Laptop Internal (Highpass 100 Hz) & Injeksi Noise AC/Kipas Terkalibrasi (SNR 16-26 dB)
* **Tanggal Eksekusi**: 28 Oktober 2026
* **Penanggung Jawab**: Yoga (Implementasi DSP, Kalibrasi SNR Numerik, & Verifikasi Respon Frekuensi)
* **Latar Belakang & Masalah**:
  Mikrofon internal bawaan laptop (seperti chip audio Realtek MEMS terintegrasi) memiliki respon akustik yang rentan menangkap getaran mekanis bodi laptop (*chassis rumble*, getaran bearing kipas pendingin, dan getaran meja) di frekuensi sub-100 Hz yang harus diredam secara terstruktur. Selain itu, mikrofon smart home di kamar tidur atau ruang tamu senantiasa terpapar derau lingkungan pendingin ruangan (AC) dan kipas angin (*household acoustic noise*). Model deep learning perlu dilatih dengan variasi noise latar belakang terkalibrasi agar tidak gagal mengenali perintah suara saat AC atau kipas angin menyala.
* **Formulasi Matematika & Rekayasa DSP**:
  1. **Filter Highpass Karakteristik Mic Laptop (100 Hz)**:
     - Menggunakan filter Butterworth Ordo 2 Highpass ($f_c = 100\text{ Hz}$, $f_s = 16.000\text{ Hz}$) diimplementasikan via Second-Order Sections (SOS) untuk kestabilan numerik mutlak.
     - Karakteristik atenuasi empiris terukur:
       - Frekuensi $30\text{ Hz}$: atenuasi $-20,95\text{ dB}$ (getaran mekanis sub-bass terpotong tuntas).
       - Frekuensi $50\text{ Hz}$: atenuasi $-12,31\text{ dB}$ (derau listrik hum jala-jala PLN tereduksi $> 75\%$).
       - Frekuensi $75\text{ Hz}$: atenuasi $-6,19\text{ dB}$.
       - Frekuensi $100\text{ Hz}$: atenuasi $-3,01\text{ dB}$ (tepat pada titik cutoff half-power teoritis $-3\text{ dB}$).
       - Frekuensi vokal inti ($200\text{ Hz} - 4.000\text{ Hz}$): atenuasi $-0,26\text{ dB}$ hingga $-0,00\text{ dB}$ (sinyal vokal ucapan manusia lolos 100% utuh tanpa pewarnaan atau distorsi).
  2. **Profil Derau Rumah Tangga Realistis (*Household Noise Profiles*)**:
     - `ac`: Sirkulasi udara kompresor pendingin ruangan (air rush lowpass 1200 Hz + hum kompresor 50/100 Hz).
     - `fan`: Derau aerodinamis bilah kipas berputar (broadband turbulence 1600 Hz + rotasi motor 120/240 Hz).
     - `ac_fan`: Profil gabungan tipikal ruangan rumah pintar di Indonesia (kamar ber-AC dengan kipas).
     - `ambient_room`: Lantai derau termal kamar tertutup (gentle thermal room floor noise).
  3. **Kalibrasi SNR Presisi Tinggi**:
     - Perhitungan daya sinyal $P_{\text{clean}} = \frac{1}{N}\sum x^2[n]$ dan $P_{\text{noise}} = \frac{1}{N}\sum w^2[n]$.
     - Skala pengali derau:
       $$\text{scale} = \sqrt{\frac{P_{\text{clean}}}{P_{\text{noise}} \cdot 10^{\text{SNR}/10}}}$$
     - Uji sweep kalibrasi SNR $[16, 18, 20, 22, 24, 26]\text{ dB}$:
       - Target $16,0\text{ dB} \rightarrow$ Terukur $16,00\text{ dB}$ (Error: $0,000\text{ dB}$, Peak: $0,8117$).
       - Target $18,0\text{ dB} \rightarrow$ Terukur $18,00\text{ dB}$ (Error: $0,000\text{ dB}$, Peak: $0,8153$).
       - Target $20,0\text{ dB} \rightarrow$ Terukur $20,00\text{ dB}$ (Error: $0,000\text{ dB}$, Peak: $0,8153$).
       - Target $22,0\text{ dB} \rightarrow$ Terukur $22,00\text{ dB}$ (Error: $0,000\text{ dB}$, Peak: $0,8129$).
       - Target $24,0\text{ dB} \rightarrow$ Terukur $24,00\text{ dB}$ (Error: $0,000\text{ dB}$, Peak: $0,7970$).
       - Target $26,0\text{ dB} \rightarrow$ Terukur $26,00\text{ dB}$ (Error: $0,000\text{ dB}$, Peak: $0,8080$).
       - Error rata-rata: $\Delta = 0,000\text{ dB}$ (presisi 100% absolut).
  4. **Kontrol Headroom Anti-Clipping**:
     - Seluruh amplitudo puncak sinyal hasil pencampuran terkendali pada rentang `0.7970 - 0.8153`, 100% patuh di bawah batas aman $\le 0,9500$.
  5. **Kecepatan & Throughput**:
     - Throughput: $32,3\text{ audio clips / detik}$ (single-thread).
     - Rata-rata latensi: $30,93\text{ ms}$ per 3,55 detik audio.
     - Real-Time Factor (RTF): $114,8\times$ lebih cepat dari pemutaran audio real-time.
* **Artefak File yang Dibuat & Dimodifikasi**:
  - `src/svara/data/augment.py`: Penambahan fungsi `apply_laptop_mic_filter`, `generate_ambient_noise`, `inject_household_noise`, `apply_laptop_mic_and_noise`, dan pembaruan `AudioAugmentor` dengan parameter `mic_filter_prob`.
  - `tests/test_augment.py`: Penambahan 5 unit test spesifik (atenuasi 50 vs 1000 Hz, profil noise, kalibrasi SNR presisi, pipeline chaining, dan integrasi augmentor).
  - `scripts/demo_laptop_mic_noise.py`: Tool benchmark respons frekuensi dan kalibrasi sweep SNR.
  - `reports/laptop_mic_noise_benchmark.json`: Data hasil evaluasi akustik dan kecepatan pemrosesan.
* **Hasil Pengujian Test Suite Repositori**:
  - Seluruh **14 unit test augmentasi** di `tests/test_augment.py` lulus $100\%$ dalam $12,55\text{ detik}$.
  - Seluruh **93 unit test** di repositori lulus $100\%$ tanpa ada error (`93 passed, 7 warnings in 107.39s`).
* **Pencapaian Milestone**:
  - Sub-step 4.2 selesai sempurna. Filter mic laptop dan injeksi noise AC/kipas siap dipakai pada augmentasi massal sub-step 4.4.
* **Nomor Commit Git**:
  - `commit 6faaa31` (*feat(augment): implement 100Hz laptop mic filter and calibrated AC/fan noise injection (Sub-step 4.2)*)
* **Status Remote Push**: Terunggah ke `main` di `https://github.com/YogUNI/SVARA.git`.

---
*(Logbook ini akan diperbarui secara otomatis setiap kali sub-step berikutnya selesai dieksekusi)*
