# DOKUMEN SPESIFIKASI & ROADMAP: PEMATANGAN REAL-WORLD DEEP LEARNING SLU SVARA

Dokumen Resmi: `docs/14_ROADMAP_PEMATANGAN_DEEP_LEARNING_REAL_WORLD.md`  
Status: **DRAFT RESMI EKSEKUSI PEMATANGAN SISTEM**  
Tim Pengembang: **Yoga (Lead ML & Systems Engineer)** & **Haikal (Acoustic Data & Field QA)**  
Afiliasi: Universitas Mercu Buana — Tugas Akhir Deep Learning (Spoken Language Understanding)  
Repositori: `YogUNI/SVARA` (Branch: `main`)

---

## 1. LATAR BELAKANG & ANALISIS KRITIS: THE SYNTHETIC PROSODY TRAP

### 1.1. Temuan Pengujian Lapangan (Empirical Observations)
Saat sistem SVARA diuji secara langsung menggunakan mikrofon fisik di browser visualizer 3D/2D, ditemukan bahwa:
> *"Nada, Intonasi, dan Kefasihan Bicara sangat diperhatikan di sini, jadi harus benar-benar mendekati robot/TTS baru bisa bekerja secara akurat."*

Ketika pengguna berbicara dengan nada bicara santai, intonasi bertanya (*rising pitch* di akhir kata), nada rendah/tinggi, atau tempo cepat, model sering kali:
1. Mengalami penurunan drastis skor keyakinan (*confidence score*) hingga di bawah ambang batas penolakan ($< 0.65$).
2. Salah mengklasifikasikan slot `action` atau `object` (misal: "naikkan volume" tertukar dengan "naikkan suhu/heat").
3. Kehilangan prediksi slot `location` (terprediksi `none`) pada kalimat perintah yang panjang.

---

### 1.2. Mengapa Akurasi di Notebook 99.5%, Tetapi di Mikrofon Fisik Terasa Kaku?
Secara ilmiah Deep Learning, fenomena ini dikenal sebagai **Acoustic Domain Shift & Synthetic Prosodic Shortcut**:

1. **Jalan Pintas Nada Datar (The Flat $F_0$ Shortcut)**:
   - Data pelatihan bahasa Indonesia SVARA digenerate menggunakan Microsoft Edge-TTS (`id-ID-ArdiNeural` dan `id-ID-GadisNeural`).
   - Suara sintetis neural TTS memiliki frekuensi dasar vokal (*Fundamental Frequency* / $F_0$) yang **sangat seragam dan kaku**:
     - Ardi: $F_0 \approx 115 - 125\text{ Hz}$ (pria datar/robotik).
     - Gadis: $F_0 \approx 210 - 230\text{ Hz}$ (wanita datar/robotik).
   - Seluruh fonem diucapkan dengan durasi suku kata yang matematis, tanpa variasi biologis, tanpa desah napas, tanpa *vocal fry*, dan tanpa fluktuasi intonasi emosional.
2. **Overfitting pada Matriks Self-Attention Wav2Vec 2.0**:
   - Model Wav2Vec 2.0 memiliki 7 lapisan konvolusi temporal (CNN) dan 12 lapisan Transformer Multi-Head Self-Attention.
   - Karena dilatih berulang kali pada dataset sintetis Ardi dan Gadis, lapisan *Self-Attention* mengambil jalan pintas: **model menghafal harmonik nada Ardi dan Gadis**, bukan mempelajari representasi variasi fonetik murni (*phonetic invariance*).
   - Dataset uji (*test set*) di notebook juga berupa suara sintetis TTS (meskipun transkripnya unseen). Akibatnya, akurasi uji tampak sangat tinggi (**99.5%**), namun rapuh ketika berhadapan dengan Out-of-Distribution (OOD) vokal biologis manusia.
3. **Fluktuasi Intonasi Suara Manusia Asli**:
   - Suara manusia asli (Yoga, Haikal, atau pengguna umum) memiliki dinamika nada yang berayun bebas:
     - Intonasi bertanya/santai: nada naik di ujung kalimat (*terminal rising contour* $\Delta F_0 \approx +40\text{ s.d } +100\text{ Hz}$).
     - Intonasi perintah tegas: nada turun tajam (*falling contour*).
     - Rentang nada manusia sangat variatif (90 Hz s.d 280 Hz).
   - Begitu manusia berbicara dengan intonasi dinamis, matriks *Self-Attention* Wav2Vec 2.0 menjadi kabur/tersebar (*diffuse*), distribusi probabilitas softmax menjadi datar (entropi tinggi), dan perintah gagal dieksekusi.

---

### 1.3. Faktor Teknis Sinyal Audio: Pemotongan Paksa & VAD Clipping

Selain karakteristik model AI, ditemukan dua faktor peredam sinyal non-model di kode SVARA:
1. **Frontend Duration Cap 3.5 Detik ([web/src/App.tsx#L547](file:///c:/Users/Axioo%20Pongo/SVARA/web/src/App.tsx#L547))**:
   - Frontend memotong paksa audio pada `Math.floor(sr * 3.5)` (hanya 3.5 detik), padahal backend mendukung hingga 5.0 detik.
   - Ucapan manusia santai seperti *"Tolong nyalakan lampu yang ada di dapur"* rata-rata berdurasi 3.8 – 4.2 detik.
   - Akibatnya: Kata **"dapur"** terpotong habis sebelum sempat dikirim ke backend! Inilah alasan utama slot `location` sering menjadi `none`.
2. **Terminal Syllable Clipping pada Dual-Threshold VAD ([src/svara/data/audio.py](file:///c:/Users/Axioo%20Pongo/SVARA/src/svara/data/audio.py))**:
   - Di ujung kalimat, energi vokal manusia secara alami menurun (*trailing energy decay*).
   - Threshold `DEFAULT_VAD_STOP_DBFS = -42.0 dBFS` dan buffer padding `150ms` terlalu sempit, sehingga suku kata akhiran (*-kan*, *-in*, *-pur*, *-tur*, *-di*) berisiko terpangkas.

---

## 2. MATRIKS TARGET 5 DOMAIN SMART HOME (FULL COVERAGE)

Pematangan sistem wajib mencakup seluruh 31 intent fungsional yang tersebar merata di 5 domain:

| No | Domain Smart Home | Slot Intent (`action`, `object`, `location`) | Variasi Frasa Baku & Frasa Santai (Colloquial / Slang) |
|---|---|---|---|
| 1 | **Pencahayaan (Lighting)** | `activate`/`deactivate` + `lights`/`lamp` + `kitchen`/`bedroom`/`washroom`/`none` | *"Nyalain lampu dapur"*, *"Padamin lampu kamar"*, *"Matiin semua lampu dong"*, *"Idupin lampu toilet"*, *"Gelap banget tolong terangin"*, *"Lampu ruang tengah tolong dimatikan"* |
| 2 | **Audio & Media (Media/Music)** | `activate`/`deactivate`/`increase`/`decrease` + `music`/`volume` + `none` | *"Puter musik lagi"*, *"Lanjutin lagunya dong"*, *"Stop musiknya"*, *"Matiin lagu"*, *"Gedein volumenya dikit"*, *"Kencengin suaranya"*, *"Kecilin dong suaranya"*, *"Pelanin volumenya"* |
| 3 | **Suhu & Iklim (Climate/Heat)** | `increase`/`decrease` + `heat` + `kitchen`/`bedroom`/`washroom`/`none` | *"Dinginin kamar tidur"*, *"Kamar gerah banget tolong turunin suhu"*, *"Panasin ruangan"*, *"Naikin suhu dapur"*, *"Kecilkan pendingin ruangan"*, *"Sejukin kamar mandi"* |
| 4 | **Perabot & Robotik (Appliances)** | `activate`/`deactivate`/`bring` + `juice`/`newspaper`/`shoes`/`socks` + `none` | *"Bawain jus jeruk"*, *"Ambilin sepatu saya"*, *"Bawakan koran pagi"*, *"Ambilin kaus kaki"*, *"Nyalakan pembuat jus"*, *"Siapkan sepatu"* |
| 5 | **Pengaturan Sistem (System/Lang)** | `change language` + `English`/`German`/`Korean`/`Chinese`/`none` + `none` | *"Ganti bahasa ke Inggris"*, *"Ubah ke bahasa Jerman"*, *"Setel bahasa Korea dong"*, *"Ganti bahasa sistem"* |

---

## 3. MASTERPLAN 4 FASE BARU: THE REAL-WORLD HARDENING ROADMAP

Untuk mematangkan sistem secara tuntas tanpa melanggar prinsip ilmiah dan zero-fabrication, roadmap diperluas dengan **FASE 11 hingga FASE 14**:

```
├── FASE 11: AUDIT & RELAKSASI SINYAL AUDIO LIVE MIC (FRONTEND & VAD BUFFER)
│   ├── [x] Sub-step 11.1: Perbaikan Buffer Audio & Truncation Cap Frontend (3.5s -> 5.0s di `web/src/App.tsx`) [SELESAI]
│   ├── [ ] Sub-step 11.2: Relaksasi Dual-Threshold Energy VAD & Safety Padding di `src/svara/data/audio.py`
│   └── [ ] Sub-step 11.3: Verifikasi Paritas Live Streaming Buffer & Anti-Clipping Unit Test
│
├── FASE 12: DYNAMIC PITCH & PROSODY AUGMENTATION (GPU ACCELERATED VIA CUDA)
│   ├── [ ] Sub-step 12.1: Implementasi Modul Torchaudio PitchShift GPU di `src/svara/data/augment.py`
│   ├── [ ] Sub-step 12.2: Eksekusi Batch Variasi Nada (±1 s.d ±4 Semitones) Penetrasi Spektrum Vokal 90-280 Hz
│   └── [ ] Sub-step 12.3: Audit Integritas Akustik & Formant Zero-Clipping Guard
│
├── FASE 13: ONLINE SPECAUGMENT & FEW-SHOT HUMAN ACOUSTIC CALIBRATION
│   ├── [ ] Sub-step 13.1: Implementasi Online Frequency & Time Masking (SpecAugment) pada DataLoader
│   ├── [ ] Sub-step 13.2: Pengambilan & Injeksi Rekaman Suara Asli (Haikal & Yoga) sebagai Acoustic Anchor
│   └── [ ] Sub-step 13.3: Perluasan Frasa Gaul/Colloquial Sehari-hari untuk Seluruh 5 Domain Smart Home
│
└── FASE 14: RETRAINING BILINGUAL ROBUST PADA GPU RTX 4060, EXPORT ONNX & FIELD-TEST
    ├── [ ] Sub-step 14.1: Retraining Fine-Tuning FP16 AMP Model Hardened pada GPU RTX 4060
    ├── [ ] Sub-step 14.2: Export Universal ONNX FP32 & Dynamic INT8 Quantization (Parity Validated)
    └── [ ] Sub-step 14.3: Validasi Lapangan Uji Suara Mikrofon Fisik di Visualizer 3D Smart Home
```

---

## 4. RINCIAN OPERASIONAL & ACCEPTANCE CRITERIA PER SUB-STEP

### FASE 11: AUDIT & RELAKSASI SINYAL AUDIO LIVE MIC (FRONTEND & VAD BUFFER)
*Penanggung Jawab: Yoga (Implementasi) & Haikal (Verifikasi Rekaman)*

* **Sub-step 11.1: Perbaikan Buffer Audio & Truncation Cap Frontend** [SELESAI - web/src/App.tsx]
  - **Lokasi Kode**: `web/src/App.tsx`
  - **Tindakan**:
    1. Ubah batasan durasi sampel: dari `const maxSamples = Math.floor(sr * 3.5);` menjadi `Math.floor(sr * 5.0);` (penuh 5.0 detik).
    2. Perlebar bantalan penutup audio (*post-padding cushion*): dari `Math.floor(sr * 0.2)` (200ms) menjadi `Math.floor(sr * 0.4)` (400ms).
    3. Turunkan ambang keheningan (*silence threshold*): dari `0.015` ke `0.008` agar desah vokal lembut di akhir kalimat tidak terpotong.
  - **Acceptance Criteria**: Pengucapan kalimat 4.5 detik *"Halo SVARA tolong matikan semua lampu di kamar mandi"* terkirim 100% lengkap tanpa kehilangan suku kata terakhir.

* **Sub-step 11.2: Relaksasi Dual-Threshold Energy VAD & Safety Padding Backend**
  - **Lokasi Kode**: `src/svara/data/audio.py`
  - **Tindakan**:
    1. Ubah `DEFAULT_VAD_STOP_DBFS` dari `-42.0 dBFS` menjadi `-48.0 dBFS`.
    2. Ubah `DEFAULT_VAD_PAD_MS` dari `150.0 ms` menjadi `250.0 ms`.
  - **Acceptance Criteria**: VAD backend tidak memotong fonem bersuara lembut (*soft voiced phonemes*) seperti konsonan nasal /m/, /n/, /ng/ dan plosif lemah di akhir kata.

* **Sub-step 11.3: Verifikasi Paritas Live Streaming Buffer & Anti-Clipping Unit Test**
  - **Tindakan**: Jalankan unit test `pytest tests/test_audio.py` dan buat tes baru yang mensimulasikan audio bervokal lembut di akhir kalimat.
  - **Acceptance Criteria**: Semua unit test audio lolos 100%.

---

### FASE 12: DYNAMIC PITCH & PROSODY AUGMENTATION (GPU ACCELERATED VIA CUDA)
*Penanggung Jawab: Yoga*

* **Sub-step 12.1: Implementasi Modul Torchaudio PitchShift GPU**
  - **Lokasi Kode**: `src/svara/data/augment.py`
  - **Algoritma**: Menggunakan `torchaudio.transforms.PitchShift(sample_rate=16000, n_steps=...)` berbasis algoritma *phase vocoder* yang diakselerasi secara native pada CUDA RTX 4060.
  - **Acceptance Criteria**: Fungsi mampu melakukan *pitch shift* 1 detik audio dalam waktu $< 3\text{ ms}$ di GPU RTX 4060.

* **Sub-step 12.2: Eksekusi Batch Variasi Nada ($\pm 1$ s.d $\pm 4$ Semitones)**
  - **Lokasi Script**: `scripts/make_pitch_augmented_dataset.py`
  - **Skema Pergeseran Nada**:
    - Suara Pria (`spk_ardi`, asal ~120 Hz): digeser $+2, +3, +4$ semitone (menghasilkan vokal pria muda/tenor ~150-180 Hz) dan $-2, -3$ semitone (vokal bariton berat ~100 Hz).
    - Suara Wanita (`spk_gadis`, asal ~220 Hz): digeser $-2, -3, -4$ semitone (vokal wanita dewasa/alto ~170-190 Hz) dan $+2, +3$ semitone (vokal tinggi/soprano ~260 Hz).
  - **Acceptance Criteria**: Menghasilkan sebaran frekuensi vokal kontinu dari 90 Hz hingga 280 Hz, melenyapkan dependensi model pada satu tinggi nada tertentu.

* **Sub-step 12.3: Audit Integritas Akustik & Formant Zero-Clipping Guard**
  - **Tindakan**: Verifikasi audio hasil modulasi nada: tidak mengalami digital clipping (Peak $\le 0.95$), rasio sampling 16 kHz terjaga, dan formasi formant fonetik vokal tetap dapat dipahami (*intelligible*).
  - **Acceptance Criteria**: Ekspor laporan `reports/pitch_augmentation_audit.json`.

---

### FASE 13: ONLINE SPECAUGMENT & FEW-SHOT HUMAN ACOUSTIC CALIBRATION
*Penanggung Jawab: Yoga (SpecAugment & Pipeline) & Haikal (Perekaman Suara Asli)*

* **Sub-step 13.1: Implementasi Online Frequency & Time Masking (SpecAugment)**
  - **Lokasi Kode**: `src/svara/models/wav2vec_slu.py` / `src/svara/data/dataset.py`
  - **Mekanisme**:
    - **Frequency Masking**: Menutup pita frekuensi acak ($F \in [10, 30]$ kanal frekuensi). Saat jalur harmonik nada tertentu hilang, model dipaksa mengidentifikasi fonem berdasarkan konteks formant lainnya.
    - **Time Masking**: Menutup blok waktu acak ($T \in [10, 40]$ bingkai waktu), melatih ketahanan model terhadap sengauan atau jeda mikrosekon saat manusia berbicara.
  - **Acceptance Criteria**: SpecAugment aktif secara dinamis hanya saat mode `is_training=True` dan non-aktif saat evaluasi/inference.

* **Sub-step 13.2: Pengambilan & Injeksi Rekaman Suara Asli (Acoustic Calibration Anchor)**
  - **Lokasi Data**: `data/own_recordings/`
  - **Peran Haikal**:
    1. Mengikuti panduan [docs/own_recordings/phrase_sheet.md](file:///c:/Users/Axioo%20Pongo/SVARA/data/own_recordings/phrase_sheet.md).
    2. Merekam 31 kalimat perintah dalam 2 kondisi (Kondisi Tenang / `quiet` dan Kondisi Berisik / `noisy`).
    3. Haikal dan Yoga menyumbang total 62 s.d 124 klip audio WAV 16 kHz Mono.
  - **Injeksi Data**: Klip suara asli manusia dimasukkan ke dalam split pelatihan sebagai *Acoustic Anchor*. Keberadaan suara pita suara asli bertindak sebagai jembatan pembiasaan encoder terhadap getaran biologis manusia.
  - **Acceptance Criteria**: Manifest `data/own_recordings/metadata.csv` terisi lengkap dan terverifikasi zero-corruption.

* **Sub-step 13.3: Perluasan Frasa Gaul/Colloquial Sehari-hari untuk Seluruh 5 Domain**
  - **Lokasi Script**: `scripts/expand_colloquial_smart_home.py`
  - **Cakupan**: Memastikan variasi kata kerja santai (*"idupin"*, *"nyalain"*, *"padamin"*, *"matiin"*, *"kencengin"*, *"gedein"*, *"pelanin"*, *"dinginin"*, *"panasin"*) tercakup seimbang di seluruh 31 intent.
  - **Acceptance Criteria**: Menambah setidaknya 1.500 frasa colloquial baru yang tervalidasi secara semantik.

---

### FASE 14: RETRAINING BILINGUAL ROBUST PADA GPU RTX 4060, EXPORT ONNX & FIELD-TEST
*Penanggung Jawab: Yoga (Training & Export) & Haikal (QA Live Mic)*

* **Sub-step 14.1: Retraining Fine-Tuning FP16 AMP Model Hardened pada GPU RTX 4060**
  - **Script**: `scripts/train_hardened_bilingual_w2v2.py`
  - **Spesifikasi**:
    - Backbone: `facebook/wav2vec2-base` (Feature Encoder CNN dibekukan, 12 Transformer layer dilatih).
    - Regularisasi: Dropout 0.1, Label Smoothing $\epsilon = 0.05$, Weight Decay 0.01, SpecAugment aktif.
    - Presisi: Mixed Precision PyTorch AMP (FP16) di GPU RTX 4060 Laptop (8GB VRAM).
  - **Acceptance Criteria**: Model mencapai konvergensi dengan validasi Exact-Match $> 98\%$ dan loss stabil tanpa divergensi.

* **Sub-step 14.2: Export Universal ONNX FP32 & Dynamic INT8 Quantization**
  - **Output File**:
    - `models/svara_bilingual_hardened.onnx` (FP32 universal graph, input dinamis).
    - `models/svara_bilingual_hardened_int8.onnx` (INT8 dynamic quantization, ukuran $\approx 90\text{ MB}$).
    - Metadata konfigurasi `models/svara_bilingual_hardened_int8.meta.json`.
  - **Acceptance Criteria**: Paritas numerik antara PyTorch dan ONNX Runtime terverifikasi (Deviasi maksimum output logit $< 10^{-4}$).

* **Sub-step 14.3: Validasi Lapangan Uji Suara Mikrofon Fisik di Visualizer 3D Smart Home**
  - **Metode Pengujian**:
    1. Haikal dan Yoga melakukan 50 pengujian suara langsung menggunakan mikrofon laptop.
    2. Skenario uji mencakup 5 gaya bicara:
       - Nada datar biasa.
       - Nada santai / gaul (*colloquial*).
       - Nada bertanya (*rising pitch* di akhir).
       - Tempo cepat (*fast speech*).
       - Tempo lambat / ragu-ragu (*hesitant speech*).
  - **Acceptance Criteria**: Akurasi keberhasilan eksekusi smart home (*Physical Mic Exact Match Rate*) meningkat signifikan dari kondisi awal, dan perintah tidak lagi ditolak hanya karena intonasi naik-turun.
  - **Laporan Resmi**: Ekspor metrik hasil pengujian lapangan ke `reports/real_world_field_test_results.json` dan dokumentasikan di laporan teknis.

---

## 5. DISTRIBUSI PERAN & TANGGUNG JAWAB TIM (YOGA & HAIKAL)

| Anggota Tim | Tanggung Jawab Utama | Sub-step yang Dikerjakan |
|---|---|---|
| **Yoga (Lead ML & Systems Engineer)** | - Rekayasa kode Python, PyTorch, dan Torchaudio.<br>- Implementasi perbaikan frontend (`web/src/App.tsx`) dan backend (`audio.py`).<br>- Pengembangan modul GPU PitchShift dan pipeline SpecAugment.<br>- Eksekusi retraining model pada GPU RTX 4060.<br>- Ekspor ONNX INT8 dan integrasi ke backend FastAPI. | Sub-step 11.1, 11.2, 11.3, 12.1, 12.2, 12.3, 13.1, 13.3, 14.1, 14.2 |
| **Haikal (Acoustic Data & Field QA)** | - Perekaman dataset suara asli manusia (`data/own_recordings/`) sesuai lembar panduan.<br>- QA mikrofon fisik: menguji 50 perintah suara di web visualizer 3D/2D.<br>- Pencatatan log kegagalan intonasi dan kecepatan bicara.<br>- Penyusunan dokumentasi pengujian lapangan dan bab laporan terkait. | Sub-step 11.3 (Review), 13.2 (Perekaman Utama), 14.3 (Pengujian QA Lapangan) |

---

## 6. WORKFLOW & COMMIT POLICY

Sesuai aturan baku proyek di `AGENTS.md`:
1. **Eksekusi Bertahap (One Sub-step at a Time)**: Setiap sub-step diselesaikan secara tuntas dan diuji sebelum melangkah ke sub-step berikutnya.
2. **Auto-Commit & Auto-Push**: Setiap kali menyelesaikan 1 sub-step, kode dan artefak **wajib langsung di-commit dan di-push ke branch `main` GitHub milik Yoga (`YogUNI/SVARA`)**.
3. **Zero Fabrication**: Tidak ada rekayasa hasil atau klaim metrik palsu. Semua data pengujian mikrofon fisik harus berasal dari eksekusi nyata yang tercatat di `reports/`.

