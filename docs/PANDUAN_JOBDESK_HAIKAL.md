# PANDUAN LENGKAP & JOBDESK HAIKAL (PROYEK DEEP LEARNING — SVARA)

Halo Haikal! Dokumen ini adalah panduan lengkap dan praktis untuk tugas kamu di proyek tugas akhir **SVARA (Smart Voice Assistant for Residential Automation)** bersama Yoga.

Kamu tidak perlu pusing dengan kode machine learning / training model, karena bagian core AI (wav2vec2), evaluasi GPU, dan backend/frontend sudah disiapkan oleh Yoga. Fokus kamu adalah **Pengambilan Data Riil (Phase P5)**, **Penyusunan Laporan Akademik Bab 1–3 (Phase P10)**, dan **Demo Video & QA (Phase P8–P9)**.

---

## 📌 TUGAS 1: Perekaman Suara Aksen Indonesia (Own Recording Dataset — Phase P5)

### Mengapa Tugas Ini Sangat Penting?
Model AI kita dilatih menggunakan data suara *native speaker* (orang bule Amerika). Pertanyaan kritis dari dosen saat sidang adalah:  
👉 *"Apakah model kalian bisa mengenali suara orang Indonesia yang ngomong bahasa Inggris dengan aksen lokal?"*  
Data rekaman dari kamu inilah yang membuktikan ke dosen bahwa sistem kita tahan (*robust*) terhadap aksen orang Indonesia!

### Rincian Target Rekaman:
1. **Target Responden**: Minimal **12 orang** (target ideal 15 orang).  
   - Boleh teman kampus, teman kos, keluarga, laki-laki & perempuan.
   - Berikan kode ID anonim (misal: `own_s01`, `own_s02`, dst.).
2. **Apa yang Dibaca?**:
   - Setiap orang membaca **31 kalimat perintah bahasa Inggris** yang ada di tabel lembar frasa di bawah.
3. **2 Kondisi Perekaman**:
   - **Kondisi Tenang (`q` / quiet)**: Direkam di dalam kamar/ruangan hening.
   - **Kondisi Berisik (`n` / noisy)**: Direkam di ruangan dengan suara latar (misal ada TV menyala, kipas angin kencang, atau suara dapur).
4. **Alat Rekam**:
   - Cukup gunakan aplikasi perekam suara bawaan HP (Voice Recorder).
   - Simpan audio dan kirimkan ke Yoga (nanti Yoga yang menjalankan converter otomatis ke format 16 kHz WAV mono).

---

## 📋 LEMBAR 31 KALIMAT PERINTAH (Berikan Teks Ini ke Responden):

Minta responden membaca kalimat bertuliskan tebal berikut secara wajar/natural:

1. **"Turn on the lamp"**
2. **"Turn on the lights in the bedroom"**
3. **"Turn on the kitchen lights"**
4. **"Turn the lights on"**
5. **"Turn on the washroom lights"**
6. **"Resume"**
7. **"Bring some juice"**
8. **"Get me the newspaper"**
9. **"Bring me my shoes"**
10. **"Bring socks"**
11. **"Set language to Chinese"**
12. **"Set my phone's language to English"**
13. **"I need to practice my German. Switch the language"**
14. **"Set language to Korean"**
15. **"Change language"**
16. **"Turn off the lamp"**
17. **"Turn the bedroom lights off"**
18. **"Turn the lights off in the kitchen"**
19. **"Switch off the lights"**
20. **"Switch off the washroom lights"**
21. **"Turn off the music"**
22. **"Turn down the bedroom heat"**
23. **"Turn the kitchen temperature down"**
24. **"Turn down the temperature"**
25. **"Turn the temperature down in the washroom"**
26. **"Turn the volume down"**
27. **"Turn up the temperature in the bedroom"**
28. **"Turn the kitchen temperature up"**
29. **"Turn up the temperature"**
30. **"Turn up the washroom temperature"**
31. **"Volume up"**

---

## 📌 TUGAS 2: Perekaman Suara Bising Rumah Tangga (Noise Clips)
Rekam 10 sampai 15 klip audio berdurasi masing-masing **10–15 detik** yang hanya berisi suara bising lingkungan rumah tangga (tanpa orang berbicara):
- Suara kipas angin berputar kencang.
- Suara air mengalir di wastafel / kamar mandi.
- Suara TV atau siaran berita di latar belakang.
- Suara kendaraan lewat di depan rumah atau suara hujan.
*(Klip ini akan digunakan Yoga untuk pengujian ketahanan model di Bab 6).*

---

## 📌 TUGAS 3: Penulisan Laporan Tugas Akhir (Bab 1, Bab 2, Bab 3)

Laporan ditulis dalam **Bahasa Indonesia** sesuai pedoman Universitas Mercu Buana:

### 1. Bab 1 — Pendahuluan
- **1.1 Latar Belakang**: Pesatnya adopsi Smart Home berbasis perintah suara (Voice Assistant). Namun sistem konvensional menggunakan 2 tahap (ASR mentranskripsi teks dulu, baru teks diproses NLP), yang rentan kesalahan ejaan (*cascading error*) dan boros komputasi. SVARA mengusulkan End-to-End Spoken Language Understanding (SLU).
- **1.2 Rumusan Masalah**: Bagaimana membangun model SLU berbasis wav2vec2 yang mampu mengenali niat perintah smart home langsung dari audio gelombang mentah, serta seberapa tangguh model terhadap aksen Indonesia dan kebisingan?
- **1.3 Tujuan Penelitian**: Melatih, mengevaluasi, mengekspor model ONNX, dan mendeploy web demo voice assistant.
- **1.4 Batasan Masalah**: Dataset Fluent Speech Commands (31 intents, 3 slot semantik: action, object, location).

### 2. Bab 2 — Tinjauan Pustaka & Dasar Teori
- **2.1 Smart Home Voice Control**: Konsep otomatisasi rumah cerdas.
- **2.2 Spoken Language Understanding (SLU)**: Perbedaan sistem *Pipeline* (ASR + NLU) vs *End-to-End SLU* (audio langsung ke intent).
- **2.3 Arsitektur Transformer & wav2vec 2.0**: Penjelasan model *self-supervised speech representation* (Lugosch et al., 2019 dan Baevski et al., 2020).
- **2.4 Daftar Referensi IEEE**: Gunakan format sitasi standar IEEE (contoh: `[1]`, `[2]`).

### 3. Bab 3 — Metodologi & Data
- **3.1 Dataset Fluent Speech Commands (FSC)**: Karakteristik data audio (16 kHz, 97 pembicara, 248 variasi frasa).
- **3.2 Skema Pembagian Data (Splits)**:
  - Split A: Pembagian standar (train, valid, test).
  - Split B1: Pembagian pembicara yang belum pernah didengar (*unseen speaker generalization*).
- **3.3 Pengumpulan Dataset Rekaman Mandiri (Own Set)**: Jelaskan prosedur yang Haikal lakukan (12 responden, 31 kalimat, kondisi tenang vs bising).

---

## 📌 TUGAS 4: Pembuatan Video Demo & QA (Pengujian Web)
1. **Pengujian Web (QA)**:
   - Coba jalankan web SVARA di browser.
   - Coba ucapkan perintah suara menggunakan mic atau unggah audio.
   - Catat jika ada kata-kata yang salah dikenali atau tombol yang lambat.
2. **Video Demo (Durasi ~3–5 Menit)**:
   - Rekam layar (*screen record*) saat web demo berjalan.
   - Struktur video:
     1. Pembukaan: Pengenalan tim dan judul proyek SVARA.
     2. Penjelasan Singkat: Arsitektur Transformer End-to-End SLU tanpa ASR perantara.
     3. Demonstrasi Langsung: Bicara lewat mic ("Turn on lights in kitchen", "Turn up the heat") dan perlihatkan denah rumah menyala responsif.
     4. Bukti Evaluasi: Tampilkan tab Lab (Akurasi 99.60%).
     5. Penutup.

---

Semangat Haikal! Semua sistem model AI dan web backend/frontend sudah siap di laptop Yoga. Jika ada pertanyaan teknis, diskusikan langsung dengan Yoga.
