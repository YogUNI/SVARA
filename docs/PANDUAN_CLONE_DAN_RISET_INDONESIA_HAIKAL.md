# PANDUAN LENGKAP CLONING REPO & RISET DATASET BAHASA INDONESIA (HAIKAL)

Halo Haikal! Dokumen ini dibuat khusus untuk memandu kamu secara langkah-demi-langkah:
1. **Cara menarik (*clone*) semua source code SVARA dari GitHub ke laptop kamu**.
2. **Cara riset & menyusun Dataset Perintah Smart Home Bahasa Indonesia (NLP Intent & Slot Filling)** sebelum nanti dilatih ke model Deep Learning Suara.

---

## 🚀 BAGIAN 1: CARA CLONE / TARIK SEMUA CODE DARI GITHUB

Pastikan laptop kamu sudah terinstall **Git** (bisa download di [git-scm.com](https://git-scm.com/) jika belum ada).

### Langkah 1: Buka Terminal / Git Bash / Command Prompt
Buka folder tempat kamu ingin menyimpan proyek ini (misal di folder `Documents` atau `Projects`).

### Langkah 2: Jalankan Perintah Git Clone
Ketik perintah berikut lalu tekan Enter:
```bash
git clone https://github.com/YogUNI/SVARA.git
```

### Langkah 3: Masuk ke Folder Proyek
```bash
cd SVARA
```

### Langkah 4: Jika Yoga Nanti Update Code Terbaru di GitHub, Cara Menariknya Cukup Ketik:
```bash
git pull origin main
```
*Dengan perintah `git pull origin main`, semua update terbaru yang Yoga buat akan langsung otomatis masuk ke laptop kamu tanpa menimpa file yang tidak perlu!*

---

## 🧠 BAGIAN 2: RISET DATASET BAHASA INDONESIA (NLP KE DEEP LEARNING)

### Mengapa Kita Butuh Riset Dataset Bahasa Indonesia Ini?
Sistem AI SVARA saat ini dilatih menggunakan dataset standar internasional **Fluent Speech Commands (FSC)** yang berbasis perintah suara **Bahasa Inggris** (*"Turn on the lights in the kitchen"*, *"Turn down the volume"*, dll).

Tugas riset kamu yang paling krusial untuk laporan dan pengembangan berikutnya adalah:
👉 **Membangun Fondasi NLP (Natural Language Processing) untuk Bahasa Indonesia**, yaitu bagaimana kebiasaan orang Indonesia sehari-hari saat memerintahkan smart home.

Di dunia Spoken Language Understanding (SLU) / NLP, setiap perintah suara dipetakan ke dalam 3 slot semantik:
- **`action`** (Kata kerja tindakan: *menyalakan, mematikan, menaikkan, menurunkan, mengambilkan, ganti bahasa*)
- **`object`** (Perangkat / benda yang dituju: *lampu, pemanas/suhu, musik, volume, koran, jus, sepatu*)
- **`location`** (Lokasi ruangan: *dapur, kamar tidur, kamar mandi, atau 'none' jika umum*)

---

## 📝 TUGAS RISET HAIKAL: MEMBUAT KORPUS FRASA BAHASA INDONESIA

Tugas kamu adalah **meriset dan mengumpulkan variasi frasa perintah orang Indonesia** untuk ke-31 intent berikut, mulai dari bahasa baku, bahasa santai sehari-hari, hingga bahasa gaul (*slang*):

### Contoh Pemetaan NLP (Frasa Asli Inggris vs Variasi Bahasa Indonesia):

| Intent ID | Intent Inggris (FSC) | Slot Semantik (Action, Object, Location) | Contoh Variasi Frasa Orang Indonesia (Baku, Santai, Gaul) |
|---|---|---|---|
| `00` | "Turn on the lamp" | `activate` + `lamp` + `none` | 1. "Nyalakan lampu meja"<br>2. "Hidupkan lampunya"<br>3. "Tolong nyalain lampu dong" |
| `01` | "Turn on the lights in the bedroom" | `activate` + `lights` + `bedroom` | 1. "Nyalakan lampu kamar tidur"<br>2. "Lampu kamar nyalain"<br>3. "Hidupkan lampu kamar" |
| `02` | "Turn on the kitchen lights" | `activate` + `lights` + `kitchen` | 1. "Nyalakan lampu dapur"<br>2. "Tolong hidupkan lampu di dapur"<br>3. "Nyalain lampu dapur dong" |
| `04` | "Turn on the washroom lights" | `activate` + `lights` + `washroom` | 1. "Nyalakan lampu kamar mandi"<br>2. "Hidupkan lampu toilet"<br>3. "Nyalain lampu WC" |
| `05` | "Resume" | `activate` + `music` + `none` | 1. "Lanjutkan musik"<br>2. "Putar lagi lagunya"<br>3. "Nyalain lagi musiknya" |
| `15` | "Turn off the lamp" | `deactivate` + `lamp` + `none` | 1. "Matikan lampu"<br>2. "Padamkan lampu"<br>3. "Matiin lampunya" |
| `16` | "Turn the bedroom lights off" | `deactivate` + `lights` + `bedroom` | 1. "Matikan lampu kamar"<br>2. "Padamkan lampu kamar tidur"<br>3. "Matiin lampu kamar" |
| `21` | "Turn off the music" | `deactivate` + `music` + `none` | 1. "Matikan musiknya"<br>2. "Stop lagunya"<br>3. "Matiin lagu dong" |
| `23` | "Turn the kitchen temperature down" | `decrease` + `heat` + `kitchen` | 1. "Turunkan suhu di dapur"<br>2. "Dinginkan dapur"<br>3. "Kecilkan pemanas dapur" |
| `25` | "Turn the volume down" | `decrease` + `volume` + `none` | 1. "Kecilkan volume"<br>2. "Turunkan suaranya"<br>3. "Kecilin suara dong berisik" |
| `30` | "Volume up" | `increase` + `volume` + `none` | 1. "Besarkan volume"<br>2. "Keraskan suaranya"<br>3. "Naikkan volume lagunya" |

---

## 🎯 TARGET OUTPUT YANG HARUS DIKERJAKAN HAIKAL:

1. **Buat File Spreadsheet (Excel / Google Sheets) atau Markdown**:
   - Berisi 31 tabel intent yang masing-masing memiliki **minimal 5 variasi kalimat bahasa Indonesia** yang biasa diucapkan orang kita sehari-hari.
   - Kolom yang dibutuhkan: `intent_id`, `english_transcript`, `indonesian_phrase`, `action`, `object`, `location`.
2. **Kumpulkan Contoh Suara Sampel Bahasa Indonesia**:
   - Minta 3–5 teman membaca kalimat bahasa Indonesia tersebut sebagai rekaman pilot (*pilot test recording*).
3. **Analisis Karakteristik Linguistik Indonesia (Untuk Bab 3 Laporan)**:
   - Catat pola bahasa: Apakah orang Indonesia lebih suka kata *"nyalain"* dibanding *"hidupkan"*?
   - Apakah kata lokasi diletakkan di depan (*"Kamar mandi tolong matiin lampunya"*) atau di belakang (*"Matiin lampu kamar mandi"*).
   - Analisis ini adalah poin plus sangat besar di mata dosen penguji karena membuktikan riset linguistik lokal yang matang!

---

Semua file panduan ini sudah tersimpan rapi di dalam repository SVARA di [`docs/PANDUAN_CLONE_DAN_RISET_INDONESIA_HAIKAL.md`](file:///c:/Users/Axioo%20Pongo/SVARA/docs/PANDUAN_CLONE_DAN_RISET_INDONESIA_HAIKAL.md).
Jika Haikal butuh bantuan atau ingin mendiskusikan daftar kalimatnya, langsung koordinasikan dengan Yoga!
