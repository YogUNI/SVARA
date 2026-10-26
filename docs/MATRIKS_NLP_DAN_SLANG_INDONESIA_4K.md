# Spesifikasi Detail Langkah 1: Rekayasa Linguistik & Matriks NLP Bahasa Indonesia (Target 4.000 - 5.000 Kalimat)

Dokumen ini mendefinisikan taksonomi linguistik, leksikon komprehensif, slang/bahasa gaul, partikel tutur spontan, dan permutasi sintaksis kalimat terbalik untuk menghasilkan **4.000 hingga 5.000 variasi kalimat bahasa Indonesia sehari-hari** pada ke-31 intent smart home sistem SVARA.

---

## 1. Taksonomi Matriks Leksikon & Slang Bahasa Indonesia

### A. Matriks Aksi (Action Matrix)
| Aksi Formal | Slang, Kolokial & Gaul | Imbuhan & Partikel |
| :--- | :--- | :--- |
| **`activate`** (Menyalakan) | nyalain, hidupin, idupin, pasang, setel, putar, aktifin, start, on-in, buka, tolong nyalakan, nyalain dong, hidupkan, tolong hidupin | tolong, coba, dong, deh, lah, sih, ya, kan, punten, minta tolong |
| **`deactivate`** (Mematikan) | matiin, padamkan, padamin, matikan, tutup, stop, off-in, cut, hentikan, tolong matiin, matiin dong, jangan nyala, matiin lah | tolong, coba, dong, deh, lah, ya, kan, plis, buruan |
| **`increase`** (Menaikkan) | naikin, gedein, tambahin, kuatin, perbesar, besarin, tinggiin, angkat, panasin, hangatkan, lebihin, pol-in | tolong, dong, dikit, banyak, sedikit, makin, agak, coba |
| **`decrease`** (Menurunkan) | turunin, kecilin, redupin, dinginin, sejukin, rendahkan, kurangi, kurangin, perlahan, kecilin dikit, buat redup | tolong, dong, dikit, sedikit, agak, rada, coba |
| **`bring`** (Membawakan) | ambilin, bawain, antar, anterin, bawakan, ambilkan, jemput, unjukin, tolong ambilin | tolong, dong, buat gua, buat gue, buat saya, sini |
| **`change language`** (Ganti Bahasa) | ganti bahasa, ubah bahasa, ganti ke, switch ke, switch bahasa, gantiin bahasa, pindah ke bahasa | tolong, dong, coba, yuk, sekarang |

---

### B. Matriks Objek (Object Matrix)
| Objek Sistem | Istilah Formal & Slang Indonesia | Variasi Frasa Benda |
| :--- | :--- | :--- |
| **`lights`** | lampu, penerangan, lampu utama, lampu ruangan, semua lampu, lampu tengah, lampu gantung | lampunya, lampu-lampu, bohlam, cahaya |
| **`lamp`** | lampu meja, lampu tidur, lampu belajar, lampu baca, lampu sudut, lampu kecil, lampu tidur sudut | lamp meja, night lamp, lampu samping |
| **`music`** | musik, lagu, audio, track, playlist, suara lagu, setelan musik, mp3 | lagunya, musiknya, playernya |
| **`volume`** | volume, suara, bunyi, sound, audio volume, kekerasan suara | volumenya, suaranya, bunyinya, sound-nya |
| **`heat`** | suhu, temperatur, ac, hawa, pendingin, penghangat, iklim kamar, temperatur ruangan | suhunya, temperaturnya, hawa ruangannya, ac-nya |
| **`juice`** | jus, jus buah, jus jeruk, minum jus, sari buah, jus segar | jusnya, minuman jus |
| **`newspaper`** | koran, koran pagi, koran hari ini, surat kabar, harian, koran kompas/tempo | korannya, bacaan koran |
| **`shoes`** | sepatu, alas kaki, sepatu kerja, sepatu jalan, sneakers | sepatunya, sendal/sepatu |
| **`socks`** | kaos kaki, kaus kaki, kaos kaki kerja, kaos kaki bersih | kaos kakinya, sepasang kaos kaki |
| **`Chinese`** | bahasa mandarin, mandarin, bahasa cina, chinese | bahasa mandarinnya |
| **`English`** | bahasa inggris, inggris, english | bahasa inggrisnya |
| **`German`** | bahasa jerman, jerman, german | bahasa jermannya |
| **`Korean`** | bahasa korea, korea, korean, hangul | bahasa koreanya |

---

### C. Matriks Lokasi (Location Matrix)
| Lokasi Sistem | Istilah Formal & Slang Indonesia | Preposisi & Frasa Tempat |
| :--- | :--- | :--- |
| **`kitchen`** | dapur, tempat masak, ruang masak, ruang makan samping, area dapur, kitchen | di dapur, pada dapur, ke dapur, area kitchen |
| **`bedroom`** | kamar tidur, kamar, tempat tidur, ruang tidur, kamar utama, bilik, bedroom | di kamar tidur, di kamar, ke kamar |
| **`washroom`** | kamar mandi, toilet, wc, kamar kecil, peturasan, washroom, restroom | di kamar mandi, di wc, di toilet, ke kamar mandi |
| **`none`** | (Tanpa lokasi eksplisit / berlaku global untuk seluruh rumah) | di sini, ruangan ini, semuanya, sekarang |

---

### D. Partikel Spontan, Kalimat Pembuka & Pengisi Percakapan
* **Kata Pembuka Spontan**: *"Tolong"*, *"Minta tolong"*, *"Coba"*, *"Bisa tolong"*, *"Eh"*, *"Bro"*, *"Google/Svara"*, *"Punten"*, *"Bisa tolongin"*, *"Tolongin dong"*.
* **Partikel Tutur Akhir**: *"dong"*, *"deh"*, *"lah"*, *"kan"*, *"sih"*, *"ya"*, *"plis"*, *"buruan"*, *"cepetan"*.
* **Keterangan Intensitas / Adverbia**: *"dikit"*, *"sedikit"*, *"banget"*, *"agak"*, *"rada"*, *"pol"*, *"full"*.

---

## 2. Taksonomi Permutasi Sintaksis (Pola Susunan Kalimat Bebas)

Orang Indonesia memiliki fleksibilitas susunan kalimat yang sangat tinggi. Sistem SVARA dilatih pada 6 pola sintaksis:

### Pola 1: Standar Formal ($Aksi \rightarrow Objek \rightarrow Lokasi$)
* *"Nyalakan lampu dapur."*
* *"Matikan musik di lorong."*
* *"Kecilkan volume suara."*

### Pola 2: Santai dengan Partikel ($Aksi + \text{Partikel} \rightarrow Objek \rightarrow Lokasi$)
* *"Nyalain dong lampu di dapur."*
* *"Matiin deh musiknya."*
* *"Redupin sedikit lampu kamar tidur ya."*

### Pola 3: Objek di Depan ($Objek \rightarrow Lokasi \rightarrow Aksi$)
* *"Lampu dapur tolong dimatikan."*
* *"Musik di ruangan tolong diputar."*
* *"Suhu kamar tidur tolong dinaikkan."*

### Pola 4: Lokasi di Depan ($Lokasi \rightarrow Objek \rightarrow Aksi$)
* *"Di dapur lampunya tolong nyalain dong."*
* *"Kamar mandi lampunya matiin ya."*
* *"Di kamar tidur suhunya tolong didinginin."*

### Pola 5: Implisit / Keadaan Sekitar ($Kondisi \rightarrow Permintaan$)
* *"Dapur gelap banget nih tolong terangin lampunya."*
* *"Kamar tidur gerah tolong sejukin ac-nya."*
* *"Berisik banget tolong kecilin suaranya."*

### Pola 6: Singkat / Perintah Cepat ($Aksi + Objek + Lokasi \text{ Tanpa Imbuhan}$)
* *"Lampu dapur mati."*
* *"Lampu kamar on."*
* *"Ac kamar turun."*
* *"Volume up."*

---

## 3. Matriks Target Distribusi Data Korpus (31 Intent)

Untuk menjamin distribusi merata (*stratified balance*) tanpa ketimpangan kelas, setiap intent dari ke-31 intent dialokasikan kuota minimum **130 hingga 150 variasi frasa unik**:

| Kelompok Domain Intent | Jumlah Intent | Target Frasa per Intent | Sub-Total Kalimat Unik |
| :--- | :---: | :---: | :---: |
| **Pencahayaan (Kitchen, Bedroom, Washroom, None, Lamp)** | 10 Intent | 140 variasi | **1.400 kalimat** |
| **Suhu / Heat (Kitchen, Bedroom, Washroom, None - Naik/Turun)** | 8 Intent | 140 variasi | **1.120 kalimat** |
| **Musik & Volume (Aktifkan, Matikan, Naikkan, Turunkan)** | 4 Intent | 150 variasi | **600 kalimat** |
| **Robot Assistant / Bring (Juice, Newspaper, Shoes, Socks)** | 4 Intent | 130 variasi | **520 kalimat** |
| **Ganti Bahasa (Mandarin, English, German, Korean, None)** | 5 Intent | 130 variasi | **650 kalimat** |
| **TOTAL KORPUS TEKS BARU** | **31 Intent** | **~138 variasi (Rata-rata)** | **4.290 KALIMAT UNIK** |

---

## 4. Rencana Implementasi Script Generator

Script pembangun korpus skala 4.000+ kalimat akan diimplementasikan pada:
👉 `scripts/generate_massive_indonesian_corpus.py`

### Algoritma Generator:
1. Memuat matriks leksikon di atas ke dalam struktur *Dictionary* terindeks.
2. Melakukan *Cartesian Product & Grammar Templates* secara deterministik (menggunakan fixed seed `seed=42`).
3. Menyaring duplikasi teks murni (*semantic deduplication*).
4. Menyimpan hasil korpus bersih ke `data/corpus/indonesian_smart_home_corpus_4k.csv` dengan kolom standar:
   `intent_id, action, object, location, english_sample, indonesian_phrase`.
