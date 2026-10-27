# Laporan Ilmiah Audit & Validasi Integritas Linguistik Korpus Bahasa Indonesia 4K
**Proyek**: SVARA (Smart Voice Assistant for Residential Automation)  
**Referensi Standar**: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md (Sub-step 2.3)  
**Status Audit**: **PASSED (100% CLEAN)**  

---

## 1. Ringkasan Eksekutif Korpus Teks
Korpus perintah suara rumah pintar berbahasa Indonesia telah melewati pengujian dan audit integritas data berlapis. Korpus ini dirancang untuk melatih dan mengevaluasi model Spoken Language Understanding (SLU) *end-to-end* berbasis Transformer (`facebook/wav2vec2-base`) yang seimbang antara bahasa Inggris (Fluent Speech Commands) dan bahasa Indonesia sehari-hari.

| Parameter Evaluasi | Nilai Empiris Riil | Standar Kelayakan Ilmiah | Hasil |
|---|:---:|:---:|:---:|
| Total Data Frasa | **4,495** | $\ge 4.000$ frasa | **LULUS** |
| Total Frasa Unik | **4,495** | $100\%$ unik | **LULUS** |
| Duplikasi Teks (Exact Duplicates) | **0** ($0.00\%$) | $0$ duplikasi | **LULUS** |
| Ambigu / Kolisi Lintas-Intent | **0** ($0.00\%$) | $0$ kolisi | **LULUS** |
| Jumlah Intent Terpetakan | **31 / 31** | Tepat $31$ intent FSC | **LULUS** |
| Keseimbangan Sampel per Intent | **Tepat 145 frasa/intent** | $\sigma = 0.00$ (Zero Class Imbalance) | **LULUS** |
| Rasio Entropi Shannon Ternormalisasi ($H / H_{max}$) | **1.000000** | $1.000000$ (Sebaran Seragam Murni) | **LULUS** |
| Kosakata Unik (Lexicon Size) | **317 kata** | $\ge 300$ kata | **LULUS** |
| Rata-rata Panjang Kalimat | **6.49 kata** ($6.39 \pm 1.33$) | $4.0 - 8.0$ kata | **LULUS** |
| Frasa Trivial Kosong (< 2 kata) | **0 frasa** | $0$ kalimat trivial | **LULUS** |

---

## 2. Analisis Keseimbangan Stratifikasi & Entropi Sebaran Kelas
Dalam klasifikasi *multi-task* dengan 31 intent gabungan, ketimpangan kelas (*class imbalance*) dapat menyebabkan model bias terhadap kelas mayoritas. Pengujian empiris menunjukkan:
- **Jumlah Sampel per Kelas**: Tepat **145 kalimat** untuk setiap intent ID dari 0 sampai 30.
- **Standar Deviasi Antar-Kelas**: $\sigma = \mathbf{0.0000}$ dan Variansi $\sigma^2 = \mathbf{0.0000}$.
- **Entropi Shannon Informasi**:
  $$H(X) = -\sum_{i=1}^{31} p_i \log_2(p_i) = 4.9542\text{ bits}$$
  $$H_{max} = \log_2(31) = 4.9542\text{ bits}$$
  $$\frac{H(X)}{H_{max}} = \mathbf{1.000000}$$
Rasio entropi $1.000000$ membuktikan bahwa sebaran korpus bersifat *perfectly uniform*, menjamin ketiadaan bias probabilitas *prior* selama tahapan *fine-tuning*.

---

## 3. Sebaran 5 Domain Fungsional Smart Home
Korpus terdistribusi ke dalam 5 domain fungsional rumah pintar secara representatif:

| Domain Fungsional Smart Home | Jumlah Intent | Jumlah Frasa | Proporsi Korpus |
|---|:---:|:---:|:---:|
| **Climate (Pengatur Suhu & AC)** | 8 intent | 1,160 frasa | 25.81% |
| **Lighting (Pencahayaan)** | 10 intent | 1,450 frasa | 32.26% |
| **Media & Audio (Hiburan & Volume)** | 4 intent | 580 frasa | 12.9% |
| **Robot Butler (Pengantaran Barang)** | 4 intent | 580 frasa | 12.9% |
| **System Language (Pengaturan Bahasa)** | 5 intent | 725 frasa | 16.13% |
| **TOTAL KESELURUHAN** | **31 intent** | **4,495 frasa** | **100.00%** |

---

## 4. Karakteristik Panjang Kalimat & Sintaksis Spontan
Kalimat ucapan pengguna smart home di dunia nyata bervariasi dari perintah satu nafas hingga kalimat kontekstual lengkap.

- **Rata-rata Kata**: 6.49 kata (Median: 6.0, Min: 2, Max: 12)
- **Rata-rata Karakter**: 37.85 karakter (Median: 38.0, Min: 11, Max: 70)
- **Sebaran Panjang Kata**:
  - 2-3 kata (Perintah singkat / direct imperative): **29 frasa**
  - 4-5 kata (Pola perintah standar): **967 frasa**
  - 6-7 kata (Perintah dengan sopan santun / partikel): **2,551 frasa**
  - 8-9 kata (Perintah terbalik dengan keluhan situasi): **895 frasa**
  - 10+ kata (Perintah kompleks majemuk): **53 frasa**

---

## 5. Cakupan Leksikon, Verba Slang, dan Partikel Daerah
Korpus berhasil merepresentasikan gaya bicara penutur bahasa Indonesia sehari-hari:
- **Total Token Kata**: 29,169 kata.
- **Kosakata Unik**: 317 kata unik (Type-Token Ratio: 0.0109).
- **Verba Slang & Kolokial**:
  - `nyalain`: 40 kali muncul
  - `matiin`: 60 kali muncul
  - `idupin`: 45 kali muncul
  - `naikin`: 29 kali muncul
  - `turunin`: 34 kali muncul
  - `bawain`: 85 kali muncul
  - `ambilin`: 69 kali muncul
  - `puter`: 28 kali muncul
  - `setel`: 123 kali muncul
  - `ademin`: 38 kali muncul
- **Partikel Spontan & Sopan Santun**:
  - `dong`: 777 kali muncul
  - `deh`: 417 kali muncul
  - `lah`: 390 kali muncul
  - `sih`: 110 kali muncul
  - `tolong`: 1011 kali muncul
  - `atuh`: 319 kali muncul
  - `bro`: 469 kali muncul
  - `cuy`: 126 kali muncul
  - `lur`: 780 kali muncul
  - `gan`: 125 kali muncul
  - `ya`: 1085 kali muncul
- **Konteks Situasional & Keluhan**:
  - `gerah`: 4 kali muncul
  - `gelap`: 3 kali muncul
  - `berisik`: 3 kali muncul
  - `silau`: 12 kali muncul
  - `haus`: 2 kali muncul
  - `dingin`: 60 kali muncul
  - `panas`: 97 kali muncul

---

## 6. Kesimpulan & Rekomendasi
Korpus teks `data/corpus/indonesian_smart_home_corpus_4k.csv` telah terbukti secara empiris:
1. **Bebas dari duplikasi** (0 duplikasi dari 4.495 baris).
2. **Bebas dari kolisi lintas-intent** (0 ambiguitas semantik).
3. **Memiliki keseimbangan kelas sempurna** (145 frasa per intent, $\sigma = 0.00$).
4. **Kaya akan variasi leksikon gaul dan partikel percakapan alami**.

Korpus ini dinyatakan **SIAP DAN VALID 100%** untuk dilanjutkan ke **Fase 3: Sintesis Suara Multi-Speaker Massal (Neural TTS 8.000 - 10.000 Audio)**.
