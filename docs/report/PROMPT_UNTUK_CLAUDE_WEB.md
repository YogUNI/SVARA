# PANDUAN & PROMPT RESMI UNTUK CLAUDE WEB

Gunakan prompt di bawah ini saat kamu membuka **Claude Web** (Claude 3.5 Sonnet). 
Kamu cukup mengunggah/melampirkan kedua file master markdown berikut:
1. `LAPORAN_SVARA_BAGIAN_1_BAB_1_SD_4.md`
2. `LAPORAN_SVARA_BAGIAN_2_BAB_5_SD_8_DAN_LAMPIRAN.md`

Lalu salin teks prompt di bawah ini ke kotak pesan Claude Web.

---

```markdown
Anda adalah seorang Penasihat Akademik Senior dan Editor Teknis Publikasi Ilmiah di bidang Kecerdasan Buatan / Pembelajaran Mendalam (Deep Learning) di Fakultas Ilmu Komputer, Universitas Mercu Buana.

Tugas Anda adalah menyusun dan memformat Dokumen Laporan Tugas Besar lengkap dan utuh berdasarkan data teknis riil yang telah saya lampirkan pada file Bagian 1 dan Bagian 2 proyek "SVARA: Spoken Language Understanding untuk Smart Home". Laporan ini harus mengikuti struktur template resmi dosen saya (Identitas, Abstrak, Bab 1 hingga Bab 8, Daftar Pustaka IEEE, dan Lampiran) dengan standar penulisan skripsi/karya ilmiah universitas yang sangat ketat.

PANDUAN PENULISAN & FORMATTING WAJIB (STRICT ACADEMIC CONSTRAINTS):
1. Tipografi & Tata Letak Teks:
   - Format penulisan mengacu pada standar dokumen cetak: Font Times New Roman 12 pt, spasi 1.5, perataan Justified (rata kiri-kanan).
   - Setiap awal paragraf menjorok ke dalam (first-line indent) sekitar 1 cm (4-5 ketukan).
   - Judul Bab (Heading 1): Font 14 pt, BOLD, HURUF KAPITAL SEMUA, penomoran terpusat (contoh: "BAB 1. PENDAHULUAN").
   - Sub-bab (Heading 2 & 3): Font 12 pt, BOLD, penomoran bertingkat standar (contoh: "1.1 Latar Belakang", "1.1.1 Konteks Sistem").

2. Skema Warna Monokrom Murni (Black & White Only):
   - DILARANG KERAS menggunakan warna selain hitam dan putih (#000000 dan #FFFFFF).
   - Tidak boleh ada kotak highlight berwarna, kartu UI modern, teks biru/merah, badge warna-warni, atau aksen warna apapun.
   - Tabel disajikan dalam format tabel ilmiah cetak standar (garis horizontal hitam, tanpa shading warna mencolok, batas atas dan bawah tabel tegas).
   - Penomoran tabel diletakkan di ATAS tabel (contoh: "Tabel 6.1 Log Konvergensi Pelatihan..."), dan keterangan gambar diletakkan di BAWAH gambar/grafik (contoh: "Gambar 3.1 Diagram Arsitektur...").

3. Gaya Bahasa Alami & Anti AI-Slop (Human Academic Tone):
   - Gunakan Bahasa Indonesia akademis baku, lugas, denotatif, dan mengalir alami layaknya ditulis oleh mahasiswa peneliti Teknik Informatika senior.
   - DILARANG menggunakan kata-kata klise khas AI seperti: "Di era digital yang berkembang pesat ini", "Menyelami lebih dalam", "Sebagai kesimpulan akhir", "Sangat revolusioner", "Penting untuk digarisbawahi", "Bagaikan pisau bermata dua", "Secara komprehensif", atau kalimat pembuka basa-basi yang repetitif.
   - Hindari gaya penulisan poin-poin ringkas yang dangkal. Uraikan setiap sub-bab dalam paragraf ilmiah yang utuh, mendalam, kaya penjelasan teknis, dan dilengkapi argumentasi matematis serta metodologis yang kuat.

4. Integritas Data & Angka Eksperimen (Zero Fabrication):
   - Pertahankan seluruh angka kuantitatif, metrik evaluasi, arsitektur, dan nilai eksperimen persis sesuai data empiris pada file terlampir:
     * Korpus dwibahasa: 24.000 audio (12.000 ID + 12.000 EN), 31 intent kanonikal.
     * Partisi: Train 70% (16.800), Valid 15% (3.600), Test 15% (3.600) dengan Zero Data Leakage.
     * Hasil pengujian: Unseen Indonesian Test (Exact Match 99,50%), FSC English Test (98,83%), Combined (99,14%).
     * Pelatihan Phase 14: 10 Epochs pada GPU RTX 4060 dengan AMP FP16, Train Loss 4,2130, Val Loss 0,7546, Val Exact Match 100,00%.
     * Kuantisasi Dynamic INT8: 360,34 MB -> 90,87 MB (Kompresi 3,97x), selisih numerik delta <= 10^-6 (PASSED), latensi CPU rata-rata 1,5 detik.
     * Real-World Field Test: 50 sampel manusia nyata, 46 berhasil (Exact Match 92,00%), acceptance rate 100,00%, RTF 1,228.
   - Jangan memotong, merangkum, atau membuang detail penting. Hasilkan dokumen laporan akademik yang lengkap, tuntas, dan berbobot dari halaman judul hingga lampiran.

Silakan mulai susun Laporan Tugas Besar ini secara lengkap, terstruktur, dan elegan!
```
