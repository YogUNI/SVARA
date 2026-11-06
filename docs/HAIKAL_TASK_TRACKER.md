# SVARA — Task Tracking Haikal

Dokumen ini memantau progres tugas **Haikal** dalam proyek tugas akhir SVARA.

---

## 📊 Status Progres Ringkas

| No | Modul Tugas | Status | Keterangan |
|---|---|---|---|
| 1 | **Git & Branching Workflow** | ✅ **Selesai** | Branch `data/haikal-indonesia-corpus` aktif & terisolasi dari `main`. |
| 2 | **Riset Korpus NLP Bahasa Indonesia (31 Intent)** | ✅ **Selesai** | 465 variasi tutur komprehensif (15 variasi/intent) tersusun di Markdown & CSV + analisis linguistik mendalam untuk Deep Learning/SLU. |
| 3 | **Dataset Suara Sintetis Bahasa Indonesia (930 Audio WAV)** | ✅ **Selesai** | 930 file audio (16 kHz mono WAV, Ardi & Gadis) + `metadata.csv` sudah ter-generate & **sukses ter-push ke GitHub** di branch `data/haikal-indonesia-corpus`. |
| 4 | **Dataset Rekaman Suara Asli & Acoustic Anchor (Sub-step 13.2)** | ✅ **Selesai** | Perekaman 124 klip audio biologis Haikal (31 intent Triple-Take: Baku, Nyantai, Slang) + metadata.csv terstandardisasi. |
| 5 | **Field-Test Lapangan 5 Domain Smart Home (Sub-step 14.3)** | ✅ **Selesai** | Benchmark 50 uji lapangan menghasilkan 78.00% exact match dengan latensi kilat 46.39 ms. |
| 6 | **Perekaman Klip Suara Bising Rumah Tangga (Noise Clips)** | ⏳ **Belum Mulai** | 10–15 klip suara lingkungan rumah (10–15 detik) untuk uji ketahanan. |
| 7 | **Penulisan Laporan Tugas Akhir (Bab 1, 2, dan 3)** | ⏳ **Belum Mulai** | Penulisan akademik Bahasa Indonesia standar IEEE. |
| 8 | **QA Pengujian Web Demo & Video Demonstrasi (P8–P9)** | ⏳ **Sedang Berjalan** | Uji coba UI web dan pembuatan video presentasi (3–5 menit). |

---

## 📋 Rincian Checklist Tugas Haikal

### Modul 1: Riset Korpus NLP Bahasa Indonesia
- [x] Inisialisasi branch kerja `data/haikal-indonesia-corpus`
- [x] Susun mapping 31 intent FSC ke slot semantik (`action`, `object`, `location`)
- [x] Buat variasi frasa penutur Indonesia per intent (Baku, Santai, Gaul/Slang, Partikel *dong/deh/nih*)
- [x] Analisis karakteristik sintaksis/topicalization (misal objek di depan: *"Kamar mandi lampunya matiin"*)
- [x] Simpan korpus ke format terstruktur (`docs/corpus_indonesia_31_intents.md` dan CSV di `data/corpus/`)

### Modul 2: Perekaman Suara (Phase P5 Own Set & Noise)
- [x] Inisialisasi tooling perekam interaktif Triple-Take (`scripts/record_own_voice.py`)
- [x] Rekam 124 klip vokal biologis Haikal (31 Intent x Baku, Nyantai, Slang) sebagai Acoustic Anchor (Sub-step 13.2)
- [x] Validasi & Standardisasi audio ke 16 kHz Mono PCM_16 anti-clipping Peak <= 0.95 (`data/own_recordings/metadata.csv` & `reports/own_recordings_audit.json`)
- [x] Eksekusi benchmark uji lapangan 5 domain smart home 50 skenario (`reports/real_world_field_test_results.json`)
- [ ] Kumpulkan 10–15 klip suara kebisingan murni rumah tangga (10–15 detik) untuk modul augmentasi bising
- [ ] Kumpulkan rekaman tambahan responden eksternal jika diperlukan untuk perluasan set evaluasi final

### Modul 3: Penulisan Laporan Akademik (Bab 1 – 3)
- [ ] **Bab 1 (Pendahuluan)**: Latar belakang Smart Home, kelemahan Pipeline ASR+NLU vs End-to-End SLU, rumusan masalah, batasan.
- [ ] **Bab 2 (Tinjauan Pustaka)**: Konsep SLU, arsitektur wav2vec 2.0 & Transformer, referensi standar IEEE.
- [ ] **Bab 3 (Metodologi & Data)**: Karakteristik FSC, skema split A & B1, pengumpulan Own Set Indonesia, korpus NLP.

### Modul 4: Pengujian Web (QA) & Video Presentasi
- [ ] Uji coba antarmuka web SVARA (responsivitas, voice input, denah rumah 3D)
- [ ] Catat feedback / temuan bug ke issue/tabel QA
- [ ] Rekam video demonstrasi (3–5 menit: perkenalan, cara kerja SLU, live demo suara, hasil akurasi)
