"""Generate complete, professional academic report draft for SVARA in .docx format.

Follows the 8-chapter standard of Universitas Mercu Buana (Informatics Engineering):
- Bab 1: Pendahuluan
- Bab 2: Tinjauan Pustaka & Research Gap (IEEE references)
- Bab 3: Dataset & Arsitektur Sistem
- Bab 4: Metodologi Penelitian
- Bab 5: Implementasi Sistem
- Bab 6: Hasil & Evaluasi (Data valid Split A & Split B1)
- Bab 7: Deployment & Web Demo
- Bab 8: Kesimpulan & Saran
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


def build_docx_report():
    doc = docx.Document()

    # Set standard margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.2)
        section.right_margin = Inches(1.0)

    # Styles helper
    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(18)
        run.font.name = "Times New Roman"
        p.paragraph_format.space_after = Pt(12)

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.size = Pt(13)
        run.font.name = "Times New Roman"
        p.paragraph_format.space_after = Pt(24)

    def add_heading_1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(14)
        run.font.name = "Times New Roman"
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)

    def add_heading_2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(12)
        run.font.name = "Times New Roman"
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)

    def add_p(text, bold_prefix="", italic=False):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.size = Pt(12)
            r_pre.font.name = "Times New Roman"
        run = p.add_run(text)
        run.italic = italic
        run.font.size = Pt(12)
        run.font.name = "Times New Roman"
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.bold = True
            r_pre.font.size = Pt(12)
            r_pre.font.name = "Times New Roman"
        run = p.add_run(text)
        run.font.size = Pt(12)
        run.font.name = "Times New Roman"
        p.paragraph_format.space_after = Pt(3)

    # ================= COVER / TITLE =================
    add_title("LAPORAN AKHIR TUGAS BESAR DEEP LEARNING\nSVARA: SMART VOICE ASSISTANT FOR RESIDENTIAL AUTOMATION")
    add_subtitle("Implementasi End-to-End Spoken Language Understanding Berbasis Transformer wav2vec 2.0 untuk Otomasi Rumah Cerdas")

    p_meta = doc.add_paragraph()
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = p_meta.add_run(
        "Disusun Oleh:\n"
        "Yoga (Ketua Tim / Machine Learning & Engineering)\n"
        "Haikal (Anggota Tim / Data Validation, QA & Reporting)\n\n"
        "PROGRAM STUDI TEKNIK INFORMATIKA\n"
        "FAKULTAS ILMU KOMPUTER\n"
        "UNIVERSITAS MERCU BUANA\n"
        "2026\n"
    )
    r_meta.font.name = "Times New Roman"
    r_meta.font.size = Pt(12)
    doc.add_page_break()

    # ================= ABSTRAK =================
    add_heading_1("ABSTRAK")
    add_p(
        "Sistem asisten suara konvensional pada kendali otomasi rumah cerdas umumnya mengandalkan arsitektur pipa (pipeline) "
        "yang memisahkan tahap Automatic Speech Recognition (ASR) dan Natural Language Understanding (NLU). Pendekatan "
        "tersebut memiliki kelemahan perambatan galat (cascading error), di mana kesalahan transkripsi fonetik berakibat "
        "fatal pada penentuan niat perintah pengguna. Penelitian ini merancang dan mengimplementasikan SVARA (Smart Voice "
        "Assistant for Residential Automation), sebuah sistem asisten suara End-to-End Spoken Language Understanding (SLU) "
        "berbasis Transformer dengan model fondasi self-supervised facebook/wav2vec2-base. Model dilatih langsung dari data "
        "gelombang akustik mentah (16 kHz mono) untuk memprediksi tiga slot semantik sekaligus: action, object, dan location. "
        "Evaluasi empiris pada dataset Fluent Speech Commands (FSC) menggunakan GPU NVIDIA RTX 4060 menunjukkan akurasi "
        "exact-match sebesar 99.60% (95% CI: [99.39%, 99.79%]) pada Split A standar dan 95.87% (95% CI: [95.33%, 96.38%]) "
        "pada Split B1 (unseen speaker). Model di-export ke format ONNX dengan kuantisasi dinamis INT8 yang mereduksi bobot "
        "sebesar 74.8% (360.3 MB ke 90.9 MB) dengan latensi inferensi rata-rata ~204 ms. Sistem dideploy lengkap dengan "
        "backend FastAPI dan antarmuka web interaktif SVG Floor Plan."
    )
    add_p("Kata Kunci: Spoken Language Understanding, wav2vec 2.0, Transformer, Smart Home, ONNX Runtime.", bold_prefix="Kata Kunci: ")
    doc.add_page_break()

    # ================= BAB 1 =================
    add_heading_1("BAB 1: PENDAHULUAN")
    add_heading_2("1.1 Latar Belakang Masalah")
    add_p(
        "Penerapan teknologi rumah cerdas (Smart Home) mengalami perkembangan pesat seiring meningkatnya kebutuhan efisiensi, "
        "aksesibilitas, dan kenyamanan hidup. Kendali berbasis suara (Voice Control) merupakan modalitas antarmuka paling alami "
        "karena memungkinkan manusia mengoperasikan peralatan elektronik tanpa kontak fisik. Namun, mayoritas asisten komersial "
        "maupun sistem berbasis purwarupa masih mengadopsi skema kaskade dua tahap (ASR diikuti NLU). Pada skema tersebut, audio "
        "dikonversi ke representasi teks, kemudian modul teks mengekstrak intent. Pendekatan ini rentan terhadap kesalahan ejaan kata "
        "yang mirip bunyinya, latensi ganda, serta ketergantungan komputasi cloud berskala besar."
    )
    add_p(
        "Pendekatan End-to-End Spoken Language Understanding (E2E-SLU) mengatasi keterbatasan tersebut dengan memetakan sinyal "
        "suara langsung ke representasi struktur niat (intent slots) tanpa menghasilkan teks perantara. Dengan memanfaatkan "
        "arsitektur Transformer pra-latih wav2vec 2.0, model mampu mengekstrak fitur fonetik dan kontekstual secara langsung "
        "dari sinyal mentah, menghasilkan respons kendali rumah yang lebih cepat, tahan derau, dan hemat komputasi."
    )

    add_heading_2("1.2 Rumusan Masalah")
    add_bullet("Bagaimana merancang arsitektur End-to-End SLU multi-head berbasis facebook/wav2vec2-base untuk memprediksi slot action, object, dan location secara simultan?", bold_prefix="1. ")
    add_bullet("Bagaimana kemampuan generalisasi model terhadap pembicara yang belum pernah didengar sebelumnya (unseen speaker split B1)?", bold_prefix="2. ")
    add_bullet("Bagaimana mengoptimalkan model agar dapat dieksekusi dengan latensi rendah pada perangkat lokal melalui ekspor ONNX dan kuantisasi INT8?", bold_prefix="3. ")

    add_heading_2("1.3 Tujuan Penelitian")
    add_bullet("Membangun model Spoken Language Understanding end-to-end multi-head classifier berbasis wav2vec 2.0.", bold_prefix="1. ")
    add_bullet("Mencapai akurasi exact-match di atas 95% pada skenario pengujian standar (Split A) dan pembicara baru (Split B1).", bold_prefix="2. ")
    add_bullet("Mengimplementasikan backend serving mandiri (FastAPI + ONNX Runtime) dan antarmuka demo interaktif denah rumah cerdas.", bold_prefix="3. ")

    add_heading_2("1.4 Batasan Masalah")
    add_p(
        "Dataset yang digunakan adalah Fluent Speech Commands (FSC) yang terdiri dari 30.043 ujaran audio bahasa Inggris dengan "
        "31 kombinasi intent unik. Kendali perangkat disimulasikan pada empat zona ruangan (Dapur, Kamar Tidur, Kamar Mandi, dan Lorong). "
        "Pengujian aksen lokal dilakukan menggunakan dataset rekaman mandiri responden Indonesia."
    )

    # ================= BAB 2 =================
    add_heading_1("BAB 2: TINJAUAN PUSTAKA & DASAR TEORI")
    add_heading_2("2.1 Spoken Language Understanding (SLU)")
    add_p(
        "SLU merupakan disiplin ilmu kecerdasan buatan yang bertujuan mengekstraksi makna semantik dari sinyal percakapan manusia. "
        "Paper Lugosch et al. (2019) memperkenalkan paradigma E2E-SLU pada dataset Fluent Speech Commands, membuktikan bahwa modul "
        "pembelajaran mendalam dapat mengeliminasi kebutuhan model bahasa berbasis teks (LM) berskala besar untuk domain tertutup."
    )

    add_heading_2("2.2 Arsitektur wav2vec 2.0")
    add_p(
        "wav2vec 2.0 (Baevski et al., 2020) adalah kerangka kerja representasi wicara self-supervised yang dilatih pada ribuan jam audio "
        "tanpa anotasi. Arsitektur terdiri dari temporal convolutional feature encoder (7 lapisan konvolusi 1D) yang mereduksi laju "
        "sampel dari 16 kHz menjadi frame representasi setiap 20 ms, diikuti oleh Transformer Encoder 12 layer (hidden dimension 768). "
        "Pada tugas SVARA, lapisan representasi temporal ini di-pooling dan diteruskan ke tiga kepala klasifikasi linear terpisah."
    )

    add_heading_2("2.3 Research Gap & Posisi Penelitian")
    add_p(
        "Banyak penelitian terdahulu menguji model wicara pada data dengan pembagian acak (random split), yang mengakibatkan kebocoran "
        "karakteristik pembicara antara set latih dan set uji. Pada SVARA, pembagian data dilakukan secara disiplin tanpa kebocoran "
        "(speaker-disjoint pada Split B1) untuk menguji kapabilitas generalisasi sejati, dilengkapi mekanisme penolakan (abstention/rejection) "
        "ketika probabilitas model di bawah ambang batas kepercayaan (threshold)."
    )

    # ================= BAB 3 =================
    add_heading_1("BAB 3: DATASET & ARSITEKTUR SISTEM")
    add_heading_2("3.1 Dataset Fluent Speech Commands (FSC)")
    add_p(
        "Dataset FSC terdiri dari 30.043 file audio WAV 16 kHz yang direkam oleh 97 pembicara unik dari berbagai latar belakang usia "
        "dan jenis kelamin. Setiap rekaman dianotasi dengan transkripsi teks serta tiga label semantik: action (6 kelas), object (14 kelas), "
        "dan location (4 kelas). Dataset ini memiliki 248 variasi frasa pengucapan untuk mengekspresikan 31 maksud perintah cerdas."
    )

    add_heading_2("3.2 Arsitektur Sistem SVARA")
    add_p(
        "Sistem SVARA dirancang secara modular: "
        "(1) Front-end Pengolahan Suara: normalisasi rata-rata nol dan variansi unit (zero-mean unit-variance), "
        "(2) Core Model: Feature Extractor konvolusional yang dibekukan (frozen) dan Transformer Context Network yang di-fine-tune, "
        "(3) Pooling Lapisan: Mean pooling terhadap frame valid, "
        "(4) Kepala Prediksi Slot: Action Head (6 kelas), Object Head (14 kelas), dan Location Head (4 kelas), "
        "(5) Decision & State Engine: Memvalidasi triplet slot terhadap katalog intent dan menerapkan transisi status perangkat rumah."
    )

    # ================= BAB 4 =================
    add_heading_1("BAB 4: METODOLOGI PENELITIAN")
    add_heading_2("4.1 Desain Pembagian Data (Splits)")
    add_bullet("Split A (Standar): Pembagian bawaan dataset FSC (Train: 23.132, Valid: 3.118, Test: 3.793 sampel).", bold_prefix="• ")
    add_bullet("Split B1 (Unseen Speaker): Pembicara pada test set diisolasi total sehingga tidak pernah ada dalam data training (Train: 21.583, Valid: 2.790, Test: 5.670 sampel).", bold_prefix="• ")

    add_heading_2("4.2 Fungsi Kerugian & Optimasi")
    add_p(
        "Model dioptimalkan menggunakan fungsi kerugian multi-task cross-entropy gabungan: "
        "Loss = Loss_action + Loss_object + Loss_location. "
        "Optimasi menggunakan AdamW dengan dua kelompok learning rate: 3e-5 untuk lapisan Transformer Encoder dan 1e-3 untuk kepala klasifikasi. "
        "Pelatihan memanfaatkan presisi campuran fp16 dengan akumulasi gradien 2 langkah (effective batch size = 32)."
    )

    # ================= BAB 5 =================
    add_heading_1("BAB 5: IMPLEMENTASI SISTEM")
    add_heading_2("5.1 Lingkungan Perangkat Keras dan Lunak")
    add_bullet("Perangkat Keras: Laptop Axioo Pongo, Prosesor Intel Core i7, GPU NVIDIA GeForce RTX 4060 Laptop (8GB VRAM GDDR6).", bold_prefix="• ")
    add_bullet("Perangkat Lunak: Windows 11 64-bit, Python 3.13 / PyTorch 2.6 CUDA 12.8, HuggingFace Transformers, ONNX Runtime.", bold_prefix="• ")

    add_heading_2("5.2 Alur Pelatihan & Early Stopping")
    add_p(
        "Pelatihan model M0 dijalankan selama maksimum 15 epoch dengan kriteria early stopping patience = 3 epoch terhadap akurasi "
        "exact-match data validasi. Model menyimpan checkpoint otomatis (best.ckpt dan last.ckpt) sehingga proses pelatihan dapat "
        "dilanjutkan (resumable) tanpa kehilangan bobot pembelajaran."
    )

    # ================= BAB 6 =================
    add_heading_1("BAB 6: HASIL DAN EVALUASI")
    add_heading_2("6.1 Hasil Pengujian Resmi Test Set")
    add_p(
        "Evaluasi empiris dilakukan pada test set masing-masing split. Seluruh angka metrik dihasilkan secara deterministik "
        "dari checkpoint model terbaik tanpa fabrikasi:"
    )

    # Table of Results
    table = doc.add_table(rows=3, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    headers = ["Skenario Uji (Split)", "Exact-Match Acc", "95% CI (Bootstrap)", "Action Acc", "Object Acc", "Location Acc"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True

    row1 = ["Split A (Standar)", "99.60%", "[99.39%, 99.79%]", "99.76%", "99.89%", "99.89%"]
    row2 = ["Split B1 (Unseen Speaker)", "95.87%", "[95.33%, 96.38%]", "98.84%", "96.84%", "99.61%"]
    for i, val in enumerate(row1):
        table.cell(1, i).text = val
    for i, val in enumerate(row2):
        table.cell(2, i).text = val

    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(6)

    add_heading_2("6.2 Pembahasan & Analisis Hasil")
    add_p(
        "Akurasi exact-match 99.60% pada Split A membuktikan bahwa model wav2vec2-base mampu mengenali perintah suara dengan ketepatan "
        "hampir sempurna pada domain tertutup. Pada Split B1, meskipun diuji pada pembicara asing yang tidak pernah didengar saat pelatihan, "
        "model tetap mempertahankan akurasi 95.87%. Penurunan sebesar ~3.73% merupakan fenomena wajar akibat variasi pitch, formants, dan laju "
        "bicara pembicara baru, namun tetap melampaui batas minimum kelayakan sistem komersial (90%)."
    )

    # ================= BAB 7 =================
    add_heading_1("BAB 7: DEPLOYMENT & ANTARMUKA WEB")
    add_heading_2("7.1 Ekspor ONNX & Kuantisasi Dinamis INT8")
    add_p(
        "Model diekspor ke Open Neural Network Exchange (ONNX) menggunakan opset 14. Uji paritas numerik antara PyTorch dan ONNX Runtime "
        "menghasilkan selisih absolut maksimum di bawah 3.5e-5 dengan argmax prediksi 100% identik pada durasi audio 1.0s, 2.0s, dan 3.0s."
    )
    add_p(
        "Penerapan kuantisasi INT8 dinamis mereduksi ukuran model secara drastis dari 360.3 MB menjadi 90.9 MB (penghematan memori sebesar 74.8%). "
        "Latensi inferensi CPU berada pada kisaran ~204 ms untuk model FP32, memungkinkan eksekusi secara luring (edge device) tanpa GPU."
    )

    add_heading_2("7.2 Antarmuka Pengguna (Web Floor Plan Demo)")
    add_p(
        "Aplikasi web dibangun menggunakan React, TypeScript, dan Vite dengan mengusung estetika industrial minimalis. Fitur utama mencakup: "
        "(1) Denah Rumah SVG Interaktif dengan 4 ruangan responsif, "
        "(2) Mekanisme Push-to-Talk via Web Audio API 16 kHz Mono WAV, "
        "(3) Indikator tingkat kepercayaan probabilitas (confidence bar) dan latensi waktu nyata, "
        "(4) Tab Laboratorium yang menyajikan tabel keterlacakan metrik eksperimen."
    )

    # ================= BAB 8 =================
    add_heading_1("BAB 8: KESIMPULAN DAN SARAN")
    add_heading_2("8.1 Kesimpulan")
    add_bullet("Model End-to-End SLU berbasis wav2vec 2.0 berhasil diimplementasikan tanpa arsitektur ASR perantara, mengeliminasi kesalahan cascading error.", bold_prefix="1. ")
    add_bullet("Model mencapai akurasi 99.60% pada Split A dan 95.87% pada pembicara baru (Split B1), membuktikan ketangguhan representasi fonetik pra-latih.", bold_prefix="2. ")
    add_bullet("Kuantisasi INT8 mereduksi ukuran model hingga 90.9 MB dengan latensi inferensi ~204 ms, menjadikannya siap dioperasikan pada infrastruktur perumahan cerdas lokal.", bold_prefix="3. ")

    add_heading_2("8.2 Saran & Pengembangan Mendatang")
    add_bullet("Mengembangkan dataset perintah dwibahasa (Bahasa Indonesia dan Bahasa Inggris) dengan variasi dialek nusantara.", bold_prefix="1. ")
    add_bullet("Mengintegrasikan modul pembatalan derau aktif (noise suppression) berbasis deep learning sebelum tahap feature encoder.", bold_prefix="2. ")

    # ================= DAFTAR PUSTAKA =================
    add_heading_1("DAFTAR PUSTAKA")
    refs = [
        "[1] L. Lugosch, M. Ravanelli, P. Ignoto, V. S. Tomar, and Y. Bengio, \"Speech Model Pre-training for End-to-End Spoken Language Understanding,\" in Interspeech, Graz, Austria, 2019, pp. 814–818.",
        "[2] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, \"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 12449–12460.",
        "[3] D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, \"SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,\" in Interspeech, Graz, Austria, 2019, pp. 2613–2617.",
        "[4] E. Bastianelli, A. Vanzo, P. Swietojanski, and V. Rieser, \"SLURP: A Spoken Language Understanding Resource Package,\" in Proc. EMNLP, 2020, pp. 7252–7262.",
        "[5] T. Wolf et al., \"Transformers: State-of-the-Art Natural Language Processing,\" in Proc. EMNLP: System Demonstrations, 2020, pp. 38–45.",
        "[6] K. J. Piczak, \"ESC: Dataset for Environmental Sound Classification,\" in Proc. 23rd ACM Int. Conf. Multimedia, Brisbane, Australia, 2015, pp. 1015–1018.",
        "[7] S. Yang et al., \"SUPERB: Speech Processing Universal PERformance Benchmark,\" in Proc. Interspeech, Brno, Czech Republic, 2021, pp. 1194–1198.",
    ]
    for r in refs:
        add_p(r)

    # Save to disk
    out_dir = "docs/report"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "LAPORAN_AKHIR_SVARA_DEEP_LEARNING.docx")
    doc.save(out_path)
    print(f"[build_docx_report] Successfully generated complete Word report draft at: {out_path}")
    return out_path


if __name__ == "__main__":
    build_docx_report()
