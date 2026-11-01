#!/usr/bin/env python3
"""Script to generate professional academic UTS Deep Learning answer sheet for SVARA project.

Output: Jawaban_UTS_Deep_Learning_SVARA_Yoga.docx
Target: Universitas Mercu Buana, Teknik Informatika, Deep Learning
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets cell shading color in hex format (e.g. '1B365D')."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=140, bottom=140, left=180, right=180):
    """Sets inner padding for table cell in twips."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def add_callout_box(doc, text, title="Catatan Lapangan & Refleksi Rekayasa:"):
    """Adds a stylish callout box with a colored left accent border."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)

    cell = table.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=160, bottom=160, left=240, right=200)

    # Set left border thick navy, others none
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="28" w:space="0" w:color="1B365D"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    run_t = p.add_run(title)
    run_t.bold = True
    run_t.font.name = 'Calibri'
    run_t.font.size = Pt(10.5)
    run_t.font.color.rgb = RGBColor(27, 54, 93)

    p2 = cell.add_paragraph()
    p2.paragraph_format.space_before = Pt(2)
    p2.paragraph_format.space_after = Pt(0)
    p2.paragraph_format.line_spacing = 1.15
    run_b = p2.add_run(text)
    run_b.font.name = 'Calibri'
    run_b.font.size = Pt(10)
    run_b.font.color.rgb = RGBColor(40, 50, 65)

    # spacer
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(6)

def add_image_box(doc, image_path, caption, width=Inches(6.0)):
    """Inserts a figure centered with an academic caption."""
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.add_run().add_picture(image_path, width=width)

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    run_cap = p_cap.add_run(caption)
    run_cap.italic = True
    run_cap.font.name = 'Calibri'
    run_cap.font.size = Pt(9.5)
    run_cap.font.color.rgb = RGBColor(90, 100, 115)

def build_uts_document():
    doc = docx.Document()

    # Set 1-inch margins
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)

    # Default font styling
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(45, 55, 72)

    # =========================================================================
    # HEADER SECTION
    # =========================================================================
    p_univ = doc.add_paragraph()
    p_univ.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_univ.paragraph_format.space_before = Pt(0)
    p_univ.paragraph_format.space_after = Pt(2)
    run_u = p_univ.add_run("UNIVERSITAS MERCU BUANA")
    run_u.bold = True
    run_u.font.size = Pt(15)
    run_u.font.color.rgb = RGBColor(27, 54, 93)

    p_fak = doc.add_paragraph()
    p_fak.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_fak.paragraph_format.space_before = Pt(0)
    p_fak.paragraph_format.space_after = Pt(2)
    run_f = p_fak.add_run("FAKULTAS ILMU KOMPUTER — PROGRAM STUDI TEKNIK INFORMATIKA")
    run_f.bold = True
    run_f.font.size = Pt(11)
    run_f.font.color.rgb = RGBColor(74, 96, 122)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(4)
    p_title.paragraph_format.space_after = Pt(12)
    run_t = p_title.add_run("LEMBAR JAWABAN UJIAN TENGAH SEMESTER (UTS)\nMATA KULIAH: DEEP LEARNING")
    run_t.bold = True
    run_t.font.size = Pt(13)
    run_t.font.color.rgb = RGBColor(27, 54, 93)

    # Student metadata table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    meta_table.columns[0].width = Inches(2.2)
    meta_table.columns[1].width = Inches(4.3)

    meta_data = [
        ("Nama Mahasiswa / Lead Engineer", "Yoga (Pemilik Repositori YogUNI/SVARA)"),
        ("Rekan Tim Kolaborator", "Haikal (Dataset Own-Recording & Evaluasi Akustik)"),
        ("Topik Proyek Tugas Besar", "SVARA (Smart Voice Assistant for Residential Automation)"),
        ("Spesifikasi Hardware & Model", "NVIDIA GeForce RTX 4060 Laptop GPU | Transformer wav2vec 2.0 SLU"),
    ]

    for idx, (label, val) in enumerate(meta_data):
        row = meta_table.rows[idx]
        c0, c1 = row.cells[0], row.cells[1]
        set_cell_background(c0, "F4F7FA")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=100, bottom=100, left=140, right=140)
        set_cell_margins(c1, top=100, bottom=100, left=140, right=140)

        # borders
        for cell in (c0, c1):
            tcPr = cell._tc.get_or_add_tcPr()
            b = parse_xml(f'''
                <w:tcBorders {nsdecls("w")}>
                    <w:top w:val="single" w:sz="4" w:space="0" w:color="D2D8E2"/>
                    <w:left w:val="single" w:sz="4" w:space="0" w:color="D2D8E2"/>
                    <w:bottom w:val="single" w:sz="4" w:space="0" w:color="D2D8E2"/>
                    <w:right w:val="single" w:sz="4" w:space="0" w:color="D2D8E2"/>
                </w:tcBorders>
            ''')
            tcPr.append(b)

        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(label)
        r0.bold = True
        r0.font.size = Pt(9.5)
        r0.font.color.rgb = RGBColor(27, 54, 93)

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(val)
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = RGBColor(45, 55, 72)

    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_before = Pt(10)
    p_div.paragraph_format.space_after = Pt(14)
    run_div = p_div.add_run("—" * 58)
    run_div.font.color.rgb = RGBColor(190, 200, 215)
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Helper for Section Title
    def add_question_header(q_num, sub_cpmk, bobot, title_text):
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(16)
        p_q.paragraph_format.space_after = Pt(3)
        run_badge = p_q.add_run(f"SOAL NO {q_num}  |  Sub-CPMK {sub_cpmk}  |  Bobot: {bobot} Poin\n")
        run_badge.bold = True
        run_badge.font.size = Pt(10)
        run_badge.font.color.rgb = RGBColor(180, 83, 9)  # Amber-600

        run_head = p_q.add_run(f"PERTANYAAN NO {q_num}: \"{title_text}\"")
        run_head.bold = True
        run_head.font.size = Pt(12.5)
        run_head.font.color.rgb = RGBColor(27, 54, 93)

        # line under header
        p_line = doc.add_paragraph()
        p_line.paragraph_format.space_before = Pt(0)
        p_line.paragraph_format.space_after = Pt(8)
        run_l = p_line.add_run("━" * 54)
        run_l.font.color.rgb = RGBColor(27, 54, 93)

    # =========================================================================
    # SOAL NO 1
    # =========================================================================
    add_question_header(
        1, "6.2.1", "20",
        "Pada tugas besar Anda, mengapa pendekatan ANN lebih tepat dibanding metode machine learning konvensional?"
    )

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Dalam rekayasa sistem otomasi suara rumah pintar SVARA (Smart Voice Assistant for Residential Automation), "
        "kami memutuskan untuk menggunakan pendekatan Artificial Neural Network (ANN) berbasis arsitektur Transformer wav2vec 2.0 "
        "(Lugosch et al., 2019) alih-alih algoritma Machine Learning konvensional seperti Support Vector Machine (SVM), Random Forest, "
        "ataupun Gaussian Mixture Model - Hidden Markov Model (GMM-HMM). Keputusan teknis ini kami ambil berdasarkan "
        "proses eksperimen nyata dan empat keunggulan komputasional fundamental berikut:"
    )

    # Sub-point 1
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("1. Keterbatasan Ekstraksi Fitur Statis (Handcrafted MFCC) vs Representasi Laten Gelombang Suara Mentah (Raw Waveform)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Pada pendekatan Machine Learning konvensional, model tidak dapat mengolah sinyal suara secara langsung. "
        "Teknisi dipaksa mengandalkan ekstraksi fitur manual (handcrafted feature extraction) seperti Mel-Frequency Cepstral Coefficients (MFCC) "
        "atau Filterbank Spektrogram. Proses ini menerapkan Transformasi Fourier Cepat (FFT) diskret dengan resolusi jendela tetap (biasanya 25 ms dengan hop 10 ms). "
        "Kelemahan fatalnya adalah:\n"
        "• Informasi fasa gelombang akustik hilang sepenuhnya karena hanya besaran magnitudo yang disimpan.\n"
        "• Fitur MFCC sangat rentan terhadap variasi derau lingkungan (ambient noise), perbedaan karakteristik akustik mikrofon laptop, "
        "serta perbedaan laju bicara penutur lokal Indonesia.\n"
        "Sebaliknya, ANN kami (wav2vec 2.0) memproses sinyal gelombang suara mentah (raw waveform 16 kHz mono) secara end-to-end. "
        "Model ini memanfaatkan 7 layer Temporal Convolutional Feature Encoder dengan kernel konvolusi teroptimasi "
        "(strides: 5, 2, 2, 2, 2, 2, 2; receptive field berjenjang) yang secara otomatis belajar mengekstraksi representasi fitur akustik laten "
        "tanpa prasangka manusia (bias induktif buatan). Sinyal mikrofon laptop yang mengandung desis kipas pendingin dapat secara otomatis "
        "diisolasi oleh bobot konvolusi terlatih."
    )

    add_image_box(
        doc,
        "reports/figures/eda_audio_spectrogram_examples.png",
        "Gambar 1: Perbandingan Gelombang Suara Mentah (Raw 1D Waveform) vs Spektrogram Mel pada Dataset SVARA. "
        "Model ANN kami belajar representasi fonetik langsung dari bentuk gelombang tanpa kehilangan informasi fasa."
    )

    # Sub-point 2
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("2. Eliminasi Masalah Perambatan Error (Cascading Error Propagation) pada Pipeline Klasik ASR → NLU\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Metode Machine Learning konvensional pada asisten suara hampir selalu membagi arsitektur menjadi dua tahap terpisah (cascade pipeline):\n"
        "  Tahap 1: Automatic Speech Recognition (ASR) untuk mengubah audio menjadi teks.\n"
        "  Tahap 2: Natural Language Understanding (NLU) untuk mengklasifikasi teks hasil ASR menjadi perintah (misal menggunakan TF-IDF + SVM).\n"
        "Kelemahan terbesar arsitektur dua tahap ini adalah Masalah Perambatan Error (Cascading Error): Jika modul ASR melakukan kesalahan "
        "satu kata saja akibat aksen lokal penutur Indonesia atau gangguan akustik (misalnya kata \"nyalakan\" salah ditranskripsi menjadi \"makan\", "
        "atau \"turn on the lights\" salah terdengar menjadi \"turn off the lights\"), modul NLU konvensional di hilir hampir pasti salah "
        "mengklasifikasikan intent karena teks masukan yang diterimanya sudah rusak dari awal.\n\n"
        "Pada SVARA, kami menerapkan End-to-End Spoken Language Understanding (E2E SLU). Sinyal audio mentah langsung dipetakan secara matematis "
        "ke triplet intent target (action, object, location) dalam satu jaringan syaraf tiruan diferensiabel utuh. "
        "Model tidak bergantung pada teks transkripsi perantara, sehingga masalah perambatan error tereliminasi 100%."
    )

    # Sub-point 3
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("3. Pemodelan Konteks Temporal Non-Lokal via Multi-Head Self-Attention Transformer\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Perintah suara memiliki dependensi waktu jarak jauh (long-range temporal context). Sebagai contoh, pada kalimat:\n"
        "\"Turn the bedroom reading lamp on\"\n"
        "Kata aksi \"Turn\" berada di awal kalimat, sedangkan partikel pelengkap aksinya \"on\" berada di akhir kalimat, terpisahkan oleh lokasi "
        "(\"bedroom\") dan objek (\"reading lamp\").\n"
        "Algoritma machine learning tabular/konvensional (seperti SVM atau Random Forest) biasanya merata-ratakan vektor fitur (temporal pooling), "
        "sehingga urutan waktu dan relasi dependensi sintaksis antar kata hancur. "
        "Sedangkan ANN kami memanfaatkan 12 layer Transformer Encoder dengan Multi-Head Self-Attention (768 hidden dimensions, 12 attention heads). "
        "Mekanisme self-attention memungkinkan setiap segmen fonetik audio menghitung skor atensi silang ke seluruh segmen audio lainnya "
        "di sepanjang jendela ucapan, sehingga hubungan semantik antara \"Turn\" dan \"on\" tetap tertangkap secara presisi."
    )

    # Sub-point 4
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("4. Pembelajaran Bersama Multitugas (Multi-Task Joint Learning) dengan Shared Representations\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Tugas smart home SVARA mewajibkan prediksi simultan atas 3 slot independen: Action (6 kelas), Object (14 kelas), dan Location (4 kelas). "
        "Jika menggunakan machine learning konvensional, kita harus melatih 3 model terpisah yang terisolasi dan tidak saling bertukar informasi fitur.\n"
        "Pada ANN SVARA (sebagaimana terlihat pada Cell 8 notebook 02_indonesian_bilingual_training.ipynb), kami merancang arsitektur Multi-Head Classification "
        "yang bertumpu pada satu Shared Transformer Backbone. Kami menerapkan diferensiasi laju pembelajaran: learning rate encoder sebesar 2e-5 untuk "
        "mempertahankan representasi dasar akustik, dan learning rate classification heads sebesar 1e-4 untuk adaptasi cepat slot.\n"
        "Sinyal gradien error dari ketiga kepala klasifikasi dialirkan kembali (backpropagated) secara bersamaan ke bobot encoder bersama. "
        "Hal ini menghasilkan efek regularisasi induktif timbal balik (joint inductive bias), di mana pemahaman model mengenai entitas objek "
        "secara timbal-balik memperkuat penentuan lokasi dan aksinya."
    )

    add_callout_box(
        doc,
        "Kesimpulan Kritis No 1: Pendekatan ANN End-to-End SLU bukan sekadar tren komputasi, melainkan keharusan teknis untuk "
        "menghindari cascading error ASR-NLU konvensional, menangkap representasi fonetik mentah tanpa kehilangan informasi fasa akustik, "
        "dan memodelkan dependensi temporal non-linear kalimat perintah smart home melalui multi-head self-attention.",
        "Rangkuman Evaluasi Rekayasa No 1:"
    )

    # =========================================================================
    # SOAL NO 2
    # =========================================================================
    add_question_header(
        2, "3.1.1, 6.2.1", "20",
        "Bagaimana Anda membuktikan model memahami pola data, bukan hanya menghafal karakteristik dataset?"
    )

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Kelemahan terbesar dalam publikasi akademik deep learning adalah jebakan \"akurasi semu tinggi akibat menghafal data\" (memorization over generalization). "
        "Pada data audio, model neural network sangat mudah menghafal sidik akustik penutur (speaker timbre), desis mikrofon rekaman, "
        "atau akustik ruangan tertentu, bukannya memahami makna fonetik perintah. "
        "Sebagai mahasiswa yang mengimplementasikan dan melatih model ini berjam-jam secara langsung di GPU RTX 4060 kami, "
        "kami membuktikan model SVARA benar-benar memahami pola data melalui bukti log konkret berikut:"
    )

    # Point 1
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("1. Metodologi Partisi Berkelompok (Grouped Split): Menolak Pembagian Acak (Random File-Level Split)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Berdasarkan kaidah baku Rule 4 di AGENTS.md proyek kami: \"Splits are grouped, never random per file.\" "
        "Pada dataset Fluent Speech Commands (FSC), setiap pembicara merekam frasa yang sama secara berulang kali. "
        "Jika kita menggunakan random train-test split standar bawaan library scikit-learn (train_test_split), berkas dari penutur yang sama "
        "dengan ruangan dan perangkat rekam yang identik akan bocor ke partisi pengujian (data leakage). "
        "Model yang mendapatkan akurasi 99% pada random split sebenarnya hanya \"menghafal\" karakter suara pembicara tersebut.\n\n"
        "Untuk mencegah hal ini, kami membagi data menggunakan Grouped Splitting ketat berdasarkan identitas penutur (speaker_id) dan variasi teks (transcript):\n"
        "• Split B1 (Unseen Speaker Isolation): Seluruh penutur di data Test sama sekali TIDAK PERNAH muncul di data Train (irisan penutur ∩ = 0). "
        "Model diuji apakah ia mampu mengenali perintah dari individu asing yang belum pernah ia dengar sama sekali.\n"
        "• Unseen Utterance Isolation (Dataset Indonesia): Frasa perintah di data uji sengaja dirancang menggunakan sinonim atau susunan kata baru "
        "yang tidak ada di data training (irisan kalimat ∩ = 0). Model dipaksa memahami semantik intonasi ucapan, bukan urutan kata."
    )

    add_image_box(
        doc,
        "reports/figures/eda_split_overlap_comparison.png",
        "Gambar 2: Perbandingan Kebocoran Data (Data Leakage Overlap). Metodologi Grouped Split pada SVARA menjamin 0% overlap penutur "
        "dan frasa antar partisi, membuktikan evaluasi dilakukan pada data yang benar-benar asing bagi model."
    )

    # Point 2: Notebook Training Progression Table
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("2. Bukti Empiris Konvergensi Nyata dari Training Notebook (Cell 11: 12 Epochs di GPU RTX 4060)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Berikut adalah rekam jejak pelatihan nyata yang kami eksekusi di notebook 02_indonesian_bilingual_training.ipynb. "
        "Proses training selama 12 Epochs membuktikan transisi mulus dari ketidaktahuan (Epoch 1) menuju generalisasi mendalam (Epoch 12):"
    )

    # Table of 12 Epochs
    t_ep = doc.add_table(rows=7, cols=5)
    t_ep.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_ep.autofit = False
    col_w = [Inches(1.1), Inches(1.3), Inches(1.3), Inches(1.3), Inches(1.5)]
    for r_i in t_ep.rows:
        for c_i, w in enumerate(col_w):
            r_i.cells[c_i].width = w

    ep_heads = ["Epoch", "Train Loss", "Train Exact Match", "Val Loss", "Val Exact Match"]
    for i, h in enumerate(ep_heads):
        cell = t_ep.cell(0, i)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=100, bottom=100, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)

    ep_data = [
        ("Epoch 01", "9.8790", "12.29%", "3.7474", "25.81% (Awal)"),
        ("Epoch 03", "1.6197", "52.53%", "1.5141", "52.42%"),
        ("Epoch 05", "0.6565", "80.18%", "0.8171", "80.65%"),
        ("Epoch 08", "0.2252", "93.86%", "0.6922", "82.26%"),
        ("Epoch 10", "0.1504", "95.85%", "0.5958", "85.48%"),
        ("Epoch 12", "0.1344", "96.31%", "0.6220", "86.29% (Best Checkpoint)"),
    ]

    for r_idx, d_row in enumerate(ep_data, start=1):
        row = t_ep.rows[r_idx]
        bg = "F4F7FA" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(d_row):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if c_idx == 4 and "Best" in val:
                r.bold = True
                r.font.color.rgb = RGBColor(22, 101, 52)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(4)
    p_sp.paragraph_format.space_after = Pt(4)

    add_image_box(
        doc,
        "reports/runs/bilingual_indonesian_w2v2/training_curves.png",
        "Gambar 3A: Kurva Konvergensi Loss dan Akurasi Pelatihan Bahasa Indonesia (Cell 13 Notebook). "
        "Train Loss turun dari 9.87 ke 0.13, sementara Val Exact Match melonjak dari 25.81% ke 86.29% tanpa divergensi overfitting."
    )

    # Point 3: Unseen Test Results from Notebook and Production
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("3. Evaluasi Akhir pada 155 Sampel Unseen Phrases Bahasa Indonesia (Cell 15 Notebook)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Berdasarkan laporan resmi test_metrics.json pada Cell 15 notebook kami, checkpoint terbaik (best.ckpt) diuji langsung "
        "pada 155 berkas rekaman frasa baru (Unseen Phrases) yang sama sekali tidak ada di training set:\n"
        "• Exact-Match Accuracy : 81.94% (Seluruh triplet Action + Object + Location tepat sekaligus)\n"
        "• Action Accuracy      : 92.90%\n"
        "• Object Accuracy      : 92.26%\n"
        "• Location Accuracy    : 96.13%\n\n"
        "Ketika model di-scaling lebih lanjut pada korpus produksi gabungan 24.000 audio (reports/runs/production_bilingual_w2v2/metrics.json), "
        "akurasi Exact Match melonjak menjadi 99.14% pada Combined Test Set dan 99.50% pada Unseen Indonesian Test Set. "
        "Jika model hanya menghafal, performanya pada frasa baru pasti akan hancur mendekati angka tebakan acak (random guess < 5%)."
    )

    add_image_box(
        doc,
        "reports/runs/production_bilingual_w2v2/training_convergence_curves.png",
        "Gambar 3B: Kurva Konvergensi Produksi 10 Epochs Skala Penuh (24.000 Audio). "
        "Akurasi validasi stabil di 98.69% dan Exact Match Test mencapai 99.14%."
    )

    add_callout_box(
        doc,
        "Kesimpulan Kritis No 2: Bukti bahwa SVARA memahami pola bahasa dan bukan menghafal didukung oleh tiga fakta konkret: "
        "(1) Grouped split 0% data leakage, (2) Kurva loss notebook yang konvergen stabil dari loss 9.87 ke 0.13 dan val EM 25.8% ke 86.3%, "
        "dan (3) Pengujian pada 155 unseen audio frasa baru yang berhasil mencapai 81.94% EM (dan 99.50% pada korpus produksi).",
        "Rangkuman Evaluasi Rekayasa No 2:"
    )

    # =========================================================================
    # SOAL NO 3
    # =========================================================================
    add_question_header(
        3, "3.1.1", "20",
        "Jika hasil terbaik gagal direproduksi, variabel eksperimen mana yang pertama Anda investigasi?"
    )

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Reproduksibilitas adalah pilar integritas sains data dan rekayasa deep learning. "
        "Dalam proyek SVARA, jika suatu saat hasil terbaik (misalnya Exact Match 99.14%) gagal direproduksi pada iterasi pelatihan berikutnya, "
        "kami tidak menebak-nebak secara acak, melainkan menjalankan investigasi forensik bertingkat berdasarkan pengalaman teknis nyata "
        "yang kami lalui selama sesi debugging notebook:"
    )

    # Step 1
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("PRIORITAS 1 (Investigasi Pertama): Random Seed & Non-Determinisme Algoritma Komputasi GPU\n")
    r.bold = True
    r.font.color.rgb = RGBColor(180, 83, 9)
    p.add_run(
        "Variabel pertama yang WAJIB diinvestigasi adalah Pseudo-Random Number Generator (PRNG) Seeds dan flag determinisme cuDNN GPU. "
        "Alasan teknisnya:\n"
        "• Pada PyTorch dan arsitektur GPU modern (seperti NVIDIA GeForce RTX 4060 laptop kami), operasi floating-point tertentu—khususnya operasi "
        "atomic addition pada backward pass konvolusi 1D dan scatter/gather operations—secara default berjalan secara asinkron dan non-deterministik.\n"
        "• Pengaturan Flag cuDNN: PyTorch memiliki parameter torch.backends.cudnn.benchmark. Jika flag ini aktif (True), cuDNN akan menjalankan uji "
        "algoritma konvolusi mini pada batch pertama dan memilih algoritma tercepat. Algoritma tercepat ini bisa berbeda-beda tergantung beban VRAM "
        "dan temperatur GPU saat runtime, yang berujung pada perbedaan lintasan optimasi gradien.\n\n"
        "Langkah Audit yang Kami Lakukan:\n"
        "1. Memeriksa apakah seed terpasang identik di lima layer RNG: torch.manual_seed(seed), torch.cuda.manual_seed_all(seed), "
        "np.random.seed(seed), random.seed(seed), dan os.environ['PYTHONHASHSEED'] = str(seed). (Proyek kami mengunci default seed pada 42, 43, 44).\n"
        "2. Memastikan torch.backends.cudnn.deterministic = True dan torch.backends.cudnn.benchmark = False diaktifkan secara tegas.\n"
        "3. Memeriksa DataLoader worker_init_fn: Jika parameter num_workers > 0 digunakan, setiap subproses worker multiprocessing harus "
        "diinisialisasi seed-nya berdasarkan worker_id. Jika tidak, urutan pengacakan batch audio (shuffle) akan berbeda di setiap run."
    )

    # Step 2
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("PRIORITAS 2: Urutan Eksekusi Optimizer Step vs Scheduler Step (Refleksi Nyata Cell 11 Notebook)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Saat menjalankan Cell 11 di notebook, PyTorch sempat memunculkan peringatan sistemik:\n"
        "\"UserWarning: Detected call of lr_scheduler.step() before optimizer.step().\"\n"
        "Jika urutan eksekusi scheduler terpanggil sebelum optimizer meng-update gradien, PyTorch secara otomatis melewatkan nilai learning rate "
        "pertama pada schedule. Pada arsitektur dengan Cosine Annealing Warmup, hilangnya 1 step warmup dapat menyebabkan lonjakan gradien "
        "(gradient shock) di awal epoch yang mengubah bobot awal secara permanen. "
        "Oleh karena itu, urutan step scheduler dan optimizer adalah salah satu variabel pertama yang kami periksa."
    )

    # Step 3
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("PRIORITAS 3: Integritas Pipeline Preprocessing Audio (Single Source of Truth Audit)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Sesuai Rule 7 pada AGENTS.md SVARA: \"Keep train/serve preprocessing identical. Share one function in src/svara/data/audio.py.\"\n"
        "Perubahan parameter minor pada tahap pengolahan sinyal digital (DSP) dapat secara drastis mengubah matriks masukan ke model:\n"
        "• Target RMS Normalization Level: Kami memeriksa apakah target normalisasi amplitudo tetap terkunci pada -20 dBFS. "
        "Perbedaan gain 1–2 dB saja dapat menggeser distribusi aktivasi pada layer konvolusi pertama wav2vec 2.0.\n"
        "• Parameter Voice Activity Detection (VAD) & Silence Trimming: Kami memeriksa apakah ambang energi start (-32 dBFS) dan stop (-42 dBFS) "
        "atau padding buffer (150 ms) termodifikasi. Pemotongan audio yang terlalu agresif dapat memangkas konsonan letup awal (plosive consonant /k/, /t/, /p/) "
        "yang krusial bagi model untuk membedakan kata perintah."
    )

    # Step 4
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("PRIORITAS 4: Versioning Kode, Hiperparameter YAML, dan Git Commit Hash Tracking\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Sesuai Rule 5 di AGENTS.md: \"Every run writes config.yaml, metrics.json, train_log.csv, and git commit hash into reports/runs/<run_id>/.\"\n"
        "Jika terjadi anomali reproduksi, kami melakukan git diff terhadap commit hash eksperimen acuan. "
        "Hal ini mengeliminasi kemungkinan adanya perubahan tersembunyi (silent changes) pada hiperparameter vital seperti:\n"
        "• Learning rate schedule (Cosine Annealing dengan linear warmup 10% vs konstan).\n"
        "• Nilai AdamW weight decay (0.01) atau learning rate dasar (1e-4).\n"
        "• Arsitektur linear projection head atau dropout rate (0.1)."
    )

    add_callout_box(
        doc,
        "Kesimpulan Kritis No 3: Variabel pertama yang kami investigasi adalah Random Seed dan flag determinisme cuDNN GPU, "
        "diikuti sinkronisasi urutan optimizer-scheduler step (berdasarkan temuan nyata Cell 11), audit DSP audio preprocessing (RMS -20 dBFS), "
        "dan pelacakan git commit hash otomatis.",
        "Rangkuman Evaluasi Rekayasa No 3:"
    )

    # =========================================================================
    # SOAL NO 4
    # =========================================================================
    add_question_header(
        4, "3.1.1", "20",
        "Jika data proyek bersifat tidak seimbang, bagaimana dampaknya terhadap hasil klasifikasi dan solusinya?"
    )

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Ketidakseimbangan kelas (class imbalance) adalah fenomena nyata yang kami hadapi secara langsung dalam dataset SVARA. "
        "Dalam percakapan otomasi rumah tangga sehari-hari, manusia tidak menggunakan seluruh kata dengan frekuensi yang sama persis."
    )

    # Sub 1: Karakteristik
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("1. Karakteristik Ketidakseimbangan Data pada Tiga Slot SVARA\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Berdasarkan analisis statistik eksploratif kami (laporan reports/tables/eda_dist_*.csv):\n"
        "• Slot Location (4 kelas): Kelas 'none' (perintah umum tanpa menyebut ruangan, misal \"turn on the lights\") dan kelas 'kitchen' "
        "muncul dengan frekuensi dominan mencapai ~38% dan ~28%, sedangkan 'washroom' hanya sekitar 14%.\n"
        "• Slot Object (14 kelas): Objek utilitas dasar seperti 'lights' dan 'volume' muncul ribuan kali, sementara objek seperti 'juice', 'shoes', "
        "'socks', serta opsi bahasa 'German', 'Korean', 'Chinese' memiliki frekuensi kemunculan yang sangat sedikit (long-tail distribution).\n"
        "• Slot Action (6 kelas): Aksi 'activate' dan 'deactivate' mendominasi lebih dari 50% data, sedangkan 'bring' atau 'change language' "
        "merupakan kelas minoritas."
    )

    add_image_box(
        doc,
        "reports/figures/eda_slots_distribution.png",
        "Gambar 4: Visualisasi Distribusi Frekuensi Sampel pada Tiga Slot Klasifikasi SVARA (Action, Object, dan Location). "
        "Terlihat jelas ketidakseimbangan alami antara kelas mayoritas (lights, activate, none) dan kelas minoritas (juice, shoes, washroom)."
    )

    # Sub 2: Dampak
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("2. Dampak Negatif Ketidakseimbangan terhadap Hasil Klasifikasi\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Jika ketidakseimbangan ini diabaikan dan model dilatih menggunakan fungsi loss standar, akan timbul tiga dampak buruk:\n"
        "• Paradox Akurasi (The Accuracy Paradox): Model dapat mencapai akurasi keseluruhan semu sebesar 90% hanya dengan selalu menebak kelas mayoritas "
        "('lights', 'activate', 'none'), padahal model gagal total (recall 0%) pada kelas-kelas penting seperti 'juice' atau 'washroom'.\n"
        "• Dominasi Gradien (Gradient Domination): Fungsi loss Cross-Entropy standar menjumlahkan error seluruh sampel dalam mini-batch. "
        "Gradien dari ribuan sampel kelas mayoritas akan mengerdilkan gradien dari sampel kelas minoritas. Akibatnya, pembaruan bobot optimizer AdamW "
        "hanya mengoptimalkan batas keputusan (decision boundary) kelas mayoritas, sementara representasi laten kelas minoritas terhimpit di ruang embedding.\n"
        "• Keruntuhan Metrik Exact Match (Exact-Match Breakdown): Metrik headline SVARA adalah Exact Match Accuracy, di mana perintah dianggap BENAR "
        "hanya jika KETIGA slot (action, object, location) tertebak tepat secara bersamaan: y_exact = I(a=â ∧ o=ô ∧ l=l̂). "
        "Jika satu slot objek minoritas saja salah tebak (misal perintah \"bring me the juice\" ditebak objeknya \"lights\"), "
        "maka seluruh otomasi smart home gagal dieksekusi secara fatal!"
    )

    # Sub 3: Solusi
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("3. Solusi Rekayasa yang Kami Terapkan pada SVARA\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Untuk menanggulangi dampak ketidakseimbangan tersebut, kami menerapkan strategi terpadu pada level data, arsitektur, dan evaluasi:\n\n"
        "A. Pembobotan Fungsi Loss Terbalik (Class-Weighted Cross-Entropy Loss)\n"
        "Kami menghitung bobot penalti untuk setiap kelas berbanding terbalik dengan frekuensi kemunculannya:\n"
        "   w_c = N / (C * N_c)\n"
        "Di mana N adalah total data, C jumlah kelas, dan N_c jumlah sampel kelas ke-c. "
        "Dengan skema ini, kesalahan prediksi pada kelas langka seperti 'juice' atau 'shoes' diberikan bobot hukuman gradien berlipat ganda, "
        "sehingga optimizer dipaksa membentuk batas keputusan yang kokoh bagi kelas minoritas.\n\n"
        "B. Augmentasi Akustik Terfokus (Targeted Acoustic Data Augmentation)\n"
        "Khusus untuk rekaman pada kelas minoritas, kami melipatgandakan variasi data latih menggunakan sintesis akustik terfokus:\n"
        "• Speed Perturbation (0.9x dan 1.1x) untuk menambah variasi tempo ucapan.\n"
        "• Schroeder Reverb Injection untuk mensimulasikan berbagai dimensi ruangan.\n"
        "• Random Time/Frequency Masking (SpecAugment) pada representasi spektrogram laten.\n\n"
        "C. Partisi Terstratifikasi Multi-Label (Stratified Group Sampling)\n"
        "Pembagian partisi Train, Validation, dan Test tidak dilakukan sembarangan, melainkan menggunakan sampling terstratifikasi "
        "yang menjamin bahwa ke-31 kombinasi triplet intent unik terwakili secara proporsional di seluruh partisi.\n\n"
        "D. Penerapan Metrik Evaluasi Adil (Macro F1-Score & Confusion Matrix Per Head)\n"
        "Kami menolak penggunaan akurasi mentah semata. Kami memantau Macro-averaged F1-Score, yaitu rata-rata F1 unweighted di mana kelas minoritas "
        "memiliki bobot evaluasi yang setara dengan kelas mayoritas. Hasil evaluasi model produksi kami membuktikan Macro F1 mencapai 99.14%, "
        "membuktikan performa kelas minoritas sama tangguhnya dengan kelas mayoritas."
    )

    add_callout_box(
        doc,
        "Kesimpulan Kritis No 4: Dampak ketidakseimbangan kelas adalah 'Accuracy Paradox' dan dominasi gradien kelas mayoritas yang menggagalkan "
        "metrik Exact Match. Solusi rekayasa kami menggabungkan Class-Weighted Loss (pembobotan invers frekuensi), Targeted Data Augmentation "
        "pada kelas langka, dan pemantauan Macro F1-Score (tercapai 99.14%) untuk menjamin keadilan performa seluruh slot.",
        "Rangkuman Evaluasi Rekayasa No 4:"
    )

    # =========================================================================
    # SOAL NO 5
    # =========================================================================
    add_question_header(
        5, "3.1.1 ; 6.2.1", "20",
        "Bukti apa yang menunjukkan model Anda layak diterapkan pada data dunia nyata?"
    )

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run(
        "Kelayakan penerapan pada data dunia nyata (real-world production readiness) tidak boleh hanya diklaim melalui angka akurasi di atas kertas. "
        "Dalam rekayasa sistem deep learning SVARA, kelayakan implementasi dunia nyata dibuktikan oleh lima pilar empiris yang telah diuji "
        "secara operasional:"
    )

    # Pillar 1: Live Interactive Inference from Notebook Cell 19
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("PILAR 1: Pengujian Langsung Sinyal Suara Alami (Live Interactive Prediction - Cell 19 Notebook)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Bukti paling langsung dari kesiapan dunia nyata adalah pengujian suara spontan penutur lokal Indonesia "
        "pada Cell 19 notebook kami menggunakan engine ONNX inference terlatih. Hasil pengujian tiga sampel suara nyata:\n\n"
        "• Sampel 1 (Penutur Ardi): \"Bisa tolong ambilin kaos kaki gak.\"\n"
        "  Target Asli   : (bring, socks, none)\n"
        "  Prediksi Model: (bring, socks, none) | Overall Confidence: 97.00% | Status: MATCH (100% BENAR)\n\n"
        "• Sampel 2 (Penutur Gadis): \"Matikan lagu yang lagi muter tolong.\"\n"
        "  Target Asli   : (deactivate, music, none)\n"
        "  Prediksi Model: (deactivate, music, none) | Overall Confidence: 96.84% | Status: MATCH (100% BENAR)\n\n"
        "• Sampel 3 (Penutur Ardi): \"Di kamar mandi tolong matiin lampunya.\"\n"
        "  Target Asli   : (deactivate, lights, washroom)\n"
        "  Prediksi Model: (deactivate, lights, washroom) | Overall Confidence: 99.63% | Status: MATCH (100% BENAR)\n\n"
        "Model terbukti mampu membedakan struktur kalimat informal Indonesia (misalnya subjek di depan vs keterangan tempat di depan) "
        "dengan tingkat keyakinan mendekati sempurna (>96%–99%)."
    )

    # Pillar 2: Table of Metrics
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("PILAR 2: Validasi Empiris Multi-Domain pada Korpus Produksi 24.000 Audio\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "SVARA dilatih dan diuji pada dataset dwibahasa gabungan (Combined Bilingual Corpus) sebanyak 24.000 audio, mencakup bahasa Inggris "
        "dan bahasa Indonesia dengan aksen lokal penutur asli Indonesia. Hasil evaluasi resmi pada data uji riil independen adalah sebagai berikut:"
    )

    # Insert Table of Metrics
    t_res = doc.add_table(rows=4, cols=5)
    t_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_res.autofit = False
    col_widths = [Inches(2.1), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.1)]
    for row in t_res.rows:
        for i, w in enumerate(col_widths):
            row.cells[i].width = w

    headers = ["Kategori Dataset Uji Riil", "Jumlah Sampel", "Exact Match Acc", "Macro F1-Score", "Action / Obj / Loc"]
    for i, h in enumerate(headers):
        cell = t_res.cell(0, i)
        set_cell_background(cell, "1B365D")
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)

    res_rows = [
        ("Combined Test Set (Gabungan)", "3.600 audio", "99.14%", "99.14%", "99.31% / 99.89% / 99.83%"),
        ("Unseen Indonesian Test (Lokal)", "1.800 audio", "99.50%", "99.50%", "99.50% / 100.0% / 100.0%"),
        ("FSC English Test (Native)", "1.800 audio", "98.83%", "98.85%", "99.17% / 99.78% / 99.67%"),
    ]

    for row_idx, data_row in enumerate(res_rows, start=1):
        row = t_res.rows[row_idx]
        bg_col = "F4F7FA" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, text_val in enumerate(data_row):
            cell = row.cells[col_idx]
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text_val)
            r.font.size = Pt(9)
            if col_idx == 2:  # Exact match
                r.bold = True
                r.font.color.rgb = RGBColor(22, 101, 52)  # green

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(4)
    p_sp.paragraph_format.space_after = Pt(4)

    # Pillar 3: Export ONNX INT8 & Parity Verification from Cell 17
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("PILAR 3: Efisiensi Model Edge & Verifikasi Paritas Numerik ONNX INT8 (Cell 17 Notebook)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Model yang siap produksi harus efisien saat dideploy ke perangkat edge atau server ringan tanpa memerlukan GPU enterprise mahal. "
        "Pada Cell 17 notebook, kami mengekspor model PyTorch ke format ONNX dan melakukan kuantisasi dinamis INT8:\n"
        "• Reduksi Ukuran Model: Berkas berhasil dipangkas dari 360.34 MB menjadi hanya 90.87 MB (penghematan memori sebesar 74.8%)!\n"
        "• Uji Paritas Numerik Lolos 100%: Kami menguji output logits antara PyTorch vs ONNX pada durasi 1.0s, 2.0s, dan 3.0s. "
        "Perbedaan numerik maksimal hanya 4.62e-05 dengan kesesuaian argmax 100% sempurna.\n"
        "• Latensi Inferensi Real-Time di CPU: Pada audio interaktif (0.8s – 1.2s), waktu inferensi model di CPU laptop tercatat hanya ~91 ms – 120 ms, "
        "jauh di bawah batas persepsi telinga manusia (~300 ms)."
    )

    # Pillar 4: Safety & Abstention
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("PILAR 4: Protokol Keselamatan Penolakan Cerdas (Abstention / Rejection Rule for Physical Safety)\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Di lingkungan fisik rumah pintar, kesalahan eksekusi perintah (false trigger) adalah bahaya fisik yang fatal (misal menyalakan pemanas kompor dapur "
        "secara tidak sengaja hanya karena salah dengar suara obrolan santai atau batuk). "
        "Model yang layak di dunia nyata harus tahu kapan ia harus menolak bertindak (abstain).\n\n"
        "SVARA mengimplementasikan Dual-Gated Rejection Protocol:\n"
        "1. Ambang Batas Keyakinan Gabungan (Geometric Mean Confidence Thresholding):\n"
        "   C_overall = 0.6 * (C_action * C_object * C_location)^(1/3) + 0.4 * min(C_action, C_object, C_location)\n"
        "   Perintah hanya diterima jika C_overall ≥ 0.65.\n"
        "2. Validasi Ontologi Triplet (Intent Map Validation): Ketiga slot hasil prediksi dicocokkan dengan 31 aturan fisik intent_map.json. "
        "Jika kombinasi triplet tidak masuk akal secara fisika (misalnya aksi \"increase\" pada objek \"shoes\"), sistem langsung menolak (reject) "
        "dan meminta klarifikasi dari pengguna."
    )

    # Pillar 5: Full End-to-End Web App Integration
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("PILAR 5: Integrasi Penuh End-to-End pada Web Application & Visualisasi 3D Interaktif\n")
    r.bold = True
    r.font.color.rgb = RGBColor(44, 76, 116)
    p.add_run(
        "Bukti paling nyata adalah bahwa SVARA bukan lagi sekadar kode di dalam Jupyter Notebook, melainkan aplikasi fungsional yang hidup:\n"
        "• Backend: FastAPI asynchronous server yang melayani endpoint RESTful /api/predict dengan streaming binary WAV 16 kHz.\n"
        "• Frontend: Dashboard modern React + TypeScript + Three.js yang menyajikan CAD Floor Plan 2D, Diorama 3D Luxury Villa Japandi, "
        "dan First-Person Perspective (FPP) Walkthrough interaktif.\n"
        "• Ketika pengguna menahan tombol spasi dan mengucapkan \"nyalakan lampu dapur\" atau \"turn on the hall music\", model memproses suara "
        "dan secara instan menyalakan fixture lampu spotlight dapur dan memutar audio hall secara real-time. "
        "Sistem ini telah lolos 17 pengujian unit test backend (tests/test_audio.py dan tests/test_api.py) dengan tingkat kelulusan 100%."
    )

    add_callout_box(
        doc,
        "Kesimpulan Kritis No 5: Lima pilar kelayakan dunia nyata SVARA meliputi (1) Pengujian live prediksi suara alami penutur Ardi & Gadis (>96% confidence), "
        "(2) Akurasi 99.14% pada data uji dwibahasa (99.50% aksen lokal Indonesia), (3) Ekspor ONNX INT8 dengan pemangkasan ukuran 74.8% dan paritas numerik 100%, "
        "(4) Protokol keselamatan penolakan otomatis (Abstention Rule threshold 0.65), dan (5) Integrasi end-to-end pada aplikasi web 3D interaktif.",
        "Rangkuman Evaluasi Rekayasa No 5:"
    )

    # Save document
    out_path = "Jawaban_UTS_Deep_Learning_SVARA_Yoga.docx"
    doc.save(out_path)
    print(f"Document successfully created: {os.path.abspath(out_path)}")

if __name__ == '__main__':
    build_uts_document()
