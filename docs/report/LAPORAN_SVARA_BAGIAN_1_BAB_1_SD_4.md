# DOKUMEN MASTER LAPORAN TUGAS BESAR — BAGIAN 1 (IDENTITAS, ABSTRAK, BAB 1 s.d. BAB 4)

---

## IDENTITAS PROYEK
* **Judul Proyek**: SVARA: Pengembangan Sistem Spoken Language Understanding Berbasis Arsitektur Wav2Vec 2.0 Multi-Head Classifier untuk Pengendalian Perangkat Rumah Pintar Bilingual Real-Time
* **Nama Mahasiswa**: 
  1. Yoga Gusti Rachmadi (Lead Machine Learning Engineer & Full-Stack Developer)
  2. Haikal (Acoustic Data Engineer & QA Field Tester)
* **NIM**: 
  1. [Isikan NIM Yoga]
  2. [Isikan NIM Haikal]
* **Program Studi**: Teknik Informatika
* **Fakultas**: Ilmu Komputer
* **Perguruan Tinggi**: Universitas Mercu Buana
* **Mata Kuliah**: Pembelajaran Mendalam (*Deep Learning*)
* **Dosen Pengampu**: [Isikan Nama Dosen Pengampu & Gelar]
* **Semester / Tahun Akademik**: Ganjil / 2025–2026
* **Tautan Repositori GitHub**: https://github.com/YogUNI/SVARA
* **Tautan Video Demonstrasi**: [Isikan Tautan Video YouTube/Drive]

---

## ABSTRAK

Sistem kendali rumah pintar berbasis suara konvensional umumnya mengandalkan arsitektur *cascaded* dua tahap, yaitu *Automatic Speech Recognition* (ASR) untuk mengubah sinyal audio menjadi transkripsi teks, diikuti oleh modul *Natural Language Understanding* (NLU) untuk mengekstraksi maksud pengguna (*intent*). Pendekatan berjenjang ini memiliki kelemahan mendasar berupa perambatan galat (*error propagation*), latensi komputasi tinggi, hilangnya isyarat paralinguistik akustik, serta beban komputasi yang berat bagi implementasi *edge hardware*. Penelitian ini mengusulkan **SVARA** (*Smart Voice Assistant for Residential Automation*), sebuah sistem kendali rumah pintar berbasis *End-to-End Spoken Language Understanding* (E2E SLU) yang memetakan sinyal wicara mentah langsung ke dalam tiga slot semantik aksi rumah pintar (*action*, *object*, *location*) secara simultan (*multi-head classification*).

Sistem dibangun dengan memanfaatkan representasi akustik *self-supervised learning* dari model fondasi `facebook/wav2vec2-base` yang digabungkan dengan tiga kepala klasifikasi *linear projection* yang terisolasi. Korpus eksperimental dibangun secara seimbang (*balanced bilingual*) sebanyak 24.000 klip wicara (12.000 bahasa Indonesia dan 12.000 bahasa Inggris) yang merepresentasikan 31 *intent* kanonikal rumah pintar, serta diperkuat dengan 124 klip rekaman vokal biologis manusia (*acoustic anchor*) yang mencakup variasi tutur formal, santai, hingga kolokial (*slang*). Evaluasi empiris pada pengujian *unseen-utterance* bahasa Indonesia menghasilkan akurasi *Exact Match* sebesar **99,50%**, dan pada *Fluent Speech Commands* (FSC) bahasa Inggris sebesar **98,83%**. Melalui teknik *acoustic anchor injection* dan *fine-tuning* terarah (10 *epochs* menggunakan *PyTorch Automatic Mixed Precision* FP16 pada GPU NVIDIA GeForce RTX 4060), sistem mencapai konvergensi optimal dengan *Exact Match* validasi sebesar **100,00%**. Untuk kebutuhan penerapan industri, graf komputasi diekspor ke Universal ONNX FP32 dan dioptimasi menggunakan *Dynamic INT8 Quantization*, berhasil mereduksi ukuran model sebesar **3,97 kali lipat (dari 360,34 MB menjadi 90,87 MB)** dengan disparitas numerik maksimal $\le 10^{-6}$ (*zero degradation*) dan latensi inferensi rata-rata 1,5 detik pada CPU server. Pengujian lapangan nyata (*real-world field test*) pada 50 sampel audio dengan gangguan derau ruangan menghasilkan akurasi *exact match* sebesar **92,00%** dengan *acceptance rate* **100,00%**. Hasil ini membuktikan bahwa arsitektur E2E SLU berbasis wav2vec 2.0 terkuantisasi sangat andal, hemat sumber daya, dan tahan terhadap variasi dialek lokal untuk automasi hunian cerdas.

**Kata kunci**: *Spoken Language Understanding*, *End-to-End SLU*, *wav2vec 2.0*, *Multi-Head Classification*, Rumah Pintar, Kuantisasi Dinamis INT8, *Acoustic Anchor*.

---

## ABSTRACT

*Conventional voice-controlled smart home systems predominantly employ a cascaded two-stage architecture: Automatic Speech Recognition (ASR) to transcribe acoustic signals into textual tokens, followed by a Natural Language Understanding (NLU) module to infer semantic intent. This cascaded paradigm inherently suffers from cumulative error propagation, elevated computational latency, loss of rich acoustic paralinguistic cues, and heavy memory footprints unsuitable for edge deployments. This research proposes SVARA (Smart Voice Assistant for Residential Automation), an End-to-End Spoken Language Understanding (E2E SLU) framework that directly maps raw continuous audio waveforms into three concurrent smart-home semantic slots (action, object, location) via an isolated multi-head classifier architecture.*

*The proposed architecture harnesses self-supervised acoustic representations from the foundational `facebook/wav2vec2-base` network paired with three dedicated task heads. The experimental corpus comprises a balanced bilingual dataset of 24,000 speech utterances (12,000 Indonesian and 12,000 English clips) covering 31 canonical home automation intents, hardened with 124 biological human voice recordings (acoustic anchors) encapsulating formal, colloquial, and slang speech styles. Empirical evaluation on strictly isolated unseen Indonesian test utterances yields a 99.50% Exact Match accuracy, while achieving 98.83% on the Fluent Speech Commands (FSC) English benchmark. Utilizing targeted acoustic anchor injection and 10 epochs of PyTorch AMP FP16 retraining on an NVIDIA GeForce RTX 4060 GPU, the model achieves complete convergence with a 100.00% validation Exact Match rate. For production deployment, the computational graph is exported to Universal ONNX FP32 and compressed via Dynamic INT8 Quantization, achieving a 3.97x size reduction (compressing 360.34 MB down to 90.87 MB) with a negligible numerical deviation of $\le 10^{-6}$ and an average CPU inference latency of 1.5 seconds. Real-world field testing under ambient household acoustic noise across 50 live samples demonstrates a 92.00% exact match accuracy with a 100.00% operational acceptance rate, confirming that quantized E2E SLU provides a resilient, resource-efficient, and accent-tolerant foundation for intelligent residential automation.*

*Keywords: Spoken Language Understanding, End-to-End SLU, wav2vec 2.0, Multi-Head Classification, Smart Home, Dynamic INT8 Quantization, Acoustic Anchor.*

---

# BAB 1. PENDAHULUAN

## 1.1 Latar Belakang
Perkembangan teknologi Internet of Things (IoT) dan hunian cerdas (*smart home*) telah mengubah interaksi manusia dengan lingkungan tempat tinggal secara fundamental. Perangkat elektronik rumah tangga seperti sistem pencahayaan, pengatur suhu udara (*air conditioner*), peranti hiburan audio-visual, hingga perangkat keamanan kini dapat dioperasikan secara terpadu melalui jaringan nirkabel. Kendati demikian, antarmuka kontrol berbasis aplikasi telepon pintar (*smartphone graphical user interface*) kerap dinilai kurang ergonomis ketika penghuni rumah sedang melakukan aktivitas lain (*hands-busy, eyes-busy*), seperti memasak, bekerja, atau mengasuh anak. Dalam konteks tersebut, interaksi berbasis suara (*voice user interface*) menjadi modalitas kontrol yang paling alami, intuitif, dan inklusif, khususnya bagi lansia dan penyandang disabilitas motorik.

Secara konvensional, asisten suara komersial dibangun menggunakan arsitektur pipa berjenjang (*cascaded pipeline*). Pipa ini terdiri atas modul *Automatic Speech Recognition* (ASR) berskala besar yang mentranskripsikan sinyal wicara menjadi rangkaian kata teks, yang kemudian diteruskan ke modul *Natural Language Understanding* (NLU) untuk melakukan klasifikasi maksud (*intent classification*) dan pengisian slot (*slot filling*). Walaupun pendekatan ini telah mapan, arsitektur berjenjang memiliki tiga kelemahan struktural yang signifikan:
1. **Perambatan Galat Kumulatif (*Error Propagation*)**: Kesalahan transkripsi fonem atau kata pada tahap ASR—terutama akibat derau lingkungan (*ambient noise*), jeda artikulasi, atau aksen lokal—secara otomatis menyesatkan penalaran semantik modul NLU, sehingga perintah gagal dieksekusi meskipun modul NLU memiliki akurasi teoritis yang tinggi.
2. **Hilangnya Isyarat Paralinguistik dan Akustik**: Transformasi audio menjadi teks membuang informasi akustik yang kaya, seperti intonasi, prosodi, penekanan nada, dan kecepatan berbicara, yang sebenarnya krusial dalam memahami intensitas perintah.
3. **Beban Komputasi dan Latensi Tinggi**: Mengoperasikan dua model *deep learning* berukuran masif (model akustik ASR puluhan juta parameter ditambah model bahasa NLU seperti BERT) membutuhkan komputasi ganda yang menghasilkan latensi di atas 3–5 detik, ketergantungan mutlak pada komputasi awan (*cloud dependency*), serta risiko kebocoran privasi rekaman suara pribadi di peladen pihak ketiga.

Sebagai paradigma alternatif yang mutakhir, pendekatan *End-to-End Spoken Language Understanding* (E2E SLU) hadir dengan memetakan gelombang suara mentah secara langsung (*audio-to-intent*) tanpa menghasilkan representasi teks perantara. Pendekatan ini secara inheren mengeliminasi perambatan galat ASR, memangkas latensi inferensi, dan memungkinkan model mengeksploitasi fitur akustik secara utuh. Terobosan representasi wicara *self-supervised learning* seperti wav2vec 2.0 (Baevski et al., 2020) membuktikan bahwa pemodelan wicara yang dilatih pada ribuan jam audio tanpa label mampu menangkap struktur fonetik dan semantik bahasa secara luar biasa, sehingga hanya membutuhkan adaptasi bobot (*fine-tuning*) yang efisien untuk tugas hilir (*downstream tasks*).

Namun, penelitian SLU yang ada saat ini menghadapi tantangan besar saat dihadapkan pada realitas empiris di Indonesia. Pertama, mayoritas korpus acuan global seperti *Fluent Speech Commands* (Lugosch et al., 2019) dan SLURP (Bastianelli et al., 2020) berfokus secara eksklusif pada penutur bahasa Inggris dengan artikulasi terisolasi. Kedua, penutur bahasa Indonesia sehari-hari memiliki variasi register tutur yang sangat heterogen—mulai dari ragam formal (*baku*), santai (*nyantai*), hingga bahasa gaul (*slang/colloquial*) dengan serapan kata partikel lokal (seperti "nyalain", "matiin", "dong", "sih", "idupin"). Ketiga, model SLU skala laboratorium kerap mengalami penurunan performa drastis (*performance degradation*) saat diuji pada perangkat keras berdaya rendah akibat tingginya konsumsi memori model representasi wicara (berkisar 360 MB hingga 1 GB).

Menjawab kesenjangan tersebut, penelitian ini mengembangkan **SVARA** (*Smart Voice Assistant for Residential Automation*). SVARA mengintegrasikan model fondasi representasi wicara wav2vec 2.0 dengan arsitektur *Multi-Head Classification* yang memisahkan klasifikasi aksi (*action*), objek (*object*), dan lokasi (*location*). Sistem diperkuat dengan korpus dwibahasa terisolasi 24.000 sampel, teknik *acoustic anchor injection* vokal biologis penutur Indonesia, optimasi kuantisasi *Dynamic INT8 Quantization* (ONNX Runtime) berukuran 90,87 MB, serta antarmuka digital kembar 3D (*3D digital twin*) interaktif untuk simulasi kendali hunian cerdas secara langsung.

## 1.2 Rumusan Masalah
Berdasarkan latar belakang di atas, rumusan masalah dalam penelitian tugas besar ini dirinci sebagai berikut:
1. Bagaimana merancang arsitektur model *deep learning* berbasis *End-to-End Spoken Language Understanding* (E2E SLU) multi-head classifier yang mampu mengekstraksi slot aksi, objek, dan lokasi rumah pintar secara akurat langsung dari gelombang audio mentah?
2. Bagaimana membangun dan memvalidasi korpus wicara perintah rumah pintar dwibahasa (Indonesia–Inggris) yang seimbang serta bebas dari kebocoran data (*zero data leakage*) pada pengujian tuturan baru (*unseen utterances*) dan penutur baru (*unseen speakers*)?
3. Sejauh mana teknik *acoustic anchor injection* menggunakan rekaman vokal manusia nyata dapat meningkatkan ketahanan (*robustness*) model terhadap variasi gaya tutur bahasa Indonesia (formal, santai, dan *slang*)?
4. Bagaimana mengoptimasi model SLU menggunakan *Dynamic INT8 Quantization* (ONNX Runtime) agar dapat dijalankan pada CPU lokal dengan ukuran berkas minimal, latensi rendah, dan tanpa mengalami degradasi presisi numerik?
5. Bagaimana performa sistem SVARA ketika diuji secara langsung (*real-world field test*) di bawah pengaruh derau akustik lingkungan perumahan nyata?

## 1.3 Tujuan Proyek
Tujuan utama yang ingin dicapai melalui penelitian proyek tugas besar ini adalah:
1. Mengimplementasikan arsitektur *End-to-End SLU* berbasis *backbone* `facebook/wav2vec2-base` dengan tiga kepala klasifikasi terpisah (*multi-head classification*) untuk slot *action* (6 kelas), *object* (14 kelas), dan *location* (4 kelas) yang mencakup 31 kombinasi *intent* kanonikal.
2. Membangun korpus dwibahasa SVARA 24.000 klip audio (12.000 klip wicara bahasa Indonesia dan 12.000 klip wicara bahasa Inggris FSC) dengan protokol pemisahan data ketat (*speaker-disjoint* dan *unseen-utterance split*) yang terverifikasi secara formal bebas kebocoran (*zero leakage audit*).
3. Mengevaluasi dampak *acoustic anchor injection* dari 124 klip vokal manusia penutur Indonesia terhadap stabilitas konvergensi model dan akurasi *Exact Match* pada gaya tutur santai dan gaul.
4. Mengekspor graf komputasi PyTorch ke format Universal ONNX FP32 dan melakukan kompresi *Dynamic INT8 Quantization* untuk mencapai rasio kompresi minimal 3x lipat dengan margin galat numerik $\le 10^{-4}$.
5. Mengintegrasikan model inferensi terkuantisasi ke dalam arsitektur peladen backend FastAPI dan aplikasi web frontend berbasis React + Three.js 3D Smart Home, serta mengukur latensi end-to-end dan akurasi pada pengujian lapangan nyata.

## 1.4 Manfaat Proyek
Penelitian ini diharapkan memberikan kontribusi nyata dari sudut pandang akademis maupun praktis:
* **Manfaat Teoretis / Akademis**:
  1. Memberikan kontribusi empiris mengenai penerapan representasi wicara *self-supervised learning* (wav2vec 2.0) pada tugas *Spoken Language Understanding* untuk bahasa non-Inggris (bahasa Indonesia) dengan register tutur kolokial.
  2. Menyediakan metodologi evaluasi yang jujur (*honest evaluation protocol*) melalui isolasi tuturan baru (*unseen utterance partition*) untuk membuktikan kemampuan generalisasi semantik akustik yang sesungguhnya.
* **Manfaat Praktis / Terapan**:
  1. **Efisiensi Komputasi & Kemandirian Lokal**: Model SLU terkuantisasi (90,87 MB) dapat dioperasikan secara lokal pada perangkat komputer personal atau komputer papan tunggal (*single-board computer*) tanpa membutuhkan kartu grafis (GPU) berdaya tinggi saat inferensi, menghemat biaya operasional peladen awan.
  2. **Privasi & Keamanan Data Penghuni**: Pemrosesan perintah suara secara lokal (*on-device/local server inference*) menjamin bahwa percakapan pribadi penghuni rumah tidak pernah dikirim ke internet, mengeliminasi risiko intersepsi data privasi.
  3. **Aksesibilitas Hunian Cerdas**: Memberikan antarmuka kendali yang ramah bagi pengguna domestik di Indonesia yang terbiasa berkomunikasi dengan bahasa sehari-hari tanpa harus menghafal instruksi berbahasa Inggris formal.

---

# BAB 2. STUDI LITERATUR

## 2.1 Teori Dasar

### 2.1.1 Paradigma Spoken Language Understanding (SLU)
*Spoken Language Understanding* (SLU) merupakan cabang kecerdasan buatan pada irisan antara pemrosesan wicara (*speech processing*) dan pemrosesan bahasa alami (*natural language processing*). Tugas utama sistem SLU adalah memetakan sinyal akustik wicara input $X = (x_1, x_2, \dots, x_T)$ secara langsung atau tidak langsung menjadi representasi semantik terstruktur $Y = (y_1, y_2, \dots, y_K)$. Dalam sistem kendali rumah pintar berbasis domain *Fluent Speech Commands* (Lugosch et al., 2019), semantik instruksi direpresentasikan sebagai tupel slot bernilai:
$$Y = (\text{action}, \text{object}, \text{location})$$
di mana kombinasi unik ketiga slot tersebut mendefinisikan sebuah intensi kanonikal (*canonical intent*).

### 2.1.2 Perbandingan Arsitektur Berjenjang (*Cascaded*) vs *End-to-End* (E2E)
Pendekatan komputasi SLU terbagi menjadi dua paradigma utama:
1. **Cascaded Architecture (ASR $\rightarrow$ NLU)**:
   Sistem memecah pemrosesan menjadi dua tahap independen:
   $$\hat{W} = \arg\max_W P(W \mid X) \quad [\text{Modul ASR}]$$
   $$\hat{Y} = \arg\max_Y P(Y \mid \hat{W}) \quad [\text{Modul NLU}]$$
   Kelemahan matematis dari formulasi ini adalah minimnya optimasi terpadu (*lack of joint optimization*). Apabila hipotesis teks $\hat{W}$ mengandung substitusi atau penghapusan kata kunci akibat derau mikrofon, probabilitas posterior $P(Y \mid \hat{W})$ akan menghasilkan estimasi slot semantik yang salah secara fatal.
2. **End-to-End Architecture (Audio $\rightarrow$ Semantics)**:
   Sistem mengestimasi distribusi posterior slot semantik secara terpadu langsung dari representasi fitur gelombang suara:
   $$\hat{Y} = \arg\max_Y P(Y \mid X)$$
   Model dilatih secara holistik untuk meminimalkan fungsi kerugian semantik (*semantic loss*), memungkinkan jaringan mempertahankan isyarat akustik intonasional serta menghindari ketergantungan pada leksikon teks perantara yang kaku.

### 2.1.3 Representasi Wicara Berbasis Wav2Vec 2.0
Wav2Vec 2.0 (Baevski et al., 2020) adalah kerangka kerja representasi wicara *self-supervised learning* yang memproses audio mentah melalui tiga komponen utama:
1. **Feature Encoder (CNN)**: Terdiri atas lapisan konvolusi satu dimensi temporal dengan blok normalisasi lapisan (*layer normalization*) dan fungsi aktivasi GELU. *Encoder* ini memproyeksikan sinyal audio mentah berkas 16 kHz berdurasi $T$ sampel menjadi representasi laten $Z = (z_1, z_2, \dots, z_{T'})$ dengan resolusi temporal setiap 20 ms dan bidang reseptif 25 ms.
2. **Context Network (Transformer)**: Mengadopsi arsitektur *Transformer encoder* (Vaswani et al., 2017) dengan mekanisme *multi-head self-attention*. Lapisan ini membangun representasi kontekstual $C = (c_1, c_2, \dots, c_{T'})$ yang mengintegrasikan informasi akustik sepanjang urutan ujaran.
3. **Quantization Module**: Mengubah representasi laten $Z$ menjadi representasi diskret $Q$ melalui teknik *Gumbel-Softmax vector quantization* yang digunakan pada tahap pra-latih (*pre-training*) dengan fungsi objektif kontrastif (*contrastive loss*) dan *diversity loss*.

Keunggulan memanfaatkan bobot *pre-trained* `wav2vec2-base` (dilatih pada 960 jam audio LibriSpeech) adalah kemampuannya mentransfer pengetahuan fonetik universal ke dalam tugas hilir dengan data beranotasi terbatas (*low-resource transfer learning*).

### 2.1.4 Kuantisasi Dinamis INT8 (*Dynamic Quantization*)
Pada penerapan inferensi di lingkungan produksi, bobot matriks komputasi jaringan saraf tiruan yang awalnya disimpan dalam presisi floating-point 32-bit (FP32) dapat dikompresi menjadi integer 8-bit (INT8) menggunakan formulasi pemetaan linier afinitas:
$$q = \text{round}\left(\frac{x}{S}\right) + Z$$
di mana $S$ merupakan faktor skala (*scale factor*) dan $Z$ adalah titik nol (*zero-point offset*). Pada *Dynamic INT8 Quantization*, bobot model (*weights*) diubah secara statis menjadi INT8 saat penyimpanan, sedangkan tensor aktivasi dinilai secara dinamis saat proses komputasi berlangsung (*runtime activation quantization*). Metode ini mampu menghemat konsumsi memori hingga 75% (kompresi 4x) dan mempercepat operasi perkalian matriks (*GEMM operations*) menggunakan instruksi vektor CPU modern (AVX-512 / VNNI).

## 2.2 Penelitian Terkait
1. **Lugosch et al. (2019)** memperkenalkan dataset *Fluent Speech Commands* (FSC) dan mengusulkan arsitektur E2E SLU berbasis konvolusi berulang (*Recurrent Neural Network* - RNN) dan *Transformer*. Hasil penelitian mereka menunjukkan bahwa model E2E mampu mencapai akurasi *exact match* di atas 98% pada pemisahan acak (*random split*), namun mengalami degradasi akurasi saat diuji pada skenario penutur yang tidak dikenal (*unseen speaker*). Kelemahan penelitian tersebut adalah pengujian belum mengevaluasi ragam bahasa non-Inggris dan variasi tuturan informal.
2. **Bastianelli et al. (2020)** merilis korpus SLURP (*Spoken Language Understanding Resource Package*) yang mencakup variasi kalimat alami yang lebih kompleks. Penelitian mereka menggarisbawahi bahwa model bertingkat ASR-NLU mengalami penurunan akurasi drastis sebesar 14,2% saat menghadapi derau lingkungan akustik nyata, sementara pendekatan E2E menunjukkan resiliensi yang jauh lebih stabil.
3. **Radfar et al. (2020)** meneliti ketahanan sistem SLU terhadap derau ruangan (*ambient household noise*) dan gangguan gema dinding (*reverberation*). Mereka menyimpulkan bahwa augmentasi data akustik menggunakan konvolusi *Room Impulse Response* (RIR) dan injeksi derau aditif sintetis sangat krusial agar model SLU tidak mengalami *overfitting* terhadap rekaman studio yang bersih.
4. **Wahyuni & Wijaya (2022)** mengeksplorasi pengenalan wicara bahasa Indonesia menggunakan arsitektur deep learning konvensional, namun penelitian tersebut masih terbatas pada transkripsi fonem ASR dan belum mengintegrasikan pemahaman semantik terpadu (*SLU end-to-end*).

## 2.3 Research Gap
Berdasarkan tinjauan pustaka di atas, teridentifikasi sejumlah kesenjangan riset (*research gap*) yang mendasari urgensi proyek SVARA:
1. **Ketiadaan Korpus SLU Terpadu Dwibahasa Indonesia–Inggris**: Belum tersedia korpus publik teranotasi slot rumah pintar yang mencakup variasi tuturan bahasa Indonesia ragam santai (*nyantai*) dan gaul (*slang*) yang dipasangkan secara seimbang dengan standar FSC global.
2. **Uji Validasi Kebocoran Data (*Data Leakage*) yang Belum Ketat**: Sebagian besar penelitian terdahulu menggunakan pemisahan data berbasis berkas acak (*random file-level split*). Karena setiap penutur merekam kalimat berulang kali, pemisahan acak menyebabkan kebocoran identitas vokal dan frasa kalimat (*acoustic & lexical leakage*), menghasilkan performa model yang *overoptimistic*. Diperlukan pemisahan berbasis *strictly unseen utterance* dan *speaker-disjoint*.
3. **Kesenjangan antara Model Laboratorium dan Komputasi Produksi Rendah Daya**: Sebagian besar publikasi menyajikan performa model FP32 pada kartu grafis kelas data center, tanpa menyertakan analisis kompresi graf, latensi CPU, dan disparitas numerik pasca-kuantisasi untuk implementasi rumah pintar lokal yang terjangkau.
4. **Ketiadaan Validasi Lapangan Multi-Gaya Tutur**: Sangat sedikit penelitian yang menyandingkan hasil akurasi model sintetis dengan pengujian rekaman vokal biologis manusia nyata (*acoustic anchor*) secara *real-time* di lingkungan perumahan tropis.

---

# BAB 3. DATASET DAN PERANCANGAN SISTEM

## 3.1 Deskripsi Dataset

### 3.1.1 Dataset Acuan Fluent Speech Commands (FSC)
Dataset FSC (Lugosch et al., 2019) terdiri atas 30.043 rekaman audio wicara perintah rumah pintar berformat WAV (16 kHz, 16-bit mono) dari 97 penutur bahasa Inggris dengan 248 variasi frasa unik. Setiap rekaman dilabeli tiga slot semantik teranotasi:
* `action`: tindakan yang diminta (misal: *activate*, *deactivate*, *increase*, *decrease*, *bring*, *change language*).
* `object`: peranti atau entitas sasaran (misal: *lights*, *music*, *heat*, *lamp*, *volume*, *juice*, *shoes*, dll.).
* `location`: ruangan spesifik (misal: *kitchen*, *bedroom*, *washroom*, atau *none* jika universal).

### 3.1.2 Korpus Sintetis dan Kolokial SVARA Bahasa Indonesia
Untuk membangun kapabilitas dwibahasa, dikembangkan korpus wicara bahasa Indonesia yang dipetakan secara simetris terhadap 31 *intent* FSC. Korpus ini mencakup:
* **Ekspansi Frasa Alami**: 4.358 variasi kalimat bahasa Indonesia yang dirancang mencakup gaya formal ("Nyalakan lampu kamar tidur"), santai ("Tolong hidupin lampu di kamar tidur dong"), hingga gaul/slang ("Idupin lampu kamar tidur ya", "Matiin musiknya sekarang").
* **Sintesis Akustik Resolusi Tinggi**: Pembangkitan audio multi-penutur sintetis berbasis neural TTS dengan variasi pitch, laju artikulasi (*speech rate*), serta konvolusi profil ruangan (*acoustic room profile*: *bedroom slow*, *living fast*, *kitchen norm*).

### 3.1.3 Dataset Rekaman Vokal Biologis Manusia (*Acoustic Anchor*)
Sebagai jangkar akustik biologis, dikumpulkan dataset rekaman vokal manusia nyata dari anggota tim (Haikal) sebanyak 124 klip audio terstandardisasi:
* 31 klip gaya tutur baku (*formal*)
* 31 klip gaya tutur santai (*casual/nyantai*)
* 31 klip gaya tutur gaul (*slang*)
* 31 klip pengujian tunggal (*single take baseline*)
Setiap berkas diverifikasi secara ketat memenuhi format 16 kHz Mono PCM_16 dengan normalisasi amplitudo RMS pada target -20.0 dBFS.

## 3.2 Analisis Dataset

### 3.2.1 Distribusi Slot Semantik dan Kelas Intent
Kombinasi ruang label pada sistem SVARA terdiri atas:
* **Slot Action (6 kelas)**: `activate`, `bring`, `change language`, `deactivate`, `decrease`, `increase`.
* **Slot Object (14 kelas)**: `Chinese`, `English`, `German`, `Korean`, `heat`, `juice`, `lamp`, `lights`, `music`, `newspaper`, `none`, `shoes`, `socks`, `volume`.
* **Slot Location (4 kelas)**: `bedroom`, `kitchen`, `none`, `washroom`.
Secara matematis, ruang kombinasi teoritis adalah $6 \times 14 \times 4 = 336$ kemungkinan, namun hanya 31 kombinasi *intent* valid yang memiliki relevansi logis operasional hunian (misal: `activate heat kitchen` adalah valid, sedangkan `bring volume washroom` adalah invalid).

### 3.2.2 Partisi Dataset dan Audit Nol Kebocoran (*Zero Data Leakage Audit*)
Korpus komprehensif SVARA 24.000 klip dipartisi secara proporsional 70% : 15% : 15% dengan pengawasan isolasi linguistik ketat:
* **Train Set (70% - 16.800 audio)**: 8.400 klip ID (3.065 frasa unik) + 8.400 klip EN (53 penutur). Durasi total: 13,09 jam.
* **Validation Set (15% - 3.600 audio)**: 1.800 klip ID (647 frasa unik) + 1.800 klip EN (13 penutur). Durasi total: 2,86 jam.
* **Test Set (15% - 3.600 audio)**: 1.800 klip ID (646 frasa unik) + 1.800 klip EN (13 penutur). Durasi total: 2,93 jam.

Audit formal kebocoran data (*zero leakage audit*) membuktikan:
$$\text{Overlap}(\text{Train}_{\text{phrase}}, \text{Test}_{\text{phrase}}) = 0 \quad (\text{Bahasa Indonesia - Strict Unseen Utterance})$$
$$\text{Overlap}(\text{Train}_{\text{speaker}}, \text{Test}_{\text{speaker}}) = 0 \quad (\text{Bahasa Inggris - Strict Speaker Disjoint})$$
Dengan demikian, model dipaksa memahami abstraksi semantik akustik, bukan menghafal kalimat latihan.

## 3.3 Arsitektur Sistem Menyeluruh
Arsitektur sistem SVARA dirancang secara modular hulu-ke-hilir (*end-to-end integrated ecosystem*), terdiri atas empat subsistem utama:
1. **Subsistem Audio Ingestion & Preprocessing**: Menangkap rekaman suara mikrofon dari peramban peramban web pengguna (Web Audio API), mendekompresi berkas biner WAV/PCM, melakukan *downsampling* menjadi 16.000 Hz mono, memotong hening awal-akhir menggunakan *Voice Activity Detection* (VAD) dengan bantalan pengaman (*safety padding*) 250 ms untuk mencegah pemotongan fonem (*anti-clipping*), dan menormalisasi energi sinyal ke -20 dBFS.
2. **Subsistem Deep Learning Inference Engine**: Model terkuantisasi `svara_hardened_int8.onnx` mengeksekusi inferensi multi-head pada ONNX Runtime CPU. Nilai logit diproyeksikan melalui *Softmax*, menghitung skor keyakinan (*confidence score*) secara terpadu. Jika keyakinan berada di bawah ambang batas penolakan ($\tau = 0,80$), sistem secara cerdas menolak mengeksekusi perintah (*abstain/reject*) demi keamanan.
3. **Subsistem Backend Microservice (FastAPI)**: Bertindak sebagai koordinator logika bisnis. Menerima payload audio melalui endpoint `/api/predict`, mengelola mesin status hunian (*in-memory simulated device state machine*), memicu transisi status perangkat (misal: menyalakan lampu ruang tamu), serta mencatat riwayat telemetri.
4. **Subsistem Frontend & 3D Digital Twin (React + Three.js)**: Menyajikan antarmuka visual dua modalitas: panel kontrol dua dimensi (*flat dashboard*) dan visualisasi interaktif tiga dimensi hunian cerdas (*first-person perspective walkthrough*). Ketika perintah diidentifikasi, entitas lampu atau pendingin ruangan pada dunia 3D seketika berubah warna dan status secara *real-time*.

## 3.4 Kebutuhan Hardware
* **Lingkungan Pelatihan (*Training & Optimization*)**:
  * Unit Pemroses Grafis (GPU): NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6 VRAM, TGP 140W, Arsitektur Ada Lovelace, Tensor Cores Generasi ke-4).
  * Prosesor Utama (CPU): Intel Core i7-13620H (10 Cores, 16 Threads, frekuensi turbo hingga 4,90 GHz).
  * Memori Utama (RAM): 16 GB DDR5 5200 MHz dual-channel.
  * Media Penyimpanan: NVMe PCIe Gen 4 SSD 1 TB (laju transfer sekuensial > 4.500 MB/s).
* **Lingkungan Inferensi / Target Penerapan (*Deployment / Production*)**:
  * Kompatibel pada CPU komputer personal standar tanpa akselerasi kartu grafis (Intel Core i3 / AMD Ryzen 3 atau lebih tinggi, alokasi RAM minimal 2 GB).

## 3.5 Kebutuhan Software
* **Sistem Operasi**: Microsoft Windows 11 Home 64-bit / Linux Ubuntu 22.04 LTS.
* **Bahasa Pemrograman & Lingkungan Komputasi**: Python 3.10 / 3.13, Node.js v20.x, TypeScript 5.x.
* **Pustaka Deep Learning & Audio**:
  * PyTorch 2.4+ dengan CUDA 12.4 Runtime.
  * Torchaudio 2.4+.
  * HuggingFace Transformers 4.40+ (model dasar `facebook/wav2vec2-base`).
  * SoundFile 0.12+, Librosa 0.10+, NumPy 1.26+, Pandas 2.2+.
* **Pustaka Inferensi & Kuantisasi**: ONNX 1.16+, ONNX Runtime 1.19+ (CPU Execution Provider).
* **Web & Visualisasi**: FastAPI 0.111+, Uvicorn 0.30+, React 18, Vite, Three.js / React Three Fiber, Tailwind CSS.

---

# BAB 4. METODOLOGI PENELITIAN

## 4.1 Metodologi Umum
Penelitian ini dijalankan mengikuti siklus terstruktur *Deep Learning Lifecycle* yang mencakup lima tahapan sistematis:
1. **Tahap Audit & Penyiapan Data**: Memvalidasi integritas dataset publik FSC, melakukan ekspansi linguistik bahasa Indonesia kolokial, mengonversi seluruh sampel ke format 16 kHz mono, dan melakukan enkripsi hash SHA-256 untuk memastikan reproduktifitas data.
2. **Tahap Pra-Pemrosesan & Augmentasi Akustik**: Menerapkan pipeline normalisasi sinyal, pemotongan VAD terstandardisasi, serta penambahan derau latar belakang (*SpecAugment* dan konvolusi gema ruangan).
3. **Tahap Pemodelan & Arsitektur Jaringan**: Mengonfigurasi arsitektur `Wav2VecSLU` multi-head dengan mekanisme *temporal mean pooling* di atas lapisan Transformer.
4. **Tahap Pelatihan & Acoustic Hardening**: Melatih model dengan strategi *two-phase training* (pelatihan basis korpus masif dilanjutkan dengan adaptasi *acoustic anchor* vokal manusia nyata menggunakan PyTorch AMP FP16).
5. **Tahap Evaluasi, Optimasi, dan Deployment**: Menguji performa generalisasi pada data *unseen*, mengekspor graf ke Universal ONNX, menerapkan *Dynamic INT8 Quantization*, mengukur disparitas numerik, dan mengintegrasikan model ke backend web.

## 4.2 Preprocessing
Untuk menjamin keselarasan absolut antara data pelatihan dan data inferensi langsung (*training-serving parity*), seluruh proses pengolahan audio menggunakan satu modul fungsi deterministik tunggal:
1. **Standardisasi Frekuensi & Saluran**: Seluruh sinyal didekodekan menjadi representasi floating-point bertaraf [-1.0, 1.0], diubah ke saluran mono melalui perataan (*channel averaging*), dan di-*resample* menjadi tepat 16.000 Hz.
2. **Normalisasi Energi Berbasis RMS**: Menghitung *Root Mean Square* (RMS) dari amplitudo sinyal dalam skala decibel relatif terhadap skala penuh (*dBFS*), kemudian menyelaraskan energinya ke target standar -20,0 dBFS:
   $$\text{Gain (dB)} = \text{Target}_{\text{RMS}} - 20 \log_{10}\left(\sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}\right)$$
3. **Voice Activity Detection (VAD) dengan Buffer Anti-Clipping**: Energi sinyal dipindai menggunakan jendela geser 20 ms. Titik awal wicara diaktifkan ketika daya sinyal melampaui -32 dBFS, dan titik henti dipicu ketika energi berada di bawah -48 dBFS selama lebih dari 200 ms berturut-turut. Untuk mencegah terpotongnya konsonan henti dan fonem letup pada akhir kata, ditambahkan bantalan durasi (*post-speech safety padding*) sebesar 250 ms (4.000 sampel).
4. **Pembatasan Durasi Sinyal**: Sinyal dibatasi pada durasi maksimal 5,0 detik (80.000 sampel). Jika melebihi, dilakukan pemotongan temporal terkontrol, dan jika kurang saat pelatihan batch, dilakukan *zero-padding* dengan pelindung *attention mask*.

## 4.3 Data Augmentation
Untuk mencegah model menghafal pola akustik artifisial, diterapkan tiga teknik augmentasi audio:
1. **SpecAugment pada Domain Waktu-Frekuensi**: Menerapkan *time masking* (hingga 10 frame temporal acak) dan *frequency masking* (hingga 8 saluran mel-spektrogram) langsung pada domain representasi laten fitur konvolusi untuk melatih ketahanan model terhadap hilangnya sebagian informasi suara.
2. **Schroeder Artificial Reverberation & Room Impulse Response (RIR)**: Melakukan konvolusi sinyal audio kering (*anechoic*) dengan profil pantulan akustik dinding hunian dengan waktu peluruhan gema $RT_{60}$ antara 0,15 hingga 0,40 detik.
3. **GPU-Accelerated Pitch Shifting & Speed Perturbation**: Memodulasi nada dasar vokal (*pitch shift*) sebesar $\pm 2$ semitone dan kecepatan artikulasi temporal antara $0,9\times$ hingga $1,1\times$ guna mensimulasikan karakteristik pita suara biologis penutur pria, wanita, maupun anak-anak.

## 4.4 Arsitektur Deep Learning
Model **Wav2VecSLU Multi-Head** yang dibangun dalam penelitian ini mengadopsi arsitektur hierarkis:
1. **Acoustic Feature Extractor**: 7 blok konvolusi 1 dimensi temporal dengan ukuran kernel $(10, 3, 3, 3, 3, 2, 2)$ dan *strides* $(5, 2, 2, 2, 2, 2, 2)$, mereduksi 16.000 sampel sinyal mentah menjadi representasi fitur dimensi 512 pada frekuensi temporal 49 Hz.
2. **Transformer Context Network**: 12 lapisan *Transformer blocks* dengan dimensi tersembunyi (*hidden dimension*) $d = 768$, 8 kepala atensi (*attention heads*), dan dimensi *feed-forward* 3.072 dengan fungsi aktivasi GeLU. Mekanisme atensi dinotasikan sebagai:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
3. **Temporal Pooling Layer**: Rangkaian vektor keluaran lapisan Transformer terakhir $H \in \mathbb{R}^{B \times T' \times 768}$ diringkas menjadi representasi vektor global tunggal $h_{\text{pool}} \in \mathbb{R}^{B \times 768}$ menggunakan rata-rata terbobot *attention mask*:
   $$h_{\text{pool}} = \frac{\sum_{t=1}^{T'} m_t \cdot H_t}{\sum_{t=1}^{T'} m_t}$$
4. **Tiga Kepala Klasifikasi Terisolasi (*Dedicated Multi-Head Linear Projections*)**:
   Vektor terpadu $h_{\text{pool}}$ diumpankan secara paralel ke dalam tiga modul *multilayer perceptron* terpisah:
   * **Action Head**: Linear(768 $\rightarrow$ 256) $\rightarrow$ Dropout(0.1) $\rightarrow$ ReLU $\rightarrow$ Linear(256 $\rightarrow$ 6)
   * **Object Head**: Linear(768 $\rightarrow$ 256) $\rightarrow$ Dropout(0.1) $\rightarrow$ ReLU $\rightarrow$ Linear(256 $\rightarrow$ 14)
   * **Location Head**: Linear(768 $\rightarrow$ 256) $\rightarrow$ Dropout(0.1) $\rightarrow$ ReLU $\rightarrow$ Linear(256 $\rightarrow$ 4)
5. **Fungsi Objektif Multi-Tugas (*Multi-Task Loss Objective*)**:
   Model dioptimasi secara terpadu meminimalkan kombinasi linear *Cross-Entropy Loss* dari ketiga kepala klasifikasi:
   $$\mathcal{L}_{\text{total}} = \lambda_{\text{act}} \mathcal{L}_{\text{CE}}(y_{\text{act}}, \hat{y}_{\text{act}}) + \lambda_{\text{obj}} \mathcal{L}_{\text{CE}}(y_{\text{obj}}, \hat{y}_{\text{obj}}) + \lambda_{\text{loc}} \mathcal{L}_{\text{CE}}(y_{\text{loc}}, \hat{y}_{\text{loc}})$$
   di mana bobot kerugian ditetapkan seimbang: $\lambda_{\text{act}} = 1,0$, $\lambda_{\text{obj}} = 1,0$, dan $\lambda_{\text{loc}} = 1,0$.

## 4.5 Hyperparameter dan Strategi Optimasi
Pelatihan model dikonfigurasi dengan hyperparameter ilmiah yang ketat:
* **Optimizer**: Decoupled AdamW ($\beta_1 = 0,9, \beta_2 = 0,98, \epsilon = 10^{-8}$) dengan *weight decay* 0,01 untuk mencegah *overfitting* pada lapisan proyeksi.
* **Learning Rate Schedule**:
  * *Base Learning Rate*: $1 \times 10^{-4}$ untuk kepala klasifikasi linear (*task heads*).
  * *Encoder Learning Rate*: $1 \times 10^{-5}$ dengan pembekuan (*freezing*) 6 lapisan konvolusi awal agar representasi fonetik dasar wav2vec 2.0 tidak rusak.
  * *Warm-up*: 10% dari total iterasi pertama, diikuti penurunan laju pemelajaran berbasis *Cosine Annealing*.
* **Ukuran Batch (*Batch Size*)**: 16 per instans komputasi dengan *Gradient Accumulation Steps* = 2 (ekuivalen ukuran batch efektif 32).
* **Mixed Precision**: PyTorch AMP (*Automatic Mixed Precision*) FP16 untuk meminimalkan jejak memori VRAM GPU hingga 45% dan meningkatkan throughput komputasi Tensor Cores.
* **Kriteria Evaluasi Utama**: *Exact Match Accuracy*, di mana sebuah prediksi hanya dianggap benar ($= 1$) jika ketiga slot ($\hat{y}_{\text{act}} = y_{\text{act}}$, $\hat{y}_{\text{obj}} = y_{\text{obj}}$, dan $\hat{y}_{\text{loc}} = y_{\text{loc}}$) tepat secara simultan:
  $$\text{Exact Match} = \frac{1}{M}\sum_{i=1}^M \mathbb{I}\left(\hat{y}_{\text{act}}^{(i)} = y_{\text{act}}^{(i)} \land \hat{y}_{\text{obj}}^{(i)} = y_{\text{obj}}^{(i)} \land \hat{y}_{\text{loc}}^{(i)} = y_{\text{loc}}^{(i)}\right)$$
