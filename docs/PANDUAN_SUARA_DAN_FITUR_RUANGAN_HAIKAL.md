# PANDUAN LENGKAP REKAMAN SUARA BERDASARKAN FITUR RUANGAN WEB DEMO (SVARA)

Dokumen ini adalah **contekan resmi (cheat-sheet)** untuk Haikal sebelum dan saat berbicara.
Setiap perintah suara di bawah ini dihubungkan langsung dengan **efek visual nyata di Web Demo (2D CAD Floor Plan, 3D Diorama, dan 3D First-Person Walkthrough)**.

### 🎙️ Sistem Rekaman Triple-Take (3 Take Beruntun per Fitur):
Untuk setiap fitur smart home di bawah, Haikal akan **mengucapkan 3 variasi berturut-turut** (total 31 intent × 3 take = 93 klip vokal):
1. **Take 1 (Versi Baku)**: Kalimat formal, jelas, dan presisi.
2. **Take 2 (Versi Nyantai)**: Kalimat santai percakapan sehari-hari.
3. **Take 3 (Versi Slang / Gaul)**: Kalimat gaul anak muda (*"idupin"*, *"matiin"*, *"kencengin"*, *"dinginin"*, dll).

> **Mengapa 3 Take Beruntun?**
> Pengulangan 3 gaya ini bertindak sebagai **Acoustic Anchor** yang sangat kuat bagi model Wav2Vec 2.0. Model tidak akan kaku dan mampu mengenali perintah baik saat kamu berbicara formal, santai, maupun gaul saat live demo!

---

## 🍳 1. ZONA DAPUR (KITCHEN)
*Perangkat yang bereaksi di Web: Lampu plafon dapur (kuning), Termostat pemanas dapur.*

| Intent ID | Target Efek di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul |
|:---------:|-------------------------|------------|-----------------------------|--------------------|
| **02** | 💡 **Lampu Dapur MENYALA** | *"Nyalakan lampu dapur"* | *"Nyalain lampu di dapur ya"* | *"Tolong idupin lampu dapur dong"* |
| **17** | 💡 **Lampu Dapur MATI** | *"Matikan lampu dapur"* | *"Matiin lampu dapur dong"* | *"Padamin lampu di dapur ya"* |
| **27** | 🔥 **Suhu Dapur NAIK (+2°C)** | *"Naikkan suhu dapur"* | *"Panasin suhu di dapur dong"* | *"Naikin temperatur dapur ya"* |
| **22** | ❄️ **Suhu Dapur TURUN (-2°C)** | *"Turunkan suhu dapur"* | *"Dinginin dapur dong"* | *"Dapur gerah nih tolong sejukin"* |

---

## 🛏️ 2. ZONA KAMAR TIDUR (BEDROOM)
*Perangkat yang bereaksi di Web: Lampu utama plafon kamar, Lampu tidur sudut (lamp meja), Termostat pemanas kamar.*

| Intent ID | Target Efek di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul |
|:---------:|-------------------------|------------|-----------------------------|--------------------|
| **01** | 💡 **Lampu Kamar MENYALA** | *"Nyalakan lampu kamar tidur"* | *"Nyalain lampu kamar dong"* | *"Idupin lampu kamar tidur ya"* |
| **16** | 💡 **Lampu Kamar MATI** | *"Matikan lampu kamar tidur"* | *"Matiin lampu kamar tidur dong"* | *"Padamin lampu di kamar ya"* |
| **00** | 🪔 **Lampu Meja / Tidur MENYALA** | *"Nyalakan lampu meja"* | *"Nyalain lampu meja dong"* | *"Tolong hidupin lampu belajar ya"* |
| **15** | 🪔 **Lampu Meja / Tidur MATI** | *"Matikan lampu meja"* | *"Matiin lampu meja dong"* | *"Tolong matiin lampu belajar ya"* |
| **26** | 🔥 **Suhu Kamar NAIK (+2°C)** | *"Naikkan suhu kamar tidur"* | *"Kamar dingin naikin suhunya dong"* | *"Panasin suhu kamar tidur ya"* |
| **21** | ❄️ **Suhu Kamar TURUN (-2°C)** | *"Turunkan suhu kamar tidur"* | *"Dinginkan kamar tidur dong"* | *"Kamar gerah banget tolong sejukin"* |

---

## 🚿 3. ZONA KAMAR MANDI (WASHROOM / TOILET)
*Perangkat yang bereaksi di Web: Lampu plafon kamar mandi, Termostat pemanas air/ruang mandi.*

| Intent ID | Target Efek di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul |
|:---------:|-------------------------|------------|-----------------------------|--------------------|
| **04** | 💡 **Lampu Kamar Mandi MENYALA** | *"Nyalakan lampu kamar mandi"* | *"Nyalain lampu kamar mandi dong"* | *"Idupin lampu toilet ya"* |
| **19** | 💡 **Lampu Kamar Mandi MATI** | *"Matikan lampu kamar mandi"* | *"Matiin lampu kamar mandi dong"* | *"Tolong matiin lampu toilet ya"* |
| **29** | 🔥 **Suhu Kamar Mandi NAIK (+2°C)** | *"Naikkan suhu kamar mandi"* | *"Kamar mandi dingin tolong hangatkan"* | *"Panasin suhu kamar mandi ya"* |
| **24** | ❄️ **Suhu Kamar Mandi TURUN (-2°C)** | *"Turunkan suhu kamar mandi"* | *"Dinginkan kamar mandi dong"* | *"Sejukin suhu di kamar mandi ya"* |

---

## 🎵 4. ZONA LORONG & AUDIO SYSTEM (HALLWAY / MEDIA)
*Perangkat yang bereaksi di Web: Speaker HomePod lorong menyala hijau, Bar volume naik/turun, Gelombang audio berdenyut.*

| Intent ID | Target Efek di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul |
|:---------:|-------------------------|------------|-----------------------------|--------------------|
| **05** | 🔊 **Musik MENYALA (Play)** | *"Nyalakan musik"* | *"Putar lagi lagunya dong"* | *"Setel musik dong bro"* |
| **20** | 🔇 **Musik MATI (Stop)** | *"Matikan musik"* | *"Matiin lagunya dong"* | *"Stop musiknya sekarang ya"* |
| **30** | 📢 **Volume NAIK (+2 Bar)** | *"Besarkan volume suara"* | *"Kencengin suaranya dong"* | *"Gedein volumenya dikit bro"* |
| **25** | 🔉 **Volume TURUN (-2 Bar)** | *"Kecilkan volume suara"* | *"Pelanin suaranya dong"* | *"Kecilin volumenya ya bro"* |

---

## 🏠 5. PENGATURAN MASTER SELURUH RUMAH (MASTER CONTROLS)
*Perangkat yang bereaksi di Web: Semua lampu di denah menyala/mati serentak, Suhu global rumah disesuaikan.*

| Intent ID | Target Efek di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul |
|:---------:|-------------------------|------------|-----------------------------|--------------------|
| **03** | 💡 **SEMUA Lampu MENYALA** | *"Nyalakan semua lampu"* | *"Nyalain semua lampu rumah dong"* | *"Terangin seluruh ruangan ya"* |
| **18** | 💡 **SEMUA Lampu MATI** | *"Matikan semua lampu"* | *"Matiin semua lampu dong"* | *"Padamin seluruh lampu rumah ya"* |
| **28** | 🔥 **Suhu Rumah NAIK (+2°C)** | *"Naikkan suhu ruangan"* | *"Rumah dingin tolong naikin suhunya"* | *"Hangatkan seluruh rumah ya"* |
| **23** | ❄️ **Suhu Rumah TURUN (-2°C)** | *"Turunkan suhu ruangan"* | *"Turunkan suhunya dong"* | *"Rumah gerah tolong sejukin ya"* |

---

## 🌐 6. PENGATURAN BAHASA ASISTEN (SYSTEM LANGUAGE)
*Perangkat yang bereaksi di Web: Badge bahasa sistem di pojok kanan atas berubah bendera/bahasa.*

| Intent ID | Target Efek di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul |
|:---------:|-------------------------|------------|-----------------------------|--------------------|
| **14** | ⚙️ **Ganti Bahasa Sistem** | *"Ganti bahasa sistem"* | *"Ganti bahasanya dong"* | *"Ubah setelan bahasa ya"* |
| **11** | 🇬🇧 **Set Bahasa Inggris** | *"Ganti ke Bahasa Inggris"* | *"Ganti bahasanya ke Bahasa Inggris"* | *"Setel bahasa Inggris dong"* |
| **10** | 🇨🇳 **Set Bahasa Mandarin** | *"Ganti ke Bahasa Mandarin"* | *"Ganti ke Bahasa Mandarin dong"* | *"Set bahasa ke Mandarin ya"* |
| **12** | 🇩🇪 **Set Bahasa Jerman** | *"Ganti ke Bahasa Jerman"* | *"Ganti bahasanya ke Bahasa Jerman"* | *"Ubah ke bahasa Jerman dong"* |
| **13** | 🇰🇷 **Set Bahasa Korea** | *"Ganti ke Bahasa Korea"* | *"Ganti ke Bahasa Korea dong"* | *"Setel bahasa Korea ya"* |

---

## 🤖 7. SMART OBJECTS / ROBOT BUTLER (BENCHMARK FSC)
*Mengapa fitur ini ada? Ini adalah 4 intent warisan dari dataset benchmark internasional Fluent Speech Commands (Lugosch et al., 2019) yang wajib diuji agar nilai Tugas Akhir kita bisa dibandingkan dengan paper jurnal. Di Web Mode FPP 3D, objek ini berwujud gelas jus di meja makan, koran di meja tamu, dan sepatu di rak.*

| Intent ID | Target Efek di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul |
|:---------:|-------------------------|------------|-----------------------------|--------------------|
| **06** | 🧃 **Robot Antar Jus** | *"Bawakan saya jus"* | *"Ambilkan jus dingin dong"* | *"Bawain jus jeruk ya bro"* |
| **07** | 📰 **Robot Antar Koran** | *"Bawakan koran untuk saya"* | *"Tolong ambilkan koran dong"* | *"Bawain koran pagi ini ya"* |
| **08** | 👟 **Robot Ambil Sepatu** | *"Ambilkan sepatu saya"* | *"Tolong bawakan sepatu dong"* | *"Bawain sepatu gua ya bro"* |
| **09** | 🧦 **Robot Ambil Kaus Kaki** | *"Ambilkan kaus kaki"* | *"Tolong bawakan kaus kaki dong"* | *"Ambilin kaos kaki bersih ya"* |

---

## 💡 CARA MEREKAM DI TERMINAL (ALUR TRIPLE-TAKE)

1. Buka file panduan ini di layar kamu sebagai contekan sebelum berbicara.
2. Jalankan script perekam di terminal:
   ```powershell
   python scripts/record_own_voice.py --speaker haikal --condition quiet
   ```
3. Script akan otomatis memandu kamu per ruangan dan per fitur:
   - **Langkah 1**: Terminal menampilkan [TAKE 1 - BAKU]. Tekan `[ENTER]`, ucapkan kalimat Baku saat status `[MEREKAM 3.5s]` aktif, lalu tekan `[ENTER]` untuk konfirmasi simpan.
   - **Langkah 2**: Terminal otomatis lanjut ke [TAKE 2 - NYANTAI]. Tekan `[ENTER]`, ucapkan kalimat Nyantai, lalu tekan `[ENTER]` untuk simpan.
   - **Langkah 3**: Terminal otomatis lanjut ke [TAKE 3 - SLANG]. Tekan `[ENTER]`, ucapkan kalimat Slang, lalu tekan `[ENTER]` untuk simpan.
   - Setelah 3 take selesai, script langsung berpindah ke fitur/ruangan berikutnya!
4. Jika kamu merasa salah bicara atau batuk pada satu take, ketik `r` saat konfirmasi untuk mengulang take tersebut.
5. Selesai 31 intent (93 rekaman, ~7-8 menit), jalankan validasi & konversi standar:
   ```powershell
   python scripts/validate_and_convert_own_recordings.py
   ```

