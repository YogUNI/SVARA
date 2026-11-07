# SVARA — Lembar Panduan Rekaman Suara Asli Bahasa Indonesia (Phrase Sheet)

**Fokus Utama**: Pematangan Engine Deep Learning Smart Home SVARA untuk Perintah Suara **Bahasa Indonesia Sehari-hari (Baku, Nyantai, & Slang)** Terstruktur per Ruangan & Fitur Web Demo.

Petunjuk untuk Pembicara (Haikal, Yoga, & Responden):
1. Setiap nomor perintah mewakili aksi nyata pada perangkat di web demo (Lampu, AC/Suhu, Musik, Volume, dll).
2. Terdapat 3 versi kalimat beruntun per intent: **Baku**, **Nyantai**, dan **Slang / Gaul**.
3. Sistem perekaman menggunakan **Triple-Take**: Pembicara merekam ketiga versi satu demi satu secara berurutan untuk menghasilkan 93 klip vokal asli (31 intent × 3 take) agar model memiliki ketahanan akustik maksimal.

---

## 🍳 1. ZONA DAPUR (KITCHEN)
*Perangkat Web: Lampu plafon dapur (kuning), Termostat pemanas dapur.*

| No | Intent ID | Fitur di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul | Slot Intent |
|:--:|:---------:|-------------------|------------|-----------------------------|--------------------|:-----------:|
| 1  | **02** | 💡 **Lampu Dapur MENYALA** | *"Nyalakan lampu dapur"* | *"Nyalain lampu di dapur ya"* | *"Tolong idupin lampu dapur dong"* | `activate` `lights` `kitchen` |
| 2  | **17** | 💡 **Lampu Dapur MATI** | *"Matikan lampu dapur"* | *"Matiin lampu dapur dong"* | *"Padamin lampu di dapur ya"* | `deactivate` `lights` `kitchen` |
| 3  | **27** | 🔥 **Suhu Dapur NAIK (+2°C)** | *"Naikkan suhu dapur"* | *"Panasin suhu di dapur dong"* | *"Naikin temperatur dapur ya"* | `increase` `heat` `kitchen` |
| 4  | **22** | ❄️ **Suhu Dapur TURUN (-2°C)** | *"Turunkan suhu dapur"* | *"Dinginin dapur dong"* | *"Dapur gerah nih tolong sejukin"* | `decrease` `heat` `kitchen` |

---

## 🛏️ 2. ZONA KAMAR TIDUR (BEDROOM)
*Perangkat Web: Lampu utama plafon kamar, Lampu tidur sudut (lamp meja), Termostat pemanas kamar.*

| No | Intent ID | Fitur di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul | Slot Intent |
|:--:|:---------:|-------------------|------------|-----------------------------|--------------------|:-----------:|
| 5  | **01** | 💡 **Lampu Kamar MENYALA** | *"Nyalakan lampu kamar tidur"* | *"Nyalain lampu kamar dong"* | *"Idupin lampu kamar tidur ya"* | `activate` `lights` `bedroom` |
| 6  | **16** | 💡 **Lampu Kamar MATI** | *"Matikan lampu kamar tidur"* | *"Matiin lampu kamar tidur dong"* | *"Padamin lampu di kamar ya"* | `deactivate` `lights` `bedroom` |
| 7  | **00** | 🪔 **Lampu Meja / Tidur MENYALA** | *"Nyalakan lampu meja"* | *"Nyalain lampu meja dong"* | *"Tolong hidupin lampu belajar ya"* | `activate` `lamp` `none` |
| 8  | **15** | 🪔 **Lampu Meja / Tidur MATI** | *"Matikan lampu meja"* | *"Matiin lampu meja dong"* | *"Tolong matiin lampu belajar ya"* | `deactivate` `lamp` `none` |
| 9  | **26** | 🔥 **Suhu Kamar NAIK (+2°C)** | *"Naikkan suhu kamar tidur"* | *"Kamar dingin naikin suhunya dong"* | *"Panasin suhu kamar tidur ya"* | `increase` `heat` `bedroom` |
| 10 | **21** | ❄️ **Suhu Kamar TURUN (-2°C)** | *"Turunkan suhu kamar tidur"* | *"Dinginkan kamar tidur dong"* | *"Kamar gerah banget tolong sejukin"* | `decrease` `heat` `bedroom` |

---

## 🚿 3. ZONA KAMAR MANDI (WASHROOM / TOILET)
*Perangkat Web: Lampu plafon kamar mandi, Termostat pemanas air/ruang mandi.*

| No | Intent ID | Fitur di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul | Slot Intent |
|:--:|:---------:|-------------------|------------|-----------------------------|--------------------|:-----------:|
| 11 | **04** | 💡 **Lampu Kamar Mandi MENYALA** | *"Nyalakan lampu kamar mandi"* | *"Nyalain lampu kamar mandi dong"* | *"Idupin lampu toilet ya"* | `activate` `lights` `washroom` |
| 12 | **19** | 💡 **Lampu Kamar Mandi MATI** | *"Matikan lampu kamar mandi"* | *"Matiin lampu kamar mandi dong"* | *"Tolong matiin lampu toilet ya"* | `deactivate` `lights` `washroom` |
| 13 | **29** | 🔥 **Suhu Kamar Mandi NAIK (+2°C)** | *"Naikkan suhu kamar mandi"* | *"Kamar mandi dingin tolong hangatkan"* | *"Panasin suhu kamar mandi ya"* | `increase` `heat` `washroom` |
| 14 | **24** | ❄️ **Suhu Kamar Mandi TURUN (-2°C)** | *"Turunkan suhu kamar mandi"* | *"Dinginkan kamar mandi dong"* | *"Sejukin suhu di kamar mandi ya"* | `decrease` `heat` `washroom` |

---

## 🎵 4. ZONA LORONG & AUDIO (HALLWAY / MEDIA)
*Perangkat Web: Speaker HomePod lorong menyala hijau, Bar volume naik/turun, Gelombang audio berdenyut.*

| No | Intent ID | Fitur di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul | Slot Intent |
|:--:|:---------:|-------------------|------------|-----------------------------|--------------------|:-----------:|
| 15 | **05** | 🔊 **Musik MENYALA (Play)** | *"Nyalakan musik"* | *"Putar lagi lagunya dong"* | *"Setel musik dong bro"* | `activate` `music` `none` |
| 16 | **20** | 🔇 **Musik MATI (Stop)** | *"Matikan musik"* | *"Matiin lagunya dong"* | *"Stop musiknya sekarang ya"* | `deactivate` `music` `none` |
| 17 | **30** | 📢 **Volume NAIK (+2 Bar)** | *"Besarkan volume suara"* | *"Kencengin suaranya dong"* | *"Gedein volumenya dikit bro"* | `increase` `volume` `none` |
| 18 | **25** | 🔉 **Volume TURUN (-2 Bar)** | *"Kecilkan volume suara"* | *"Pelanin suaranya dong"* | *"Kecilin volumenya ya bro"* | `decrease` `volume` `none` |

---

## 🏠 5. PENGATURAN MASTER SELURUH RUMAH (MASTER CONTROLS)
*Perangkat Web: Semua lampu di denah menyala/mati serentak, Suhu global rumah disesuaikan.*

| No | Intent ID | Fitur di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul | Slot Intent |
|:--:|:---------:|-------------------|------------|-----------------------------|--------------------|:-----------:|
| 19 | **03** | 💡 **SEMUA Lampu MENYALA** | *"Nyalakan semua lampu"* | *"Nyalain semua lampu rumah dong"* | *"Terangin seluruh ruangan ya"* | `activate` `lights` `none` |
| 20 | **18** | 💡 **SEMUA Lampu MATI** | *"Matikan semua lampu"* | *"Matiin semua lampu dong"* | *"Padamin seluruh lampu rumah ya"* | `deactivate` `lights` `none` |
| 21 | **28** | 🔥 **Suhu Rumah NAIK (+2°C)** | *"Naikkan suhu ruangan"* | *"Rumah dingin tolong naikin suhunya"* | *"Hangatkan seluruh rumah ya"* | `increase` `heat` `none` |
| 22 | **23** | ❄️ **Suhu Rumah TURUN (-2°C)** | *"Turunkan suhu ruangan"* | *"Turunkan suhunya dong"* | *"Rumah gerah tolong sejukin ya"* | `decrease` `heat` `none` |

---

## 🌐 6. PENGATURAN BAHASA ASISTEN (SYSTEM LANGUAGE)
*Perangkat Web: Badge bahasa sistem di pojok kanan atas berubah bendera/bahasa.*

| No | Intent ID | Fitur di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul | Slot Intent |
|:--:|:---------:|-------------------|------------|-----------------------------|--------------------|:-----------:|
| 23 | **14** | ⚙️ **Ganti Bahasa Sistem** | *"Ganti bahasa sistem"* | *"Ganti bahasanya dong"* | *"Ubah setelan bahasa ya"* | `change language` `none` `none` |
| 24 | **11** | 🇬🇧 **Set Bahasa Inggris** | *"Ganti ke Bahasa Inggris"* | *"Ganti bahasanya ke Bahasa Inggris"* | *"Setel bahasa Inggris dong"* | `change language` `English` `none` |
| 25 | **10** | 🇨🇳 **Set Bahasa Mandarin** | *"Ganti ke Bahasa Mandarin"* | *"Ganti ke Bahasa Mandarin dong"* | *"Set bahasa ke Mandarin ya"* | `change language` `Chinese` `none` |
| 26 | **12** | 🇩🇪 **Set Bahasa Jerman** | *"Ganti ke Bahasa Jerman"* | *"Ganti bahasanya ke Bahasa Jerman"* | *"Ubah ke bahasa Jerman dong"* | `change language` `German` `none` |
| 27 | **13** | 🇰🇷 **Set Bahasa Korea** | *"Ganti ke Bahasa Korea"* | *"Ganti ke Bahasa Korea dong"* | *"Setel bahasa Korea ya"* | `change language` `Korean` `none` |

---

## 🤖 7. SMART OBJECTS / ROBOT BUTLER (BENCHMARK FSC)
*Penjelasan: 4 intent ini wajib diuji agar nilai Tugas Akhir kita bisa dibandingkan dengan paper jurnal internasional (Lugosch et al., 2019). Di Web Mode FPP 3D, objek ini berwujud gelas jus di meja makan, koran di meja tamu, dan sepatu di rak.*

| No | Intent ID | Fitur di Web Demo | Versi Baku | Versi Nyantai (Rekomendasi) | Versi Slang / Gaul | Slot Intent |
|:--:|:---------:|-------------------|------------|-----------------------------|--------------------|:-----------:|
| 28 | **06** | 🧃 **Robot Antar Jus** | *"Bawakan saya jus"* | *"Ambilkan jus dingin dong"* | *"Bawain jus jeruk ya bro"* | `bring` `juice` `none` |
| 29 | **07** | 📰 **Robot Antar Koran** | *"Bawakan koran untuk saya"* | *"Tolong ambilkan koran dong"* | *"Bawain koran pagi ini ya"* | `bring` `newspaper` `none` |
| 30 | **08** | 👟 **Robot Ambil Sepatu** | *"Ambilkan sepatu saya"* | *"Tolong bawakan sepatu dong"* | *"Bawain sepatu gua ya bro"* | `bring` `shoes` `none` |
| 31 | **09** | 🧦 **Robot Ambil Kaus Kaki** | *"Ambilkan kaus kaki"* | *"Tolong bawakan kaus kaki dong"* | *"Ambilin kaos kaki bersih ya"* | `bring` `socks` `none` |
