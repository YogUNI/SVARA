# SVARA — Lembar Panduan Rekaman Suara Asli Bahasa Indonesia (Phrase Sheet)

**Fokus Utama**: Pematangan Engine Deep Learning Smart Home SVARA untuk Perintah Suara **Bahasa Indonesia Sehari-hari (Natural & Colloquial)**.

Petunjuk untuk Pembicara (Haikal, Yoga, & Responden):
1. Bacalah setiap kalimat perintah Bahasa Indonesia di bawah ini dengan **intonasi santai, alami, dan jelas** sebagaimana kamu berbicara dengan asisten pintar di rumah.
2. Setiap pembicara membaca seluruh 31 kalimat perintah dalam 2 kondisi:
   - **Kondisi Tenang (q / quiet)**: Ruangan hening/kamar tidur.
   - **Kondisi Berisik (n / noisy)**: Dengan latar belakang suara kipas angin, TV, atau suasana rumah.
3. Format penamaan file WAV: `own_<speakerId>_<kondisi>_<intent_id:02d>.wav`
   - Contoh Haikal Tenang: `own_haikal_q_00.wav`, `own_haikal_q_01.wav`
   - Contoh Yoga Berisik: `own_yoga_n_00.wav`, `own_yoga_n_01.wav`

---

## Tabel 31 Kalimat Perintah Smart Home Bahasa Indonesia

| No | Intent ID | Kalimat Perintah yang Dibaca (Bahasa Indonesia) | Variasi Gaul / Sehari-hari | Action | Object | Location |
|:--:|:---------:|:------------------------------------------------|:---------------------------|:------:|:------:|:--------:|
| 1  | **00** | **"Nyalakan lampu meja"** | *"Tolong hidupin lampunya dong"* | `activate` | `lamp` | `none` |
| 2  | **01** | **"Nyalakan lampu kamar tidur"** | *"Nyalain lampu kamar dong"* | `activate` | `lights` | `bedroom` |
| 3  | **02** | **"Nyalakan lampu dapur"** | *"Tolong hidupin lampu dapur"* | `activate` | `lights` | `kitchen` |
| 4  | **03** | **"Nyalakan semua lampu"** | *"Nyalain lampunya ya"* | `activate` | `lights` | `none` |
| 5  | **04** | **"Nyalakan lampu kamar mandi"** | *"Idupin lampu toilet dong"* | `activate` | `lights` | `washroom` |
| 6  | **05** | **"Putar lagi lagunya"** | *"Lanjutin musiknya dong"* | `activate` | `music` | `none` |
| 7  | **06** | **"Bawakan saya jus"** | *"Ambilin jus dingin dong"* | `bring` | `juice` | `none` |
| 8  | **07** | **"Ambilkan koran untuk saya"** | *"Tolong bawain koran hari ini"* | `bring` | `newspaper` | `none` |
| 9  | **08** | **"Ambilkan sepatu saya"** | *"Bawain sepatu gua dong"* | `bring` | `shoes` | `none` |
| 10 | **09** | **"Ambilkan kaus kaki"** | *"Ambilin kaos kaki dong"* | `bring` | `socks` | `none` |
| 11 | **10** | **"Ganti ke bahasa Mandarin dong"** | *"Ubah bahasa ke Mandarin"* | `change language` | `Chinese` | `none` |
| 12 | **11** | **"Ganti bahasanya ke Bahasa Inggris"** | *"Setel bahasa ke Inggris dong"* | `change language` | `English` | `none` |
| 13 | **12** | **"Ganti bahasanya ke Bahasa Jerman"** | *"Ubah bahasanya ke Jerman"* | `change language` | `German` | `none` |
| 14 | **13** | **"Ganti ke bahasa Korea dong"** | *"Set bahasa ke Korea"* | `change language` | `Korean` | `none` |
| 15 | **14** | **"Ganti bahasa sistem"** | *"Ganti bahasa dong"* | `change language` | `none` | `none` |
| 16 | **15** | **"Matikan lampu meja"** | *"Matiin lampunya dong"* | `deactivate` | `lamp` | `none` |
| 17 | **16** | **"Matikan lampu kamar"** | *"Matiin lampu kamar tidur"* | `deactivate` | `lights` | `bedroom` |
| 18 | **17** | **"Matikan lampu dapur"** | *"Tolong matiin lampu di dapur"* | `deactivate` | `lights` | `kitchen` |
| 19 | **18** | **"Matikan semua lampu"** | *"Padamin seluruh lampu rumah"* | `deactivate` | `lights` | `none` |
| 20 | **19** | **"Matikan lampu kamar mandi"** | *"Matiin lampu toilet dong"* | `deactivate` | `lights` | `washroom` |
| 21 | **20** | **"Matikan musiknya"** | *"Stop lagunya dong"* | `deactivate` | `music` | `none` |
| 22 | **21** | **"Dinginkan kamar tidur dong"** | *"Turunin suhu kamar tidur"* | `decrease` | `heat` | `bedroom` |
| 23 | **22** | **"Dinginkan dapur dong"** | *"Turunin suhu di dapur"* | `decrease` | `heat` | `kitchen` |
| 24 | **23** | **"Turunkan suhunya dong"** | *"Kurangi suhu ruangan"* | `decrease` | `heat` | `none` |
| 25 | **24** | **"Dinginkan kamar mandi dong"** | *"Turunin suhu kamar mandi"* | `decrease` | `heat` | `washroom` |
| 26 | **25** | **"Kecilkan volumenya dong"** | *"Pelanin suaranya ya"* | `decrease` | `volume` | `none` |
| 27 | **26** | **"Naikkan suhu kamar tidur dong"** | *"Kamar dingin banget naikin suhunya"* | `increase` | `heat` | `bedroom` |
| 28 | **27** | **"Naikkan suhu dapur dong"** | *"Panaskan suhu di dapur"* | `increase` | `heat` | `kitchen` |
| 29 | **28** | **"Naikkan suhunya dong"** | *"Ruangan dingin tolong naikin suhunya"* | `increase` | `heat` | `none` |
| 30 | **29** | **"Naikkan suhu kamar mandi dong"** | *"Panasin suhu kamar mandi"* | `increase` | `heat` | `washroom` |
| 31 | **30** | **"Besarkan suaranya dong"** | *"Kencengin volumenya"* | `increase` | `volume` | `none` |
