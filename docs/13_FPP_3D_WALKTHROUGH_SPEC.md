# 13 — Spesifikasi & Roadmap 3D FPP Walkthrough (Game-Style Interactive Smart Home)

Dokumen ini mendefinisikan arsitektur teknis, desain visual, sistem kontrol, dan tahapan eksekusi detail untuk fitur **First-Person Perspective (FPP) Walkthrough** pada antarmuka web SVARA. Fitur ini dirancang sebagai inovasi penyajian demo kelas atas (*interactive game-style simulation*) tanpa mengorbankan integritas engine Deep Learning inti (*Transformer wav2vec 2.0*).

---

## 1. Filosofi & Tujuan Sistem
- **Tujuan Utama**: Menghadirkan pengalaman otomasi rumah cerdas interaktif di mana pengguna/penguji dapat "berjalan langsung" di dalam rumah virtual dari sudut pandang mata manusia (FPP layaknya game PUBG Mobile / FPS), berbicara secara alami melalui kontrol suara (*push-to-talk*), dan menyaksikan perangkat bereaksi di depan mata secara real-time.
- **Kepatuhan Akademik**: Model AI di balik layar tetap model asli hasil pelatihan pada GPU RTX 4060 (`facebook/wav2vec2-base`, akurasi 99.60% Split A & 95.87% Split B1). Visualisasi 3D adalah representasi fisik dari status state machine perangkat.

---

## 2. Arsitektur Teknologi (The Tech Stack)

| Lapisan Sistem | Teknologi yang Digunakan | Peran & Rincian Teknis |
| :--- | :--- | :--- |
| **Graphics Core** | **Three.js (WebGL2 / WebGPU Pipeline)** | Rendering 3D interior rumah, kamera perspektif, lighting, dan bayangan dinamis. |
| **Material & Shading** | **Physically Based Rendering (PBR)** | Material MeshStandardMaterial dengan roughness, metalness, dan ambient occlusion untuk tekstur lantai kayu, dinding, dan perabotan. |
| **Camera & Control** | **PointerLockControls + Kinematic Physics** | Mengunci mouse browser untuk rotasi 360°, navigasi WASD dengan akselerasi/deselerasi inersia, dan ayunan langkah (*head bobbing*). |
| **Voice Interface** | **Web Audio API + Spacebar Hook** | Merekam audio 16 kHz Mono WAV saat tombol `Spacebar` ditahan, mengirim ke `/api/predict` saat dilepas. |
| **Smart State Engine** | **Reactive State Sync** | Sinkronisasi real-time status perangkat (lampu menyala dengan PointLight/SpotLight, pemanas memancarkan warna termal, speaker bergetar). |

---

## 3. Rincian Fitur Utama (Feature Breakdown)

### A. Kontrol FPP Manusia Nyata (Character Controller)
1. **Pointer Lock Look**: Klik pada layar 3D untuk mengunci kursor. Gerakan mouse mengontrol sudut pandang kamera (Pitch & Yaw) dengan batas sudut vertikal (-85° hingga +85° agar tidak pusing).
2. **WASD Walk & Sprint**:
   - `W`: Melangkah maju.
   - `S`: Melangkah mundur.
   - `A`: Geser ke kiri (Strafe Left).
   - `D`: Geser ke kanan (Strafe Right).
   - `Left Shift`: Berjalan cepat (*Sprint*).
3. **Ketinggian Mata Manusia (Eye Level)**: Posisi kamera diatur tetap pada sumbu $Y = 1.65\text{ m}$ (tinggi rata-rata pandangan mata manusia).
4. **Head Bobbing Sinusoidal**: Kamera mengayun halus naik-turun dan miring kiri-kanan saat karakter bergerak:
   $$\Delta Y = \sin(\text{time} \times \text{walk\_speed}) \times \text{amplitude}$$
5. **Dinding Pembatas (Collision Boundary)**: Dinding ruangan memiliki batas koordinat sehingga kamera tidak tembus keluar ruangan.

### B. Lingkungan Interior Realistis (The Smart House Environment)
1. **Dapur (Kitchen)**: Meja counter marmer, kompor induksi, lemari es, dan lampu plafon hangat (*warm downlight*).
2. **Kamar Tidur (Bedroom)**: Kasur, meja samping dengan lampu tidur (*bedside lamp*), dan unit pemanas termal di dinding.
3. **Kamar Mandi (Washroom)**: Area wastafel, cermin, dan pencahayaan khusus.
4. **Ruang Tamu & Lorong (Living Room & Hall)**: Sofa modern, sistem speaker cerdas dengan visualisasi gelombang suara saat musik aktif.

### C. Kehadiran Tubuh Karakter & Smartwatch (First-Person Presence)
1. **Model Lengan & Smartwatch**: Di pojok bawah layar terlihat lengan karakter dengan smartwatch cerdas di pergelangan tangan kiri.
2. **Animasi Angkat Tangan (Walkie-Talkie Pose)**: Saat tombol `Spacebar` ditekan untuk berbicara, lengan kiri terangkat ke arah mulut dengan cincin hologram audio yang menyala.

### D. Head-Up Display (HUD) Cyberpunk / Modern Minimalis
1. **Crosshair Reticle (`+`)**: Titik bidik presisi di tengah layar.
2. **Room Location Tag (Pojok Kiri Atas)**: Indikator ruangan yang otomatis mendeteksi posisi pemain (misal: *"📍 Dapur (Kitchen)"*).
3. **Voice Command Banner (Pojok Bawah Layar)**: Menampilkan hasil inferensi AI langsung (Action, Object, Location, Confidence %, dan Latensi ms).
4. **Mini-Map Radar (Pojok Kanan Atas)**: Denah radar kecil yang memperlihatkan posisi karakter dan sudut pandang arah hadap.

### E. Multi-Camera View Toggle (Tombol `V`)
- **Mode 1: FPP Walkthrough (Game Mode)** — Masuk ke dalam rumah & berjalan bebas.
- **Mode 2: Isometric 3D (The Sims Mode)** — Pandangan isometrik dari atas dengan dinding terbuka.
- **Mode 3: 2D Blueprint** — Denah skematik teknis arsitektur SVG.

---

## 4. Tahapan Pengerjaan Rinci (Step-by-Step Execution Plan)

```
[ Step 1: Controller Fondasi ] ──► [ Step 2: Interior Ruangan & PBR ] ──► [ Step 3: Pencahayaan Dinamis ]
                                                                                   │
[ Step 6: Polish & Performance ] ◄── [ Step 5: Tangan 3D & Smartwatch ] ◄── [ Step 4: Integrasi Suara & HUD ]
```

### 🔹 STEP 1 — Kinematic FPP Controller & Pointer Lock Setup
- [ ] Buat modul `web/src/components/fpp/FirstPersonController.ts`.
- [ ] Implementasikan `PointerLockControls` Three.js dengan event listener `click` dan `lock/unlock`.
- [ ] Tambahkan listener keyboard (`keydown`, `keyup`) untuk input `WASD`, `Shift`, dan `Space`.
- [ ] Tambahkan kalkulasi pergerakan vektor kecepatan (*velocity vector*) dengan redaman gesekan (*damping/friction* = 10.0).
- [ ] Terapkan rumus matematika *Head Bobbing* dinamis berdasarkan kecepatan gerak.
- [ ] Tambahkan *AABB Bounding Box Collision* untuk membatasi gerakan agar tidak tembus dinding luar dan partisi.

### 🔹 STEP 2 — PBR Smart House Interior Scene Builder
- [ ] Buat modul `web/src/components/fpp/InteriorScene.ts`.
- [ ] Render 4 ruangan lengkap dengan lantai bertekstur, plester dinding, dan ambang pintu antar ruangan.
- [ ] Tambahkan perabotan 3D detail:
  - Dapur: Meja kabinet kompor, kulkas, wastafel cuci piring.
  - Kamar Tidur: Tempat tidur dengan selimut, lampu nakas (*lamp*), unit pemanas dinding (*radiator heater*).
  - Kamar Mandi: Area shower/bath dan cermin.
  - Ruang Keluarga/Lorong: Sofa panjang, meja kopi, dan Smart Speaker box.
- [ ] Atur material `MeshStandardMaterial` dengan parameter `roughness = 0.4`, `metalness = 0.1` untuk pantulan pencahayaan fotorealistik.

### 🔹 STEP 3 — Dynamic Lighting & Visual Device States
- [ ] Pasang sumber cahaya nyata di setiap ruangan:
  - Lampu plafon Dapur: `PointLight` (Warna `#F2B33D`, radius 10m).
  - Lampu plafon Kamar Tidur: `PointLight` + Lampu meja tidur `SpotLight`.
  - Lampu Kamar Mandi: `PointLight` terarah.
  - Pemanas Dinding: Emissive glow warna merah bata (`#C4492F`) dengan animasi pulsa suhu.
  - Smart Speaker Lorong: Cincin partikel suara / scale pulsator saat musik menyala (`PLAY`).
- [ ] Hubungkan properti intensitas cahaya secara reaktif ke `devices: DeviceState` dari React props.

### 🔹 STEP 4 — Speech Control via Spacebar & In-Game HUD Overlay
- [ ] Pasang listener tombol `Spacebar` (tahan untuk rekam suara, lepas untuk kirim audio).
- [ ] Buat antarmuka HUD Game:
  - Crosshair titik bidik (`+`) di tengah layar.
  - Indikator lokasi ruangan aktif berdasarkan koordinat X/Z pemain.
  - Voice Status Overlay: animasikan indikator saat pengguna menahan Spasi (*"Listening..."*).
  - Result Banner: menampilkan teks perintah suara yang berhasil dieksekusi model AI.
- [ ] Buat Mini-Map Radar di sudut layar menggunakan kamera orthographic sekunder yang memetakan posisi pemain secara real-time.

### 🔹 STEP 5 — First-Person Arms & Smartwatch Walkie-Talkie
- [ ] Pasang model lengan 3D di layer kamera FPP agar bergerak mengikuti arah pandang.
- [ ] Desain jam tangan pintar (*Smartwatch*) di pergelangan tangan kiri dengan layar emissive yang menampilkan logo SVARA dan audio waveform.
- [ ] Buat animasi perpindahan posisi tangan (*lerp translation & rotation*): lengan terangkat mendekat ke kamera saat Spasi ditekan dan kembali ke posisi santai saat dilepas.

### 🔹 STEP 6 — View Toggle, Optimasi WebGL, & Testing
- [ ] Sediakan tombol toggle / shortcut `V` untuk berpindah antara mode FPP, Isometrik 3D, dan 2D SVG.
- [ ] Optimalkan alokasi memori WebGL (dispose geometry, texture, dan material saat unmount).
- [ ] Uji responsivitas performa frame-rate (target solid 60 FPS pada GPU laptop).
- [ ] Lakukan pengujian end-to-end: jalan ke kamar tidur ➡️ tekan Spasi ➡️ *"Turn on the bedroom lights"* ➡️ lampu menyala seketika di hadapan pemain.
