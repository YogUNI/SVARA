"""Interactive CLI Voice Recorder for SVARA Own Recordings Dataset (Bahasa Indonesia).

Task: Sub-step 13.2 / P5-02
Reference: docs/01 §8, docs/10 P5-02, docs/14 FASE 13
Usage:
    python scripts/record_own_voice.py --speaker haikal --condition quiet --take 1
    python scripts/record_own_voice.py --speaker haikal --condition quiet --take 2
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import soundfile as sf

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

# 31 Perintah Smart Home Bahasa Indonesia Terstruktur per Ruangan & Efek Fitur Web Demo
PHRASES = [
    # --- ZONA KAMAR TIDUR (BEDROOM) ---
    {
        "intent_id": 0,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Lampu Meja / Lampu Belajar MENYALA",
        "action": "activate", "object": "lamp", "location": "none",
        "phrase": "Nyalain lampu meja dong",
        "baku": "Nyalakan lampu meja",
        "nyantai": "Nyalain lampu meja dong",
        "slang": "Tolong hidupin lampu belajar ya",
    },
    {
        "intent_id": 1,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Lampu Utama Plafon Kamar MENYALA",
        "action": "activate", "object": "lights", "location": "bedroom",
        "phrase": "Nyalain lampu kamar dong",
        "baku": "Nyalakan lampu kamar tidur",
        "nyantai": "Nyalain lampu kamar dong",
        "slang": "Idupin lampu kamar tidur ya",
    },
    {
        "intent_id": 15,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Lampu Meja / Lampu Belajar PADAM",
        "action": "deactivate", "object": "lamp", "location": "none",
        "phrase": "Matiin lampu meja dong",
        "baku": "Matikan lampu meja",
        "nyantai": "Matiin lampu meja dong",
        "slang": "Tolong matiin lampu belajar ya",
    },
    {
        "intent_id": 16,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Lampu Utama Plafon Kamar PADAM",
        "action": "deactivate", "object": "lights", "location": "bedroom",
        "phrase": "Matiin lampu kamar tidur dong",
        "baku": "Matikan lampu kamar tidur",
        "nyantai": "Matiin lampu kamar tidur dong",
        "slang": "Padamin lampu di kamar ya",
    },
    {
        "intent_id": 26,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Suhu / Pemanas Kamar Tidur NAIK (+2°C)",
        "action": "increase", "object": "heat", "location": "bedroom",
        "phrase": "Kamar dingin naikin suhunya dong",
        "baku": "Naikkan suhu kamar tidur",
        "nyantai": "Kamar dingin naikin suhunya dong",
        "slang": "Panasin suhu kamar tidur ya",
    },
    {
        "intent_id": 21,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Suhu / Pendingin Kamar Tidur TURUN (-2°C)",
        "action": "decrease", "object": "heat", "location": "bedroom",
        "phrase": "Dinginkan kamar tidur dong",
        "baku": "Turunkan suhu kamar tidur",
        "nyantai": "Dinginkan kamar tidur dong",
        "slang": "Kamar gerah banget tolong sejukin",
    },

    # --- ZONA DAPUR (KITCHEN) ---
    {
        "intent_id": 2,
        "room": "DAPUR (KITCHEN)",
        "feature": "Lampu Plafon Dapur MENYALA",
        "action": "activate", "object": "lights", "location": "kitchen",
        "phrase": "Nyalain lampu di dapur ya",
        "baku": "Nyalakan lampu dapur",
        "nyantai": "Nyalain lampu di dapur ya",
        "slang": "Tolong idupin lampu dapur dong",
    },
    {
        "intent_id": 17,
        "room": "DAPUR (KITCHEN)",
        "feature": "Lampu Plafon Dapur PADAM",
        "action": "deactivate", "object": "lights", "location": "kitchen",
        "phrase": "Matiin lampu dapur dong",
        "baku": "Matikan lampu dapur",
        "nyantai": "Matiin lampu dapur dong",
        "slang": "Padamin lampu di dapur ya",
    },
    {
        "intent_id": 27,
        "room": "DAPUR (KITCHEN)",
        "feature": "Suhu / Pemanas Dapur NAIK (+2°C)",
        "action": "increase", "object": "heat", "location": "kitchen",
        "phrase": "Panasin suhu di dapur dong",
        "baku": "Naikkan suhu dapur",
        "nyantai": "Panasin suhu di dapur dong",
        "slang": "Naikin temperatur dapur ya",
    },
    {
        "intent_id": 22,
        "room": "DAPUR (KITCHEN)",
        "feature": "Suhu / Pendingin Dapur TURUN (-2°C)",
        "action": "decrease", "object": "heat", "location": "kitchen",
        "phrase": "Dinginin dapur dong",
        "baku": "Turunkan suhu dapur",
        "nyantai": "Dinginin dapur dong",
        "slang": "Dapur gerah nih tolong sejukin",
    },

    # --- ZONA KAMAR MANDI (WASHROOM / TOILET) ---
    {
        "intent_id": 4,
        "room": "KAMAR MANDI (WASHROOM)",
        "feature": "Lampu Kamar Mandi MENYALA",
        "action": "activate", "object": "lights", "location": "washroom",
        "phrase": "Nyalain lampu kamar mandi dong",
        "baku": "Nyalakan lampu kamar mandi",
        "nyantai": "Nyalain lampu kamar mandi dong",
        "slang": "Idupin lampu toilet ya",
    },
    {
        "intent_id": 19,
        "room": "KAMAR MANDI (WASHROOM)",
        "feature": "Lampu Kamar Mandi PADAM",
        "action": "deactivate", "object": "lights", "location": "washroom",
        "phrase": "Matiin lampu kamar mandi dong",
        "baku": "Matikan lampu kamar mandi",
        "nyantai": "Matiin lampu kamar mandi dong",
        "slang": "Tolong matiin lampu toilet ya",
    },
    {
        "intent_id": 29,
        "room": "KAMAR MANDI (WASHROOM)",
        "feature": "Suhu Kamar Mandi NAIK (+2°C)",
        "action": "increase", "object": "heat", "location": "washroom",
        "phrase": "Kamar mandi dingin tolong hangatkan",
        "baku": "Naikkan suhu kamar mandi",
        "nyantai": "Kamar mandi dingin tolong hangatkan",
        "slang": "Panasin suhu kamar mandi ya",
    },
    {
        "intent_id": 24,
        "room": "KAMAR MANDI (WASHROOM)",
        "feature": "Suhu Kamar Mandi TURUN (-2°C)",
        "action": "decrease", "object": "heat", "location": "washroom",
        "phrase": "Dinginkan kamar mandi dong",
        "baku": "Turunkan suhu kamar mandi",
        "nyantai": "Dinginkan kamar mandi dong",
        "slang": "Sejukin suhu di kamar mandi ya",
    },

    # --- ZONA LORONG & AUDIO (HALLWAY / MEDIA) ---
    {
        "intent_id": 5,
        "room": "LORONG & AUDIO (HALLWAY)",
        "feature": "Speaker Musik Lorong MENYALA",
        "action": "activate", "object": "music", "location": "none",
        "phrase": "Putar lagi lagunya dong",
        "baku": "Nyalakan musik",
        "nyantai": "Putar lagi lagunya dong",
        "slang": "Setel musik dong bro",
    },
    {
        "intent_id": 20,
        "room": "LORONG & AUDIO (HALLWAY)",
        "feature": "Speaker Musik Lorong PADAM (Stop)",
        "action": "deactivate", "object": "music", "location": "none",
        "phrase": "Matiin lagunya dong",
        "baku": "Matikan musik",
        "nyantai": "Matiin lagunya dong",
        "slang": "Stop musiknya sekarang ya",
    },
    {
        "intent_id": 30,
        "room": "LORONG & AUDIO (HALLWAY)",
        "feature": "Volume Speaker NAIK (+2 Bar)",
        "action": "increase", "object": "volume", "location": "none",
        "phrase": "Kencengin suaranya dong",
        "baku": "Besarkan volume suara",
        "nyantai": "Kencengin suaranya dong",
        "slang": "Gedein volumenya dikit bro",
    },
    {
        "intent_id": 25,
        "room": "LORONG & AUDIO (HALLWAY)",
        "feature": "Volume Speaker TURUN (-2 Bar)",
        "action": "decrease", "object": "volume", "location": "none",
        "phrase": "Pelanin suaranya dong",
        "baku": "Kecilkan volume suara",
        "nyantai": "Pelanin suaranya dong",
        "slang": "Kecilin volumenya ya bro",
    },

    # --- ZONA SELURUH RUMAH (MASTER / GLOBAL) ---
    {
        "intent_id": 3,
        "room": "SELURUH RUMAH (MASTER)",
        "feature": "SEMUA Lampu Rumah MENYALA",
        "action": "activate", "object": "lights", "location": "none",
        "phrase": "Nyalain semua lampu rumah dong",
        "baku": "Nyalakan semua lampu",
        "nyantai": "Nyalain semua lampu rumah dong",
        "slang": "Terangin seluruh ruangan ya",
    },
    {
        "intent_id": 18,
        "room": "SELURUH RUMAH (MASTER)",
        "feature": "SEMUA Lampu Rumah PADAM",
        "action": "deactivate", "object": "lights", "location": "none",
        "phrase": "Matiin semua lampu dong",
        "baku": "Matikan semua lampu",
        "nyantai": "Matiin semua lampu dong",
        "slang": "Padamin seluruh lampu rumah ya",
    },
    {
        "intent_id": 28,
        "room": "SELURUH RUMAH (MASTER)",
        "feature": "Suhu Rumah NAIK (+2°C)",
        "action": "increase", "object": "heat", "location": "none",
        "phrase": "Rumah dingin tolong naikin suhunya",
        "baku": "Naikkan suhu ruangan",
        "nyantai": "Rumah dingin tolong naikin suhunya",
        "slang": "Hangatkan seluruh rumah ya",
    },
    {
        "intent_id": 23,
        "room": "SELURUH RUMAH (MASTER)",
        "feature": "Suhu Rumah TURUN (-2°C)",
        "action": "decrease", "object": "heat", "location": "none",
        "phrase": "Turunkan suhunya dong",
        "baku": "Turunkan suhu ruangan",
        "nyantai": "Turunkan suhunya dong",
        "slang": "Rumah gerah tolong sejukin ya",
    },

    # --- PENGATURAN BAHASA ASISTEN (SYSTEM LANGUAGE) ---
    {
        "intent_id": 14,
        "room": "PENGATURAN SISTEM",
        "feature": "Ganti Bahasa Sistem",
        "action": "change language", "object": "none", "location": "none",
        "phrase": "Ganti bahasanya dong",
        "baku": "Ganti bahasa sistem",
        "nyantai": "Ganti bahasanya dong",
        "slang": "Ubah setelan bahasa ya",
    },
    {
        "intent_id": 11,
        "room": "PENGATURAN SISTEM",
        "feature": "Setel Bahasa ke INGGRIS",
        "action": "change language", "object": "English", "location": "none",
        "phrase": "Ganti bahasanya ke Bahasa Inggris",
        "baku": "Ganti ke Bahasa Inggris",
        "nyantai": "Ganti bahasanya ke Bahasa Inggris",
        "slang": "Setel bahasa Inggris dong",
    },
    {
        "intent_id": 10,
        "room": "PENGATURAN SISTEM",
        "feature": "Setel Bahasa ke MANDARIN",
        "action": "change language", "object": "Chinese", "location": "none",
        "phrase": "Ganti ke bahasa Mandarin dong",
        "baku": "Ganti ke Bahasa Mandarin",
        "nyantai": "Ganti ke Bahasa Mandarin dong",
        "slang": "Set bahasa ke Mandarin ya",
    },
    {
        "intent_id": 12,
        "room": "PENGATURAN SISTEM",
        "feature": "Setel Bahasa ke JERMAN",
        "action": "change language", "object": "German", "location": "none",
        "phrase": "Ganti bahasanya ke Bahasa Jerman",
        "baku": "Ganti ke Bahasa Jerman",
        "nyantai": "Ganti bahasanya ke Bahasa Jerman",
        "slang": "Ubah ke bahasa Jerman dong",
    },
    {
        "intent_id": 13,
        "room": "PENGATURAN SISTEM",
        "feature": "Setel Bahasa ke KOREA",
        "action": "change language", "object": "Korean", "location": "none",
        "phrase": "Ganti ke bahasa Korea dong",
        "baku": "Ganti ke Bahasa Korea",
        "nyantai": "Ganti ke Bahasa Korea dong",
        "slang": "Setel bahasa Korea ya",
    },

    # --- BENCHMARK INTERNASIONAL FSC: ROBOT PELAYAN & OBJEK 3D FPP ---
    {
        "intent_id": 6,
        "room": "ROBOT PELAYAN / BENCHMARK FSC",
        "feature": "Ambil Gelas Jus (Meja Tamu FPP)",
        "action": "bring", "object": "juice", "location": "none",
        "phrase": "Ambilkan jus dingin dong",
        "baku": "Bawakan saya jus",
        "nyantai": "Ambilkan jus dingin dong",
        "slang": "Bawain jus jeruk ya bro",
    },
    {
        "intent_id": 7,
        "room": "ROBOT PELAYAN / BENCHMARK FSC",
        "feature": "Ambil Koran Cetak (Meja Kopi FPP)",
        "action": "bring", "object": "newspaper", "location": "none",
        "phrase": "Tolong ambilkan koran dong",
        "baku": "Bawakan koran untuk saya",
        "nyantai": "Tolong ambilkan koran dong",
        "slang": "Bawain koran pagi ini ya",
    },
    {
        "intent_id": 8,
        "room": "ROBOT PELAYAN / BENCHMARK FSC",
        "feature": "Ambil Sepatu (Rak Sepatu Masuk FPP)",
        "action": "bring", "object": "shoes", "location": "none",
        "phrase": "Tolong bawakan sepatu dong",
        "baku": "Ambilkan sepatu saya",
        "nyantai": "Tolong bawakan sepatu dong",
        "slang": "Bawain sepatu gua ya bro",
    },
    {
        "intent_id": 9,
        "room": "ROBOT PELAYAN / BENCHMARK FSC",
        "feature": "Ambil Kaus Kaki (Lemari Pakaian FPP)",
        "action": "bring", "object": "socks", "location": "none",
        "phrase": "Tolong bawakan kaus kaki dong",
        "baku": "Ambilkan kaus kaki",
        "nyantai": "Tolong bawakan kaus kaki dong",
        "slang": "Ambilin kaos kaki bersih ya",
    },
]


def record_clip(duration_s: float = 3.5, sr: int = 16000) -> np.ndarray:
    """Records audio from default input device."""
    if not HAS_SOUNDDEVICE:
        raise RuntimeError("sounddevice is not installed. Run: pip install sounddevice")
    num_samples = int(duration_s * sr)
    recording = sd.rec(num_samples, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return recording.squeeze()


def main():
    parser = argparse.ArgumentParser(description="Perekam Suara Bahasa Indonesia untuk Kalibrasi Akustik SVARA")
    parser.add_argument("--speaker", type=str, default="haikal", help="ID Pembicara (contoh: haikal, yoga)")
    parser.add_argument("--condition", type=str, choices=["quiet", "noisy"], default="quiet", help="Kondisi rekaman (quiet: tenang, noisy: berisik)")
    parser.add_argument("--device", type=str, default="laptop_mic", help="Device mic (laptop_mic, phone, headset)")
    parser.add_argument("--gender", type=str, default="male", help="Jenis kelamin pembicara (male/female)")
    parser.add_argument("--age-range", type=str, default="20-25", help="Rentang usia pembicara")
    parser.add_argument("--duration", type=float, default=3.5, help="Durasi perekaman per klip (detik)")
    parser.add_argument("--take", type=int, default=1, help="Nomor pengambilan / variasi rekaman (contoh: 1, 2, 3)")
    parser.add_argument("--start-index", type=int, default=0, help="Mulai dari intent ke-n (0-30)")
    parser.add_argument("--out-dir", type=str, default="data/own_recordings/wavs", help="Folder output WAV")
    parser.add_argument("--metadata-csv", type=str, default="data/own_recordings/metadata.csv", help="Lokasi metadata CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_csv), exist_ok=True)

    cond_code = "q" if args.condition == "quiet" else "n"
    speaker_id = f"own_{args.speaker.lower()}"

    print("=" * 75)
    print("      SVARA PEREKAM SUARA ASLI SMART HOME (PANDUAN LENGKAP FITUR WEB)")
    print("=" * 75)
    print(f"Pembicara : {args.speaker.upper()} ({speaker_id})")
    print(f"Kondisi   : {args.condition.upper()} (Kode: {cond_code}) | Putaran (Take): #{args.take}")
    print(f"Device    : {args.device} | Durasi: {args.duration}s per rekaman")
    print(f"Target Dir: {args.out_dir}")
    print("=" * 75)
    print("Petunjuk Singkat:")
    print("1. Layar akan menampilkan RUANGAN, TARGET EFEK DI WEB, dan 3 PILIHAN KALIMAT.")
    print("2. Pilih satu kalimat yang paling enak kamu ucapkan (Baku / Nyantai / Slang).")
    print("3. Tekan [ENTER], ucapkan dengan wajar, santai, dan jelas.")
    print("=" * 75)

    metadata_records = []

    for idx in range(args.start_index, len(PHRASES)):
        item = PHRASES[idx]
        intent_id = item["intent_id"]
        room = item["room"]
        feature = item["feature"]
        baku = item["baku"]
        nyantai = item["nyantai"]
        slang = item["slang"]
        act = item["action"]
        obj = item["object"]
        loc = item["location"]

        take_suffix = f"_t{args.take}" if args.take > 1 else ""
        filename = f"{speaker_id}_{cond_code}_{intent_id:02d}{take_suffix}.wav"
        file_path = os.path.join(args.out_dir, filename)

        while True:
            print("\n" + "=" * 75)
            print(f"👉 NOMOR [{idx + 1}/31] • INTENT {intent_id:02d}")
            print(f"📍 RUANGAN   : {room}")
            print(f"🎯 EFEK WEB  : {feature}")
            print(f"⚙️  SLOT AI   : action='{act}', object='{obj}', location='{loc}'")
            print("-" * 75)
            print("PILIHAN KALIMAT (Ucapkan salah satu):")
            print(f"  [A] Versi Baku    : \"{baku}\"")
            print(f"  [B] Versi Nyantai : \"{nyantai}\"  <-- Paling Rekomendasi")
            print(f"  [C] Versi Slang   : \"{slang}\"")
            print("-" * 75)
            user_cmd = input("Tekan [ENTER] untuk mulai rekam (atau 's': lewati / 'q': keluar): ").strip().lower()

            if user_cmd == "q":
                print("Perekaman dihentikan oleh pengguna.")
                return
            if user_cmd == "s":
                print(f"Melewati intent {intent_id:02d}.")
                break

            print(f"\n🎙️  [MEREKAM {args.duration}s] ... Silakan bicara sekarang!")
            audio = record_clip(duration_s=args.duration, sr=16000)
            print("✅ Rekaman selesai!")

            # Quick audio check
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio**2)))
            rms_db = 20 * np.log10(rms + 1e-9)

            print(f"   📊 Peak Level: {peak:.3f} | RMS: {rms_db:.1f} dBFS")
            if peak < 0.05:
                print("   ⚠️ Peringatan: Suara terlalu pelan! Dekatkan bibir ke mikrofon.")
            elif peak > 0.98:
                print("   ⚠️ Peringatan: Audio hampir clipping! Jauhkan mikrofon sedikit.")

            # Option to review/retake
            review = input("Simpan rekaman ini? [ENTER: Ya, 'r': Ulangi, 's': Lewati]: ").strip().lower()
            if review == "r":
                print("Mengulang rekaman...")
                continue
            if review == "s":
                print("Dilewati.")
                break

            # Anti-clipping peak guard (<= 0.95)
            if peak > 0.95:
                audio = audio * (0.95 / peak)

            # Save WAV
            sf.write(file_path, audio, samplerate=16000, subtype="PCM_16")
            print(f"💾 Tersimpan ke: {file_path}")

            metadata_records.append({
                "path": os.path.relpath(file_path, start=".").replace("\\", "/"),
                "speakerId": speaker_id,
                "gender": args.gender,
                "condition": args.condition,
                "transcript": nyantai,
                "action": act,
                "object": obj,
                "location": loc,
                "intent_id": intent_id,
                "room": room,
                "feature": feature,
            })
            break

    # Append to metadata CSV if any recordings made
    if metadata_records:
        df_new = pd.DataFrame(metadata_records)
        if os.path.exists(args.metadata_csv):
            df_old = pd.read_csv(args.metadata_csv)
            df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["path"], keep="last")
        else:
            df_combined = df_new

        df_combined.to_csv(args.metadata_csv, index=False)
        print("\n" + "=" * 75)
        print(f"🎉 SUKSES! {len(metadata_records)} rekaman telah disimpan dan didaftarkan ke:")
        print(f"   {args.metadata_csv}")
        print("=" * 75)


if __name__ == "__main__":
    main()
