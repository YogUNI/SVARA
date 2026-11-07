"""Interactive CLI Voice Recorder for SVARA (Triple-Take: Baku, Nyantai, Slang).

Task: Sub-step 13.2 / P5-02
Reference: docs/01 §8, docs/10 P5-02, docs/14 FASE 13

Fitur Utama:
- Merekam 3 variasi berturut-turut untuk setiap intent:
  1. Versi Baku (Formal & Presisi)
  2. Versi Nyantai (Casual Sehari-hari)
  3. Versi Slang (Gaul & Spontan)
- Menghasilkan 93 rekaman vokal manusia asli berkualitas tinggi (31 intent x 3 variasi)
- Format: 16 kHz Mono PCM_16 anti-clipping (Peak <= 0.95)
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

# 31 Intent Lengkap dengan 3 Variasi Kalimat Nyata per Ruangan
PHRASES = [
    # --- ZONA KAMAR TIDUR (BEDROOM) ---
    {
        "intent_id": 0,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Lampu Meja / Belajar MENYALA",
        "action": "activate", "object": "lamp", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Nyalakan lampu meja"},
            {"style": "nyantai", "label": "Nyantai", "text": "Nyalain lampu meja dong"},
            {"style": "slang", "label": "Slang", "text": "Tolong hidupin lampu belajar ya"},
        ]
    },
    {
        "intent_id": 1,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Lampu Utama Plafon Kamar MENYALA",
        "action": "activate", "object": "lights", "location": "bedroom",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Nyalakan lampu kamar tidur"},
            {"style": "nyantai", "label": "Nyantai", "text": "Nyalain lampu kamar dong"},
            {"style": "slang", "label": "Slang", "text": "Idupin lampu kamar tidur ya"},
        ]
    },
    {
        "intent_id": 15,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Lampu Meja / Belajar PADAM",
        "action": "deactivate", "object": "lamp", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Matikan lampu meja"},
            {"style": "nyantai", "label": "Nyantai", "text": "Matiin lampu meja dong"},
            {"style": "slang", "label": "Slang", "text": "Tolong matiin lampu belajar ya"},
        ]
    },
    {
        "intent_id": 16,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Lampu Utama Plafon Kamar PADAM",
        "action": "deactivate", "object": "lights", "location": "bedroom",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Matikan lampu kamar tidur"},
            {"style": "nyantai", "label": "Nyantai", "text": "Matiin lampu kamar tidur dong"},
            {"style": "slang", "label": "Slang", "text": "Padamin lampu di kamar ya"},
        ]
    },
    {
        "intent_id": 26,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Suhu / Pemanas Kamar Tidur NAIK (+2°C)",
        "action": "increase", "object": "heat", "location": "bedroom",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Naikkan suhu kamar tidur"},
            {"style": "nyantai", "label": "Nyantai", "text": "Kamar dingin naikin suhunya dong"},
            {"style": "slang", "label": "Slang", "text": "Panasin suhu kamar tidur ya"},
        ]
    },
    {
        "intent_id": 21,
        "room": "KAMAR TIDUR (BEDROOM)",
        "feature": "Suhu / Pendingin Kamar Tidur TURUN (-2°C)",
        "action": "decrease", "object": "heat", "location": "bedroom",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Turunkan suhu kamar tidur"},
            {"style": "nyantai", "label": "Nyantai", "text": "Dinginkan kamar tidur dong"},
            {"style": "slang", "label": "Slang", "text": "Kamar gerah banget tolong sejukin"},
        ]
    },

    # --- ZONA DAPUR (KITCHEN) ---
    {
        "intent_id": 2,
        "room": "DAPUR (KITCHEN)",
        "feature": "Lampu Plafon Dapur MENYALA",
        "action": "activate", "object": "lights", "location": "kitchen",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Nyalakan lampu dapur"},
            {"style": "nyantai", "label": "Nyantai", "text": "Nyalain lampu di dapur ya"},
            {"style": "slang", "label": "Slang", "text": "Tolong idupin lampu dapur dong"},
        ]
    },
    {
        "intent_id": 17,
        "room": "DAPUR (KITCHEN)",
        "feature": "Lampu Plafon Dapur PADAM",
        "action": "deactivate", "object": "lights", "location": "kitchen",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Matikan lampu dapur"},
            {"style": "nyantai", "label": "Nyantai", "text": "Matiin lampu dapur dong"},
            {"style": "slang", "label": "Slang", "text": "Padamin lampu di dapur ya"},
        ]
    },
    {
        "intent_id": 27,
        "room": "DAPUR (KITCHEN)",
        "feature": "Suhu / Pemanas Dapur NAIK (+2°C)",
        "action": "increase", "object": "heat", "location": "kitchen",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Naikkan suhu dapur"},
            {"style": "nyantai", "label": "Nyantai", "text": "Panasin suhu di dapur dong"},
            {"style": "slang", "label": "Slang", "text": "Naikin temperatur dapur ya"},
        ]
    },
    {
        "intent_id": 22,
        "room": "DAPUR (KITCHEN)",
        "feature": "Suhu / Pendingin Dapur TURUN (-2°C)",
        "action": "decrease", "object": "heat", "location": "kitchen",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Turunkan suhu dapur"},
            {"style": "nyantai", "label": "Nyantai", "text": "Dinginin dapur dong"},
            {"style": "slang", "label": "Slang", "text": "Dapur gerah nih tolong sejukin"},
        ]
    },

    # --- ZONA KAMAR MANDI (WASHROOM / TOILET) ---
    {
        "intent_id": 4,
        "room": "KAMAR MANDI (WASHROOM)",
        "feature": "Lampu Kamar Mandi MENYALA",
        "action": "activate", "object": "lights", "location": "washroom",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Nyalakan lampu kamar mandi"},
            {"style": "nyantai", "label": "Nyantai", "text": "Nyalain lampu kamar mandi dong"},
            {"style": "slang", "label": "Slang", "text": "Idupin lampu toilet ya"},
        ]
    },
    {
        "intent_id": 19,
        "room": "KAMAR MANDI (WASHROOM)",
        "feature": "Lampu Kamar Mandi PADAM",
        "action": "deactivate", "object": "lights", "location": "washroom",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Matikan lampu kamar mandi"},
            {"style": "nyantai", "label": "Nyantai", "text": "Matiin lampu kamar mandi dong"},
            {"style": "slang", "label": "Slang", "text": "Tolong matiin lampu toilet ya"},
        ]
    },
    {
        "intent_id": 29,
        "room": "KAMAR MANDI (WASHROOM)",
        "feature": "Suhu Kamar Mandi NAIK (+2°C)",
        "action": "increase", "object": "heat", "location": "washroom",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Naikkan suhu kamar mandi"},
            {"style": "nyantai", "label": "Nyantai", "text": "Kamar mandi dingin tolong hangatkan"},
            {"style": "slang", "label": "Slang", "text": "Panasin suhu kamar mandi ya"},
        ]
    },
    {
        "intent_id": 24,
        "room": "KAMAR MANDI (WASHROOM)",
        "feature": "Suhu Kamar Mandi TURUN (-2°C)",
        "action": "decrease", "object": "heat", "location": "washroom",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Turunkan suhu kamar mandi"},
            {"style": "nyantai", "label": "Nyantai", "text": "Dinginkan kamar mandi dong"},
            {"style": "slang", "label": "Slang", "text": "Sejukin suhu di kamar mandi ya"},
        ]
    },

    # --- ZONA LORONG & AUDIO (HALLWAY / MEDIA) ---
    {
        "intent_id": 5,
        "room": "LORONG & AUDIO (HALLWAY)",
        "feature": "Speaker Musik Lorong MENYALA",
        "action": "activate", "object": "music", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Nyalakan musik"},
            {"style": "nyantai", "label": "Nyantai", "text": "Putar lagi lagunya dong"},
            {"style": "slang", "label": "Slang", "text": "Setel musik dong bro"},
        ]
    },
    {
        "intent_id": 20,
        "room": "LORONG & AUDIO (HALLWAY)",
        "feature": "Speaker Musik Lorong PADAM (Stop)",
        "action": "deactivate", "object": "music", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Matikan musik"},
            {"style": "nyantai", "label": "Nyantai", "text": "Matiin lagunya dong"},
            {"style": "slang", "label": "Slang", "text": "Stop musiknya sekarang ya"},
        ]
    },
    {
        "intent_id": 30,
        "room": "LORONG & AUDIO (HALLWAY)",
        "feature": "Volume Speaker NAIK (+2 Bar)",
        "action": "increase", "object": "volume", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Besarkan volume suara"},
            {"style": "nyantai", "label": "Nyantai", "text": "Kencengin suaranya dong"},
            {"style": "slang", "label": "Slang", "text": "Gedein volumenya dikit bro"},
        ]
    },
    {
        "intent_id": 25,
        "room": "LORONG & AUDIO (HALLWAY)",
        "feature": "Volume Speaker TURUN (-2 Bar)",
        "action": "decrease", "object": "volume", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Kecilkan volume suara"},
            {"style": "nyantai", "label": "Nyantai", "text": "Pelanin suaranya dong"},
            {"style": "slang", "label": "Slang", "text": "Kecilin volumenya ya bro"},
        ]
    },

    # --- ZONA SELURUH RUMAH (MASTER / GLOBAL) ---
    {
        "intent_id": 3,
        "room": "SELURUH RUMAH (MASTER)",
        "feature": "SEMUA Lampu Rumah MENYALA",
        "action": "activate", "object": "lights", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Nyalakan semua lampu"},
            {"style": "nyantai", "label": "Nyantai", "text": "Nyalain semua lampu rumah dong"},
            {"style": "slang", "label": "Slang", "text": "Terangin seluruh ruangan ya"},
        ]
    },
    {
        "intent_id": 18,
        "room": "SELURUH RUMAH (MASTER)",
        "feature": "SEMUA Lampu Rumah PADAM",
        "action": "deactivate", "object": "lights", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Matikan semua lampu"},
            {"style": "nyantai", "label": "Nyantai", "text": "Matiin semua lampu dong"},
            {"style": "slang", "label": "Slang", "text": "Padamin seluruh lampu rumah ya"},
        ]
    },
    {
        "intent_id": 28,
        "room": "SELURUH RUMAH (MASTER)",
        "feature": "Suhu Rumah NAIK (+2°C)",
        "action": "increase", "object": "heat", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Naikkan suhu ruangan"},
            {"style": "nyantai", "label": "Nyantai", "text": "Rumah dingin tolong naikin suhunya"},
            {"style": "slang", "label": "Slang", "text": "Hangatkan seluruh rumah ya"},
        ]
    },
    {
        "intent_id": 23,
        "room": "SELURUH RUMAH (MASTER)",
        "feature": "Suhu Rumah TURUN (-2°C)",
        "action": "decrease", "object": "heat", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Turunkan suhu ruangan"},
            {"style": "nyantai", "label": "Nyantai", "text": "Turunkan suhunya dong"},
            {"style": "slang", "label": "Slang", "text": "Rumah gerah tolong sejukin ya"},
        ]
    },

    # --- PENGATURAN BAHASA ASISTEN (SYSTEM LANGUAGE) ---
    {
        "intent_id": 14,
        "room": "PENGATURAN SISTEM",
        "feature": "Ganti Bahasa Sistem",
        "action": "change language", "object": "none", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Ganti bahasa sistem"},
            {"style": "nyantai", "label": "Nyantai", "text": "Ganti bahasanya dong"},
            {"style": "slang", "label": "Slang", "text": "Ubah setelan bahasa ya"},
        ]
    },
    {
        "intent_id": 11,
        "room": "PENGATURAN SISTEM",
        "feature": "Setel Bahasa ke INGGRIS",
        "action": "change language", "object": "English", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Ganti ke Bahasa Inggris"},
            {"style": "nyantai", "label": "Nyantai", "text": "Ganti bahasanya ke Bahasa Inggris"},
            {"style": "slang", "label": "Slang", "text": "Setel bahasa Inggris dong"},
        ]
    },
    {
        "intent_id": 10,
        "room": "PENGATURAN SISTEM",
        "feature": "Setel Bahasa ke MANDARIN",
        "action": "change language", "object": "Chinese", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Ganti ke Bahasa Mandarin"},
            {"style": "nyantai", "label": "Nyantai", "text": "Ganti ke Bahasa Mandarin dong"},
            {"style": "slang", "label": "Slang", "text": "Set bahasa ke Mandarin ya"},
        ]
    },
    {
        "intent_id": 12,
        "room": "PENGATURAN SISTEM",
        "feature": "Setel Bahasa ke JERMAN",
        "action": "change language", "object": "German", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Ganti ke Bahasa Jerman"},
            {"style": "nyantai", "label": "Nyantai", "text": "Ganti bahasanya ke Bahasa Jerman"},
            {"style": "slang", "label": "Slang", "text": "Ubah ke bahasa Jerman dong"},
        ]
    },
    {
        "intent_id": 13,
        "room": "PENGATURAN SISTEM",
        "feature": "Setel Bahasa ke KOREA",
        "action": "change language", "object": "Korean", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Ganti ke Bahasa Korea"},
            {"style": "nyantai", "label": "Nyantai", "text": "Ganti ke Bahasa Korea dong"},
            {"style": "slang", "label": "Slang", "text": "Setel bahasa Korea ya"},
        ]
    },

    # --- BENCHMARK INTERNASIONAL FSC: ROBOT PELAYAN & OBJEK 3D FPP ---
    {
        "intent_id": 6,
        "room": "ROBOT PELAYAN / BENCHMARK FSC",
        "feature": "Ambil Gelas Jus (Meja Tamu FPP)",
        "action": "bring", "object": "juice", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Bawakan saya jus"},
            {"style": "nyantai", "label": "Nyantai", "text": "Ambilkan jus dingin dong"},
            {"style": "slang", "label": "Slang", "text": "Bawain jus jeruk ya bro"},
        ]
    },
    {
        "intent_id": 7,
        "room": "ROBOT PELAYAN / BENCHMARK FSC",
        "feature": "Ambil Koran Cetak (Meja Kopi FPP)",
        "action": "bring", "object": "newspaper", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Bawakan koran untuk saya"},
            {"style": "nyantai", "label": "Nyantai", "text": "Tolong ambilkan koran dong"},
            {"style": "slang", "label": "Slang", "text": "Bawain koran pagi ini ya"},
        ]
    },
    {
        "intent_id": 8,
        "room": "ROBOT PELAYAN / BENCHMARK FSC",
        "feature": "Ambil Sepatu (Rak Sepatu Masuk FPP)",
        "action": "bring", "object": "shoes", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Ambilkan sepatu saya"},
            {"style": "nyantai", "label": "Nyantai", "text": "Tolong bawakan sepatu dong"},
            {"style": "slang", "label": "Slang", "text": "Bawain sepatu gua ya bro"},
        ]
    },
    {
        "intent_id": 9,
        "room": "ROBOT PELAYAN / BENCHMARK FSC",
        "feature": "Ambil Kaus Kaki (Lemari Pakaian FPP)",
        "action": "bring", "object": "socks", "location": "none",
        "variations": [
            {"style": "baku", "label": "Baku", "text": "Ambilkan kaus kaki"},
            {"style": "nyantai", "label": "Nyantai", "text": "Tolong bawakan kaus kaki dong"},
            {"style": "slang", "label": "Slang", "text": "Ambilin kaos kaki bersih ya"},
        ]
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
    parser = argparse.ArgumentParser(description="Perekam Suara Triple-Take (Baku, Nyantai, Slang) untuk SVARA")
    parser.add_argument("--speaker", type=str, default="haikal", help="ID Pembicara (contoh: haikal, yoga)")
    parser.add_argument("--condition", type=str, choices=["quiet", "noisy"], default="quiet", help="Kondisi rekaman (quiet: tenang, noisy: berisik)")
    parser.add_argument("--device", type=str, default="laptop_mic", help="Device mic (laptop_mic, phone, headset)")
    parser.add_argument("--gender", type=str, default="male", help="Jenis kelamin pembicara (male/female)")
    parser.add_argument("--age-range", type=str, default="20-25", help="Rentang usia pembicara")
    parser.add_argument("--duration", type=float, default=3.5, help="Durasi perekaman per klip (detik)")
    parser.add_argument("--start-index", type=int, default=0, help="Mulai dari nomor intent ke-n (0-30)")
    parser.add_argument("--style", type=str, choices=["all", "baku", "nyantai", "slang"], default="all", help="Gaya yang ingin direkam (all: rekam ketiga-tiganya berturut-turut)")
    parser.add_argument("--out-dir", type=str, default="data/own_recordings/wavs", help="Folder output WAV")
    parser.add_argument("--metadata-csv", type=str, default="data/own_recordings/metadata.csv", help="Lokasi metadata CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_csv), exist_ok=True)

    cond_code = "q" if args.condition == "quiet" else "n"
    speaker_id = f"own_{args.speaker.lower()}"

    print("=" * 80)
    print("      SVARA PEREKAM SUARA ASLI SMART HOME (TRIPLE-TAKE PIPELINE)")
    print("=" * 80)
    print(f"Pembicara : {args.speaker.upper()} ({speaker_id})")
    print(f"Kondisi   : {args.condition.upper()} (Kode: {cond_code})")
    print(f"Mode Gaya : {args.style.upper()} ({'3 Rekaman (Baku, Nyantai, Slang)' if args.style == 'all' else 'Hanya 1 Gaya'})")
    print(f"Device    : {args.device} | Durasi per rekaman: {args.duration} detik")
    print(f"Target Dir: {args.out_dir}")
    print("=" * 80)
    print("Petunjuk Kerja:")
    print("Untuk SETIAP intent, kamu akan merekam berturut-turut:")
    print("  1. Versi Baku    (Jelas & Formal)")
    print("  2. Versi Nyantai (Santai Sehari-hari)")
    print("  3. Versi Slang   (Gaul & Spontan)")
    print("Tekan [ENTER], ucapkan dengan jelas dan wajar. Total 31 intent x 3 = 93 klip suara!")
    print("=" * 80)

    metadata_records = []

    for idx in range(args.start_index, len(PHRASES)):
        item = PHRASES[idx]
        intent_id = item["intent_id"]
        room = item["room"]
        feature = item["feature"]
        act = item["action"]
        obj = item["object"]
        loc = item["location"]

        # Filter variations if user specified single style
        vars_to_record = item["variations"]
        if args.style != "all":
            vars_to_record = [v for v in vars_to_record if v["style"] == args.style]

        print("\n" + "#" * 80)
        print(f"📌 RUANGAN: {room} | EFEK WEB: {feature}")
        print(f"   Nomor [{idx + 1}/31] • Intent {intent_id:02d} • Slot: ({act}, {obj}, {loc})")
        print("#" * 80)

        for v_idx, var in enumerate(vars_to_record):
            v_style = var["style"]
            v_label = var["label"]
            v_text = var["text"]

            filename = f"{speaker_id}_{cond_code}_{intent_id:02d}_{v_style}.wav"
            file_path = os.path.join(args.out_dir, filename)

            while True:
                print(f"\n👉 [TAKE {v_idx + 1}/{len(vars_to_record)} - {v_label.upper()}]")
                print(f"   Kalimat yang diucapkan: \"{v_text}\"")
                cmd = input("   Tekan [ENTER] untuk mulai rekam (atau 's': lewati / 'q': keluar): ").strip().lower()

                if cmd == "q":
                    print("\nPerekaman dihentikan oleh pengguna.")
                    _save_metadata(metadata_records, args.metadata_csv)
                    return
                if cmd == "s":
                    print(f"   Melewati variasi {v_label}.")
                    break

                print(f"   🎙️  [MEREKAM {args.duration}s] ... Silakan ucapkan: \"{v_text}\"")
                audio = record_clip(duration_s=args.duration, sr=16000)
                print("   ✅ Rekaman selesai!")

                # Quick audio check
                peak = float(np.max(np.abs(audio)))
                rms = float(np.sqrt(np.mean(audio**2)))
                rms_db = 20 * np.log10(rms + 1e-9)

                print(f"      📊 Peak: {peak:.3f} | RMS: {rms_db:.1f} dBFS")
                if peak < 0.05:
                    print("      ⚠️ Suara agak pelan! Lebih dekat ke mic ya.")
                elif peak > 0.98:
                    print("      ⚠️ Audio hampir clipping! Jauhkan mic sedikit.")

                # Review option
                review = input("   Simpan rekaman ini? [ENTER: Ya, 'r': Ulangi, 's': Lewati]: ").strip().lower()
                if review == "r":
                    print("   Mengulang take ini...")
                    continue
                if review == "s":
                    print("   Dilewati.")
                    break

                # Peak guard
                if peak > 0.95:
                    audio = audio * (0.95 / peak)

                sf.write(file_path, audio, samplerate=16000, subtype="PCM_16")
                print(f"   💾 Tersimpan: {file_path}")

                metadata_records.append({
                    "path": os.path.relpath(file_path, start=".").replace("\\", "/"),
                    "speakerId": speaker_id,
                    "device": args.device,
                    "condition": args.condition,
                    "style": v_style,
                    "transcript": v_text,
                    "action": act,
                    "object": obj,
                    "location": loc,
                    "intent_id": intent_id,
                    "room": room,
                    "feature": feature,
                })
                break

    _save_metadata(metadata_records, args.metadata_csv)


def _save_metadata(records, csv_path):
    if records:
        df_new = pd.DataFrame(records)
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["path"], keep="last")
        else:
            df_combined = df_new

        df_combined.to_csv(csv_path, index=False)
        print("\n" + "=" * 80)
        print(f"🎉 SUKSES! {len(records)} rekaman telah disimpan dan didaftarkan ke:")
        print(f"   {csv_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
