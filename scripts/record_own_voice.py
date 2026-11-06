"""Interactive CLI Voice Recorder for SVARA Own Recordings Dataset (Bahasa Indonesia).

Task: Sub-step 13.2 / P5-02
Reference: docs/01 §8, docs/10 P5-02, docs/14 FASE 13
Usage:
    python scripts/record_own_voice.py --speaker haikal --condition quiet --device laptop_mic
    python scripts/record_own_voice.py --speaker yoga --condition quiet --start-index 0
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

# 31 Perintah Smart Home Bahasa Indonesia Alami & Sehari-hari
PHRASES = [
    {"intent_id": 0, "phrase": "Nyalakan lampu meja", "action": "activate", "object": "lamp", "location": "none"},
    {"intent_id": 1, "phrase": "Nyalakan lampu kamar tidur", "action": "activate", "object": "lights", "location": "bedroom"},
    {"intent_id": 2, "phrase": "Nyalakan lampu dapur", "action": "activate", "object": "lights", "location": "kitchen"},
    {"intent_id": 3, "phrase": "Nyalakan semua lampu", "action": "activate", "object": "lights", "location": "none"},
    {"intent_id": 4, "phrase": "Nyalakan lampu kamar mandi", "action": "activate", "object": "lights", "location": "washroom"},
    {"intent_id": 5, "phrase": "Putar lagi lagunya", "action": "activate", "object": "music", "location": "none"},
    {"intent_id": 6, "phrase": "Bawakan saya jus", "action": "bring", "object": "juice", "location": "none"},
    {"intent_id": 7, "phrase": "Ambilkan koran untuk saya", "action": "bring", "object": "newspaper", "location": "none"},
    {"intent_id": 8, "phrase": "Ambilkan sepatu saya", "action": "bring", "object": "shoes", "location": "none"},
    {"intent_id": 9, "phrase": "Ambilkan kaus kaki", "action": "bring", "object": "socks", "location": "none"},
    {"intent_id": 10, "phrase": "Ganti ke bahasa Mandarin dong", "action": "change language", "object": "Chinese", "location": "none"},
    {"intent_id": 11, "phrase": "Ganti bahasanya ke Bahasa Inggris", "action": "change language", "object": "English", "location": "none"},
    {"intent_id": 12, "phrase": "Ganti bahasanya ke Bahasa Jerman", "action": "change language", "object": "German", "location": "none"},
    {"intent_id": 13, "phrase": "Ganti ke bahasa Korea dong", "action": "change language", "object": "Korean", "location": "none"},
    {"intent_id": 14, "phrase": "Ganti bahasa sistem", "action": "change language", "object": "none", "location": "none"},
    {"intent_id": 15, "phrase": "Matikan lampu meja", "action": "deactivate", "object": "lamp", "location": "none"},
    {"intent_id": 16, "phrase": "Matikan lampu kamar", "action": "deactivate", "object": "lights", "location": "bedroom"},
    {"intent_id": 17, "phrase": "Matikan lampu dapur", "action": "deactivate", "object": "lights", "location": "kitchen"},
    {"intent_id": 18, "phrase": "Matikan semua lampu", "action": "deactivate", "object": "lights", "location": "none"},
    {"intent_id": 19, "phrase": "Matikan lampu kamar mandi", "action": "deactivate", "object": "lights", "location": "washroom"},
    {"intent_id": 20, "phrase": "Matikan musiknya", "action": "deactivate", "object": "music", "location": "none"},
    {"intent_id": 21, "phrase": "Dinginkan kamar tidur dong", "action": "decrease", "object": "heat", "location": "bedroom"},
    {"intent_id": 22, "phrase": "Dinginkan dapur dong", "action": "decrease", "object": "heat", "location": "kitchen"},
    {"intent_id": 23, "phrase": "Turunkan suhunya dong", "action": "decrease", "object": "heat", "location": "none"},
    {"intent_id": 24, "phrase": "Dinginkan kamar mandi dong", "action": "decrease", "object": "heat", "location": "washroom"},
    {"intent_id": 25, "phrase": "Kecilkan volumenya dong", "action": "decrease", "object": "volume", "location": "none"},
    {"intent_id": 26, "phrase": "Naikkan suhu kamar tidur dong", "action": "increase", "object": "heat", "location": "bedroom"},
    {"intent_id": 27, "phrase": "Naikkan suhu dapur dong", "action": "increase", "object": "heat", "location": "kitchen"},
    {"intent_id": 28, "phrase": "Naikkan suhunya dong", "action": "increase", "object": "heat", "location": "none"},
    {"intent_id": 29, "phrase": "Naikkan suhu kamar mandi dong", "action": "increase", "object": "heat", "location": "washroom"},
    {"intent_id": 30, "phrase": "Besarkan suaranya dong", "action": "increase", "object": "volume", "location": "none"},
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
    parser.add_argument("--take", type=int, default=1, help="Nomor pengambilan / variasi rekaman (contoh: 1 untuk nada biasa, 2 untuk intonasi santai/tanya)")
    parser.add_argument("--start-index", type=int, default=0, help="Mulai dari intent ke-n (0-30)")
    parser.add_argument("--out-dir", type=str, default="data/own_recordings/wavs", help="Folder output WAV")
    parser.add_argument("--metadata-csv", type=str, default="data/own_recordings/metadata.csv", help="Lokasi metadata CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_csv), exist_ok=True)

    cond_code = "q" if args.condition == "quiet" else "n"
    speaker_id = f"own_{args.speaker.lower()}"

    print("=" * 70)
    print("SVARA PEREKAM SUARA ASLI BAHASA INDONESIA (SUB-STEP 13.2)")
    print("=" * 70)
    print(f"Pembicara : {args.speaker.upper()} ({speaker_id})")
    print(f"Kondisi   : {args.condition.upper()} (Kode: {cond_code})")
    print(f"Device    : {args.device}")
    print(f"Durasi    : {args.duration} detik per rekaman (16 kHz Mono PCM_16)")
    print(f"Target Dir: {args.out_dir}")
    print("=" * 70)
    print("Petunjuk:")
    print("1. Baca kalimat Bahasa Indonesia di layar dengan suara santai, jelas, dan natural.")
    print("2. Tekan ENTER untuk mulai merekam, lalu langsung ucapkan kalimatnya.")
    print("3. Ketik 'r' lalu ENTER jika ingin mengulang (retake) rekaman terakhir.")
    print("4. Ketik 'q' lalu ENTER untuk keluar kapan saja.")
    print("=" * 70)

    metadata_rows = []
    if os.path.exists(args.metadata_csv):
        try:
            existing_df = pd.read_csv(args.metadata_csv)
            metadata_rows = existing_df.to_dict("records")
        except Exception:
            metadata_rows = []

    for idx in range(args.start_index, len(PHRASES)):
        item = PHRASES[idx]
        intent_id = item["intent_id"]
        phrase = item["phrase"]
        act = item["action"]
        obj = item["object"]
        loc = item["location"]

        take_suffix = f"_t{args.take}" if args.take > 1 else ""
        filename = f"{speaker_id}_{cond_code}_{intent_id:02d}{take_suffix}.wav"
        file_path = os.path.join(args.out_dir, filename)

        while True:
            print(f"\n[{idx + 1}/31] Intent {intent_id:02d}: {act} | {obj} | {loc}")
            print(f"👉 Kalimat yang dibaca: \"{phrase}\"")
            user_cmd = input("Tekan ENTER untuk mulai rekam (atau 's': lewati / 'q': keluar): ").strip().lower()

            if user_cmd == "q":
                print("Perekaman dihentikan oleh pengguna.")
                return
            if user_cmd == "s":
                print(f"Melewati intent {intent_id:02d}.")
                break

            print(f"🎙️  MEREKAM selama {args.duration}s... Ucapkan: \"{phrase}\"")
            audio = record_clip(duration_s=args.duration, sr=16000)
            print("✅ Rekaman selesai!")

            # Quick audio check
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio**2)))
            rms_db = 20 * np.log10(rms + 1e-9)

            print(f"   Peak Level: {peak:.3f} | RMS: {rms_db:.1f} dBFS")
            if peak < 0.05:
                print("   ⚠️ Peringatan: Suara terlalu pelan! Dekatkan mikrofon.")
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
            print(f"💾 File tersimpan: {file_path}")

            # Update metadata
            meta_entry = {
                "path": f"data/own_recordings/wavs/{filename}".replace("\\", "/"),
                "speakerId": speaker_id,
                "device": args.device,
                "condition": args.condition,
                "transcript": phrase,
                "action": act,
                "object": obj,
                "location": loc,
                "intent_id": intent_id,
                "gender": args.gender,
                "age_range": args.age_range,
                "duration_seconds": round(len(audio) / 16000, 2),
                "peak_amplitude": round(float(np.max(np.abs(audio))), 4),
            }

            metadata_rows = [r for r in metadata_rows if r.get("path") != meta_entry["path"]]
            metadata_rows.append(meta_entry)

            # Save CSV
            df_out = pd.DataFrame(metadata_rows)
            df_out.to_csv(args.metadata_csv, index=False)
            break

    print("\n" + "=" * 70)
    print(f"SELURUH SESI PEREKAMAN UNTUK {args.speaker.upper()} SELESAI!")
    print(f"Metadata tersimpan di: {args.metadata_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
