"""Interactive CLI Voice Recorder for SVARA Own Recordings Dataset.

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

PHRASES = [
    {"intent_id": 0, "phrase": "Turn on the lamp", "action": "activate", "object": "lamp", "location": "none"},
    {"intent_id": 1, "phrase": "Turn on the lights in the bedroom", "action": "activate", "object": "lights", "location": "bedroom"},
    {"intent_id": 2, "phrase": "Turn on the kitchen lights", "action": "activate", "object": "lights", "location": "kitchen"},
    {"intent_id": 3, "phrase": "Turn the lights on", "action": "activate", "object": "lights", "location": "none"},
    {"intent_id": 4, "phrase": "Turn on the washroom lights", "action": "activate", "object": "lights", "location": "washroom"},
    {"intent_id": 5, "phrase": "Resume", "action": "activate", "object": "music", "location": "none"},
    {"intent_id": 6, "phrase": "Bring some juice", "action": "bring", "object": "juice", "location": "none"},
    {"intent_id": 7, "phrase": "Get me the newspaper", "action": "bring", "object": "newspaper", "location": "none"},
    {"intent_id": 8, "phrase": "Bring me my shoes", "action": "bring", "object": "shoes", "location": "none"},
    {"intent_id": 9, "phrase": "Bring socks", "action": "bring", "object": "socks", "location": "none"},
    {"intent_id": 10, "phrase": "Set language to Chinese", "action": "change language", "object": "Chinese", "location": "none"},
    {"intent_id": 11, "phrase": "Set my phone's language to English", "action": "change language", "object": "English", "location": "none"},
    {"intent_id": 12, "phrase": "I need to practice my German. Switch the language", "action": "change language", "object": "German", "location": "none"},
    {"intent_id": 13, "phrase": "Set language to Korean", "action": "change language", "object": "Korean", "location": "none"},
    {"intent_id": 14, "phrase": "Change language", "action": "change language", "object": "none", "location": "none"},
    {"intent_id": 15, "phrase": "Turn off the lamp", "action": "deactivate", "object": "lamp", "location": "none"},
    {"intent_id": 16, "phrase": "Turn the bedroom lights off", "action": "deactivate", "object": "lights", "location": "bedroom"},
    {"intent_id": 17, "phrase": "Turn the lights off in the kitchen", "action": "deactivate", "object": "lights", "location": "kitchen"},
    {"intent_id": 18, "phrase": "Switch off the lights", "action": "deactivate", "object": "lights", "location": "none"},
    {"intent_id": 19, "phrase": "Switch off the washroom lights", "action": "deactivate", "object": "lights", "location": "washroom"},
    {"intent_id": 20, "phrase": "Turn off the music", "action": "deactivate", "object": "music", "location": "none"},
    {"intent_id": 21, "phrase": "Turn down the bedroom heat", "action": "decrease", "object": "heat", "location": "bedroom"},
    {"intent_id": 22, "phrase": "Turn the kitchen temperature down", "action": "decrease", "object": "heat", "location": "kitchen"},
    {"intent_id": 23, "phrase": "Turn down the temperature", "action": "decrease", "object": "heat", "location": "none"},
    {"intent_id": 24, "phrase": "Turn the temperature down in the washroom", "action": "decrease", "object": "heat", "location": "washroom"},
    {"intent_id": 25, "phrase": "Turn the volume down", "action": "decrease", "object": "volume", "location": "none"},
    {"intent_id": 26, "phrase": "Turn up the temperature in the bedroom", "action": "increase", "object": "heat", "location": "bedroom"},
    {"intent_id": 27, "phrase": "Turn the kitchen temperature up", "action": "increase", "object": "heat", "location": "kitchen"},
    {"intent_id": 28, "phrase": "Turn up the temperature", "action": "increase", "object": "heat", "location": "none"},
    {"intent_id": 29, "phrase": "Turn up the washroom temperature", "action": "increase", "object": "heat", "location": "washroom"},
    {"intent_id": 30, "phrase": "Volume up", "action": "increase", "object": "volume", "location": "none"},
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
    parser = argparse.ArgumentParser(description="Interactive Voice Recorder for SVARA Own Recordings")
    parser.add_argument("--speaker", type=str, default="haikal", help="Speaker ID (e.g. haikal, yoga, s01)")
    parser.add_argument("--condition", type=str, choices=["quiet", "noisy"], default="quiet", help="Recording condition (quiet or noisy)")
    parser.add_argument("--device", type=str, default="laptop_mic", help="Recording device (laptop_mic, phone, headset)")
    parser.add_argument("--gender", type=str, default="male", help="Speaker gender (male/female)")
    parser.add_argument("--age-range", type=str, default="20-25", help="Speaker age range")
    parser.add_argument("--duration", type=float, default=3.5, help="Recording duration per clip in seconds")
    parser.add_argument("--start-index", type=int, default=0, help="Starting intent index (0-30)")
    parser.add_argument("--out-dir", type=str, default="data/own_recordings/wavs", help="Output directory for wav files")
    parser.add_argument("--metadata-csv", type=str, default="data/own_recordings/metadata.csv", help="Metadata CSV path")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metadata_csv), exist_ok=True)

    cond_code = "q" if args.condition == "quiet" else "n"
    speaker_id = f"own_{args.speaker.lower()}"

    print("=" * 65)
    print("SVARA INTERACTIVE VOICE RECORDER (SUB-STEP 13.2)")
    print("=" * 65)
    print(f"Pembicara : {args.speaker} ({speaker_id})")
    print(f"Kondisi    : {args.condition.upper()} (Kode: {cond_code})")
    print(f"Device     : {args.device}")
    print(f"Durasi     : {args.duration} detik per rekaman (16 kHz Mono)")
    print(f"Target Dir : {args.out_dir}")
    print("=" * 65)
    print("Petunjuk:")
    print("1. Baca kalimat di layar dengan suara santai, jelas, dan natural.")
    print("2. Tekan ENTER untuk mulai merekam, lalu langsung ucapkan kalimatnya.")
    print("3. Ketik 'r' lalu ENTER jika ingin mengulang (retake) rekaman terakhir.")
    print("4. Ketik 'q' lalu ENTER untuk keluar kapan saja.")
    print("=" * 65)

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

        filename = f"{speaker_id}_{cond_code}_{intent_id:02d}.wav"
        file_path = os.path.join(args.out_dir, filename)

        while True:
            print(f"\n[{idx + 1}/31] Intent {intent_id:02d}: {act} | {obj} | {loc}")
            print(f"Kalimat yang dibaca: \"{phrase}\"")
            user_cmd = input("Tekan ENTER untuk mulai rekam (atau ketik 's' skip / 'q' keluar): ").strip().lower()

            if user_cmd == "q":
                print("Perekaman dihentikan oleh pengguna.")
                return
            if user_cmd == "s":
                print(f"Melewati intent {intent_id:02d}.")
                break

            print(f"🎙️  MEREKAM selama {args.duration}s... Bicara sekarang!")
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

            # Replace existing or append
            metadata_rows = [r for r in metadata_rows if r.get("path") != meta_entry["path"]]
            metadata_rows.append(meta_entry)

            # Save CSV
            df_out = pd.DataFrame(metadata_rows)
            df_out.to_csv(args.metadata_csv, index=False)
            break

    print("\n" + "=" * 65)
    print(f"SELURUH SESI PEREKAMAN UNTUK {args.speaker.upper()} SELESAI!")
    print(f"Metadata tersimpan di: {args.metadata_csv}")
    print("=" * 65)


if __name__ == "__main__":
    main()
