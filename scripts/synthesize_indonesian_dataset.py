"""Synthesize Indonesian Smart Home Speech Dataset from Corpus.

Generates 16 kHz Mono WAV audio from data/corpus/indonesian_smart_home_corpus_31_intents.csv
using Microsoft Edge Neural Indonesian voices:
- id-ID-ArdiNeural (Male speaker)
- id-ID-GadisNeural (Female speaker)

Follows Rule 7 of AGENTS.md (16 kHz mono WAV, standardized audio metadata).
"""

import os
import sys
import csv
import asyncio
import io
import tempfile
import soundfile as sf
import numpy as np
from scipy import signal
import edge_tts

# Target configuration matching SVARA audio preprocessing
TARGET_SR = 16000
VOICES = [
    {"id": "id-ID-ArdiNeural", "gender": "male", "speaker_prefix": "tts_ardi"},
    {"id": "id-ID-GadisNeural", "gender": "female", "speaker_prefix": "tts_gadis"}
]

def resample_and_mono(audio_bytes: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """Reads audio bytes, converts to mono, and resamples to target_sr."""
    with io.BytesIO(audio_bytes) as bio:
        data, sr = sf.read(bio, dtype="float32")

    # Mono conversion
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to 16 kHz if necessary
    if sr != target_sr:
        num_target_samples = int(round(len(data) * float(target_sr) / sr))
        data = signal.resample(data, num_target_samples).astype(np.float32)

    return data

async def generate_single_audio(text: str, voice: str) -> bytes:
    """Uses edge_tts to synthesize speech text into raw mp3 audio bytes."""
    communicate = edge_tts.Communicate(text, voice)
    mp3_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
    return mp3_buffer.getvalue()

async def run_batch_synthesis(limit: int = None):
    corpus_csv = os.path.join("data", "corpus", "indonesian_smart_home_corpus_31_intents.csv")
    if not os.path.exists(corpus_csv):
        print(f"Error: {corpus_csv} not found.")
        return

    output_audio_dir = os.path.join("data", "synthetic_indonesian", "wavs_16k")
    os.makedirs(output_audio_dir, exist_ok=True)
    metadata_csv = os.path.join("data", "synthetic_indonesian", "metadata.csv")

    rows = []
    with open(corpus_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if limit:
        rows = rows[:limit]

    print(f"Mulai sintesis audio untuk {len(rows)} kalimat korpus...")
    print(f"Format target: 16 kHz WAV mono (Standar wav2vec2 SVARA)")
    print(f"Suara: id-ID-ArdiNeural (Pria) dan id-ID-GadisNeural (Wanita)")

    metadata_records = []
    total_tasks = len(rows) * len(VOICES)
    done_count = 0

    for row_idx, row in enumerate(rows):
        intent_id = int(row["intent_id"])
        action = row["action"]
        obj = row["object"]
        loc = row["location"]
        phrase = row["indonesian_phrase"]

        for v_info in VOICES:
            voice_id = v_info["id"]
            gender = v_info["gender"]
            speaker = v_info["speaker_prefix"]

            file_id = f"id_intent{intent_id:02d}_{row_idx:03d}_{speaker}"
            filename = f"{file_id}.wav"
            rel_wav_path = os.path.join("wavs_16k", filename).replace("\\", "/")
            abs_wav_path = os.path.join(output_audio_dir, filename)

            # Generate if not exists
            if not os.path.exists(abs_wav_path):
                try:
                    mp3_data = await generate_single_audio(phrase, voice_id)
                    wav_data = resample_and_mono(mp3_data, TARGET_SR)
                    sf.write(abs_wav_path, wav_data, TARGET_SR, subtype="PCM_16")
                except Exception as e:
                    print(f"Gagal generate {file_id}: {e}")
                    continue

            # Record metadata
            metadata_records.append({
                "path": rel_wav_path,
                "speakerId": speaker,
                "gender": gender,
                "transcript": phrase,
                "action": action,
                "object": obj,
                "location": loc,
                "intent_id": intent_id
            })

            done_count += 1
            if done_count % 50 == 0 or done_count == total_tasks:
                print(f"Progres: {done_count}/{total_tasks} audio WAV selesai dibuat.")

    # Save metadata CSV
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "speakerId", "gender", "transcript", "action", "object", "location", "intent_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_records)

    print(f"\nAlhamdulillah! Selesai membuat {len(metadata_records)} audio dataset suara sintetis.")
    print(f"Audio tersimpan di: {output_audio_dir}")
    print(f"Metadata tersimpan di: {metadata_csv}")

if __name__ == "__main__":
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(run_batch_synthesis(limit=limit_arg))
