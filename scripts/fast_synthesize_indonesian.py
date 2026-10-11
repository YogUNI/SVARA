"""Fast Concurrent Synthesizer for Expanded Indonesian Smart Home Dataset.

Generates 16 kHz Mono WAV audio from data/corpus/indonesian_smart_home_corpus_31_intents.csv
using Microsoft Edge Neural Indonesian voices:
- id-ID-ArdiNeural (Male speaker)
- id-ID-GadisNeural (Female speaker)

Uses asyncio.Semaphore for high concurrency (e.g. 15-20 parallel downloads).
"""

import os
import sys
import csv
import asyncio
import io
import soundfile as sf
import numpy as np
from scipy import signal
import edge_tts
from tqdm.asyncio import tqdm

TARGET_SR = 16000
VOICES = [
    {"id": "id-ID-ArdiNeural", "gender": "male", "speaker_prefix": "tts_ardi"},
    {"id": "id-ID-GadisNeural", "gender": "female", "speaker_prefix": "tts_gadis"}
]

def resample_and_mono(audio_bytes: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    with io.BytesIO(audio_bytes) as bio:
        data, sr = sf.read(bio, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        num_target_samples = int(round(len(data) * float(target_sr) / sr))
        data = signal.resample(data, num_target_samples).astype(np.float32)
    return data

async def synth_task(sem, phrase, voice_id, abs_path, retries=3):
    if os.path.exists(abs_path) and os.path.getsize(abs_path) > 1000:
        return True
    async with sem:
        for attempt in range(retries):
            try:
                communicate = edge_tts.Communicate(phrase, voice_id)
                mp3_buf = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_buf.write(chunk["data"])
                mp3_data = mp3_buf.getvalue()
                if len(mp3_data) == 0:
                    raise ValueError("Empty audio received")
                wav_data = resample_and_mono(mp3_data, TARGET_SR)
                sf.write(abs_path, wav_data, TARGET_SR, subtype="PCM_16")
                return True
            except Exception as e:
                if attempt == retries - 1:
                    print(f"\n[Error] {abs_path}: {e}")
                    return False
                await asyncio.sleep(0.5 * (attempt + 1))

async def main():
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

    print(f"=== SINTESIS CEPAT DATASET INDONESIA (CONCURRENCY: 15) ===")
    print(f"Total frasa korpus: {len(rows)} frasa x 2 suara (Pria & Wanita)")
    print(f"Total target audio: {len(rows) * len(VOICES)} file WAV (16 kHz mono)")

    sem = asyncio.Semaphore(15)
    tasks = []
    metadata_records = []

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

            tasks.append(synth_task(sem, phrase, voice_id, abs_wav_path))
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

    print(f"Memulai download & konversi audio paralel...")
    await tqdm.gather(*tasks, desc="TTS Synthesis")

    # Filter out records whose wav failed or doesn't exist
    valid_records = []
    for r in metadata_records:
        full_p = os.path.join("data", "synthetic_indonesian", r["path"])
        if os.path.exists(full_p) and os.path.getsize(full_p) > 1000:
            valid_records.append(r)

    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "speakerId", "gender", "transcript", "action", "object", "location", "intent_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_records)

    print(f"\n[SELESAI] Total audio WAV berhasil disintesis: {len(valid_records)} / {len(metadata_records)}")
    print(f"Metadata berhasil diperbarui di: {metadata_csv}")

if __name__ == "__main__":
    asyncio.run(main())
