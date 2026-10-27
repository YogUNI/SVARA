"""High-Performance Multi-Worker Neural TTS Synthesis Engine for Massive Indonesian Dataset.

Task: Sub-step 3.1 (Phase 3: Multi-Speaker Voice Synthesis)
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md, docs/01_DATASET_AND_EVALUATION.md

Architecture:
- Asynchronous producer-consumer worker queue with asyncio.Semaphore(15)
- Microsoft Edge Neural TTS voices:
    1. id-ID-ArdiNeural (Male Indonesian speaker, 'spk_ardi')
    2. id-ID-GadisNeural (Female Indonesian speaker, 'spk_gadis')
- Automatic exponential backoff retry mechanism (network resilience)
- Stateful checkpointing & automatic resumption (skips already synthesized audio)
- Real-time in-memory audio conversion & signal hygiene:
    - Decodes MP3 stream via soundfile
    - Stereo-to-mono downmix
    - Polyphase sinc resampling to strictly 16,000 Hz (scipy.signal.resample_poly)
    - Headroom peak normalization (0.95 peak limit)
    - Saves as 16-bit PCM Mono WAV (PCM_16)
- Thread-safe metadata manifest export (data/synthetic_indonesian_massive/metadata.csv)
"""

import os
import sys
import csv
import io
import time
import math
import random
import asyncio
import argparse
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import scipy.signal as signal
import soundfile as sf
import edge_tts

DEFAULT_CORPUS_CSV = "data/corpus/indonesian_smart_home_corpus_4k.csv"
DEFAULT_OUTPUT_DIR = "data/synthetic_indonesian_massive"
DEFAULT_CONCURRENCY = 15
TARGET_SAMPLE_RATE = 16000

DEFAULT_VOICES = [
    {
        "id": "id-ID-ArdiNeural",
        "gender": "male",
        "speaker_id": "spk_ardi",
        "display_name": "Ardi (Pria)",
    },
    {
        "id": "id-ID-GadisNeural",
        "gender": "female",
        "speaker_id": "spk_gadis",
        "display_name": "Gadis (Wanita)",
    },
]


def convert_mp3_to_16k_mono_wav(
    mp3_bytes: bytes,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, float]:
    """Decodes in-memory MP3 bytes, converts to mono, resamples to target_sr, and normalizes peak."""
    with io.BytesIO(mp3_bytes) as bio:
        audio, sr = sf.read(bio, dtype="float32")

    # 1. Downmix stereo to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 2. High-quality polyphase sinc resampling
    if sr != target_sr:
        gcd = math.gcd(int(target_sr), int(sr))
        up = int(target_sr // gcd)
        down = int(sr // gcd)
        audio = signal.resample_poly(audio, up, down).astype(np.float32)

    # 3. Peak normalization with 0.5 dB headroom (peak <= 0.95)
    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if peak > 0.95:
        audio = (audio / peak) * 0.95
    elif peak > 0.0 and peak < 0.30:
        # Boost extremely quiet clips
        audio = (audio / peak) * 0.85

    duration = float(len(audio) / target_sr)
    return audio, duration


def verify_wav_file(wav_path: str, expected_sr: int = TARGET_SAMPLE_RATE) -> bool:
    """Verifies that an existing WAV file is valid, non-corrupted, mono, and expected sample rate."""
    if not os.path.exists(wav_path):
        return False
    if os.path.getsize(wav_path) < 1000:
        return False
    try:
        info = sf.info(wav_path)
        if info.samplerate != expected_sr or info.channels != 1 or info.subtype != "PCM_16":
            return False
        return info.duration > 0.2
    except Exception:
        return False


class MassiveTTSAsyncEngine:
    """Production asynchronous multi-worker synthesis engine."""

    def __init__(
        self,
        corpus_csv_path: str = DEFAULT_CORPUS_CSV,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        concurrency: int = DEFAULT_CONCURRENCY,
        voices: Optional[List[Dict[str, str]]] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ):
        self.corpus_csv_path = corpus_csv_path
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.voices = voices or DEFAULT_VOICES
        self.rate = rate
        self.pitch = pitch

        self.wav_dir = os.path.join(self.output_dir, "wavs_16k")
        self.metadata_csv_path = os.path.join(self.output_dir, "metadata.csv")
        os.makedirs(self.wav_dir, exist_ok=True)

        self.completed_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        self.lock: Optional[asyncio.Lock] = None
        self.manifest_records: List[Dict[str, Any]] = []

    async def synthesize_single(
        self,
        phrase: str,
        voice_id: str,
        dest_wav_path: str,
        max_retries: int = 4,
    ) -> Tuple[bool, float]:
        """Synthesizes one phrase to a 16kHz mono WAV file with exponential backoff retry."""
        # 1. Check existing valid WAV file (resumability)
        if verify_wav_file(dest_wav_path):
            info = sf.info(dest_wav_path)
            return True, info.duration

        # 2. Attempt Edge TTS communication with backoff
        for attempt in range(max_retries):
            try:
                communicate = edge_tts.Communicate(
                    text=phrase,
                    voice=voice_id,
                    rate=self.rate,
                    pitch=self.pitch,
                )
                mp3_buf = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_buf.write(chunk["data"])

                mp3_data = mp3_buf.getvalue()
                if len(mp3_data) < 200:
                    raise ValueError(f"Received empty audio buffer (size: {len(mp3_data)} bytes)")

                # Convert to clean 16 kHz 16-bit Mono WAV
                wav_data, duration = convert_mp3_to_16k_mono_wav(mp3_data, TARGET_SAMPLE_RATE)
                sf.write(dest_wav_path, wav_data, TARGET_SAMPLE_RATE, subtype="PCM_16")

                # Verify written file
                if not verify_wav_file(dest_wav_path):
                    raise IOError("Written WAV failed post-write verification.")

                return True, duration

            except Exception as e:
                backoff_delay = 1.0 * (2 ** attempt) + random.uniform(0.1, 0.5)
                if attempt == max_retries - 1:
                    print(f"\n[!] ERROR ({voice_id}): '{phrase[:30]}...' -> {e}")
                    return False, 0.0
                await asyncio.sleep(backoff_delay)

        return False, 0.0

    async def _worker(
        self,
        worker_id: int,
        queue: asyncio.Queue,
        total_items: int,
        dry_run: bool = False,
    ):
        """Worker task consuming items from queue."""
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            try:
                idx, total, row, voice_info = item
                voice_id = voice_info["id"]
                spk_id = voice_info["speaker_id"]
                gender = voice_info["gender"]
                intent_id = int(row["intent_id"])
                phrase = str(row["indonesian_phrase"]).strip()

                filename = f"id_intent{intent_id:02d}_{idx:04d}_{spk_id}.wav"
                dest_wav_path = os.path.join(self.wav_dir, filename)
                rel_path = os.path.join("wavs_16k", filename).replace("\\", "/")

                success = False
                duration = 0.0

                if dry_run:
                    success = True
                    duration = 2.5
                else:
                    success, duration = await self.synthesize_single(phrase, voice_id, dest_wav_path)

                async with self.lock:
                    if success:
                        self.completed_count += 1
                        self.manifest_records.append({
                            "path": rel_path,
                            "speakerId": spk_id,
                            "gender": gender,
                            "voice_name": voice_id,
                            "transcript": phrase,
                            "action": row["action"],
                            "object": row["object"],
                            "location": row["location"],
                            "domain": row.get("domain", "General"),
                            "intent_id": intent_id,
                            "duration": round(duration, 3),
                        })
                    else:
                        self.failed_count += 1

                    total_processed = self.completed_count + self.failed_count
                    if total_processed % 50 == 0 or total_processed == total_items:
                        pct = (total_processed / total_items) * 100
                        print(
                            f"\r[*] Progress: {total_processed:,}/{total_items:,} ({pct:5.1f}%) | "
                            f"Success: {self.completed_count:,} | Failed: {self.failed_count}",
                            end="",
                            flush=True,
                        )
            except Exception as e:
                print(f"\n[Worker {worker_id} Error]: {e}", flush=True)
                async with self.lock:
                    self.failed_count += 1
            finally:
                queue.task_done()

    async def run(
        self,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Runs the multi-worker synthesis pipeline."""
        if not os.path.exists(self.corpus_csv_path):
            raise FileNotFoundError(f"Corpus CSV not found at: {self.corpus_csv_path}")

        rows = []
        with open(self.corpus_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        if limit is not None and limit > 0:
            rows = rows[:limit]

        total_phrases = len(rows)
        total_audio_targets = total_phrases * len(self.voices)

        print("\n" + "=" * 75, flush=True)
        print("   SVARA: ENGINE SINTESIS SUARA MULTI-SPEAKER MASSAL (NEURAL TTS)", flush=True)
        print("=" * 75, flush=True)
        print(f"Total Frasa Korpus             : {total_phrases:,} frasa", flush=True)
        print(f"Jumlah Pembicara (Voices)      : {len(self.voices)} ({', '.join([v['display_name'] for v in self.voices])})", flush=True)
        print(f"Total Target Audio WAV         : {total_audio_targets:,} file (16 kHz 16-bit Mono)", flush=True)
        print(f"Tingkat Konkurensi (Workers)   : {self.concurrency} parallel async workers", flush=True)
        print(f"Direktori Output Audio         : {self.wav_dir}", flush=True)
        print(f"Mode Eksekusi                  : {'DRY RUN (Simulasi)' if dry_run else 'LIVE TTS DOWNLOAD'}", flush=True)
        print("=" * 75 + "\n", flush=True)

        self.lock = asyncio.Lock()
        start_time = time.time()
        queue: asyncio.Queue = asyncio.Queue()

        # Enqueue all jobs
        task_idx = 0
        for row in rows:
            for v_info in self.voices:
                queue.put_nowait((task_idx, total_audio_targets, row, v_info))
                task_idx += 1

        # Spawn workers (adjust to not exceed total items)
        actual_concurrency = min(self.concurrency, max(1, total_audio_targets))
        workers = []
        for w_id in range(actual_concurrency):
            w = asyncio.create_task(
                self._worker(w_id, queue, total_audio_targets, dry_run=dry_run)
            )
            workers.append(w)

        # Wait until queue is completely processed
        await queue.join()

        # Stop workers
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)

        elapsed = time.time() - start_time

        # Save metadata manifest
        fieldnames = [
            "path", "speakerId", "gender", "voice_name", "transcript",
            "action", "object", "location", "domain", "intent_id", "duration"
        ]
        with open(self.metadata_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.manifest_records)

        durations = [r["duration"] for r in self.manifest_records]
        total_duration_hours = sum(durations) / 3600.0 if durations else 0.0

        print("\n\n" + "=" * 75)
        print("      RINGKASAN EKSEKUSI SINTESIS SUARA INDONESIA MASSAL")
        print("=" * 75)
        print(f"Total Audio Berhasil Disimpan  : {len(self.manifest_records):,} / {total_audio_targets:,} file")
        print(f"Total Audio Gagal              : {self.failed_count}")
        print(f"Waktu Eksekusi Total           : {elapsed:.2f} detik ({elapsed / 60:.2f} menit)")
        if durations:
            print(f"Total Durasi Audio             : {total_duration_hours:.2f} jam ({sum(durations):.1f} detik)")
            print(f"Rata-rata Durasi per Klip      : {np.mean(durations):.2f} detik (Min: {min(durations):.2f}s, Max: {max(durations):.2f}s)")
        print(f"Lokasi File Manifest Metadata  : {self.metadata_csv_path}")
        print("=" * 75 + "\n")

        return {
            "total_targets": total_audio_targets,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "elapsed_seconds": round(elapsed, 2),
            "total_duration_hours": round(total_duration_hours, 3),
            "manifest_path": self.metadata_csv_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Massive Indonesian Neural TTS Synthesizer")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS_CSV, help="Path to corpus CSV file")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Number of concurrent async workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of phrases to process (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without network calls")
    args = parser.parse_args()

    engine = MassiveTTSAsyncEngine(
        corpus_csv_path=args.corpus,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
    )
    asyncio.run(engine.run(limit=args.limit, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
