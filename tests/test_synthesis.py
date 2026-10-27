"""Unit tests for the Massive Indonesian Neural TTS Synthesis Engine.

Task: Sub-step 3.1 (Phase 3: Multi-Speaker Voice Synthesis)
Reference: docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md, AGENTS.md
"""

import os
import io
import tempfile
import asyncio
import pandas as pd
import numpy as np
import soundfile as sf
import pytest

from scripts.synthesize_massive_indonesian import (
    convert_mp3_to_16k_mono_wav,
    verify_wav_file,
    MassiveTTSAsyncEngine,
    DEFAULT_VOICES,
    TARGET_SAMPLE_RATE,
)


def test_audio_conversion_mono_resample_headroom():
    """Verify that in-memory audio conversion outputs 16 kHz mono with proper headroom."""
    sr_in = 24000
    duration_s = 1.0
    # Create stereo 24 kHz tone
    t = np.linspace(0, duration_s, int(sr_in * duration_s), endpoint=False)
    left = 0.8 * np.sin(2 * np.pi * 440 * t)
    right = 0.6 * np.sin(2 * np.pi * 440 * t)
    stereo_audio = np.stack([left, right], axis=1).astype(np.float32)

    # Encode to MP3 in memory
    mp3_buf = io.BytesIO()
    sf.write(mp3_buf, stereo_audio, sr_in, format="MP3")
    mp3_bytes = mp3_buf.getvalue()

    # Convert with engine function
    converted, out_duration = convert_mp3_to_16k_mono_wav(mp3_bytes, TARGET_SAMPLE_RATE)

    assert converted.ndim == 1, "Audio must be single-channel mono"
    assert len(converted) == int(round(duration_s * TARGET_SAMPLE_RATE)), "Audio length mismatch at 16 kHz"
    assert np.max(np.abs(converted)) <= 0.950001, "Audio peak must not exceed 0.95 headroom limit"
    assert abs(out_duration - duration_s) < 0.05, "Duration mismatch"


def test_verify_wav_file():
    """Verify that verify_wav_file accurately inspects sample rate, channel count, and file health."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Non-existent file
        assert not verify_wav_file(os.path.join(tmpdir, "nonexistent.wav"))

        # Valid 16 kHz mono 16-bit PCM WAV
        valid_wav = os.path.join(tmpdir, "valid.wav")
        audio = (0.5 * np.sin(np.linspace(0, 1, 16000))).astype(np.float32)
        sf.write(valid_wav, audio, 16000, subtype="PCM_16")
        assert verify_wav_file(valid_wav, expected_sr=16000)

        # Invalid sample rate (24 kHz)
        invalid_sr_wav = os.path.join(tmpdir, "invalid_sr.wav")
        sf.write(invalid_sr_wav, audio, 24000, subtype="PCM_16")
        assert not verify_wav_file(invalid_sr_wav, expected_sr=16000)

        # Corrupted / empty file (< 1000 bytes)
        corrupted_wav = os.path.join(tmpdir, "corrupt.wav")
        with open(corrupted_wav, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 20)
        assert not verify_wav_file(corrupted_wav)


def test_engine_initialization_and_voices():
    """Verify engine configuration and dual-voice Indonesian speaker mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MassiveTTSAsyncEngine(output_dir=tmpdir, concurrency=4)
        assert len(engine.voices) == 2
        voice_ids = [v["id"] for v in engine.voices]
        speaker_ids = [v["speaker_id"] for v in engine.voices]
        genders = [v["gender"] for v in engine.voices]

        assert "id-ID-ArdiNeural" in voice_ids
        assert "id-ID-GadisNeural" in voice_ids
        assert "spk_ardi" in speaker_ids
        assert "spk_gadis" in speaker_ids
        assert "male" in genders
        assert "female" in genders


@pytest.mark.asyncio
async def test_engine_dry_run_pipeline():
    """Verify end-to-end dry-run execution, worker queuing, and manifest creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MassiveTTSAsyncEngine(output_dir=tmpdir, concurrency=4)
        res = await engine.run(limit=5, dry_run=True)

        assert res["total_targets"] == 10  # 5 phrases * 2 voices
        assert res["completed"] == 10
        assert res["failed"] == 0

        manifest_path = os.path.join(tmpdir, "metadata.csv")
        assert os.path.exists(manifest_path)
        df = pd.read_csv(manifest_path)
        assert len(df) == 10
        expected_cols = [
            "path", "speakerId", "gender", "voice_name", "transcript",
            "action", "object", "location", "domain", "intent_id", "duration"
        ]
        for col in expected_cols:
            assert col in df.columns
            assert df[col].notna().all()


@pytest.mark.asyncio
async def test_live_single_synthesis_smoke():
    """Smoke test: synthesize 1 phrase using live Edge TTS and verify 16 kHz 16-bit PCM WAV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MassiveTTSAsyncEngine(output_dir=tmpdir, concurrency=1)
        dest_wav = os.path.join(tmpdir, "smoke_test_live.wav")

        success, duration = await engine.synthesize_single(
            phrase="Halo SVARA tolong nyalakan lampu kamar tidur",
            voice_id="id-ID-ArdiNeural",
            dest_wav_path=dest_wav,
        )

        assert success, "Live synthesis failed"
        assert duration > 0.5, "Duration too short"
        assert os.path.exists(dest_wav), "WAV file was not written to disk"

        # Verify acoustic format
        info = sf.info(dest_wav)
        assert info.samplerate == 16000, f"Expected 16kHz, got {info.samplerate}"
        assert info.channels == 1, f"Expected mono, got {info.channels} channels"
        assert info.subtype == "PCM_16", f"Expected PCM_16 bit depth, got {info.subtype}"

        # Test resumability / skip logic
        success_cached, duration_cached = await engine.synthesize_single(
            phrase="Halo SVARA tolong nyalakan lampu kamar tidur",
            voice_id="id-ID-ArdiNeural",
            dest_wav_path=dest_wav,
        )
        assert success_cached, "Cached file should be accepted immediately"
        assert abs(duration_cached - duration) < 1e-4, "Cached duration should match"
