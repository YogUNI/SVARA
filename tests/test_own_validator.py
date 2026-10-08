"""Unit tests for own-recordings converter and validator tooling (Task P5-02)."""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf

from svara.data.own_validator import convert_audio_file, validate_own_recordings


def test_convert_audio_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a synthetic 44.1 kHz stereo audio
        src_path = os.path.join(tmpdir, "stereo_44k.wav")
        dst_path = os.path.join(tmpdir, "mono_16k.wav")

        stereo_data = np.random.uniform(-0.5, 0.5, size=(44100, 2)).astype(np.float32)
        sf.write(src_path, stereo_data, 44100)

        ok, msg = convert_audio_file(src_path, dst_path, target_sr=16000)
        assert ok, f"Conversion failed: {msg}"

        info = sf.info(dst_path)
        assert info.samplerate == 16000
        assert info.channels == 1


def test_validate_own_recordings():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "own_s01_q_00.wav")
        sf.write(wav_path, np.zeros(32000, dtype=np.float32), 16000)

        meta_csv = os.path.join(tmpdir, "metadata.csv")
        with open(meta_csv, "w", encoding="utf-8") as f:
            f.write("path,speakerId,device,condition,transcript,action,object,location,intent_id\n")
            f.write("own_s01_q_00.wav,own_s01,phone,quiet,Turn on the lamp,activate,lamp,none,0\n")

        res = validate_own_recordings(
            metadata_csv_path=meta_csv,
            audio_base_dir=tmpdir,
            intent_map_path="configs/intent_map.json",
        )
        assert res["valid"], f"Validation failed with errors: {res['errors']}"
        assert res["total_rows"] == 1
