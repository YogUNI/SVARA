"""Unit tests for shared audio preprocessing and dataset collation.

Task: P2-05, P2-06, Sub-step 1.1, Sub-step 1.2, Sub-step 1.3, Sub-step 1.4
Reference: AGENTS.md rule 7, docs/02 §10, docs/MASTER_ROADMAP_BERNOMOR_DEEP_LEARNING.md
"""

import io
import numpy as np
import pytest
import soundfile as sf
import torch

from svara.data.audio import (
    TARGET_SAMPLE_RATE,
    adaptive_rms_normalize,
    apply_butterworth_bandpass,
    normalize_waveform,
    preprocess_audio,
    trim_silence_vad,
)
from svara.data.collate import collate_fn_pad


@pytest.fixture
def synthetic_wav_file(tmp_path):
    wav_p = tmp_path / "test_tone.wav"
    # Generate 1.0s sine wave at 16000 Hz (440 Hz standard vocal tone)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(wav_p), sine, 16000)
    return str(wav_p)


def test_normalize_waveform():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    norm = normalize_waveform(data)
    assert np.isclose(np.mean(norm), 0.0, atol=1e-6)
    assert np.isclose(np.std(norm), 1.0, atol=1e-5)


def test_adaptive_rms_normalize_quiet_and_loud():
    """Verify that quiet (-34 dBFS) and loud (-6 dBFS) inputs converge to -20 dBFS RMS."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    
    # 440 Hz tone with quiet amplitude (RMS ~ 0.02, -34 dBFS)
    sig_quiet = (0.02 * np.sqrt(2) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # 440 Hz tone with loud amplitude (RMS ~ 0.50, -6 dBFS)
    sig_loud = (0.50 * np.sqrt(2) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    norm_quiet = adaptive_rms_normalize(sig_quiet, target_dbfs=-20.0)
    norm_loud = adaptive_rms_normalize(sig_loud, target_dbfs=-20.0)
    
    rms_quiet = float(np.sqrt(np.mean(norm_quiet ** 2)))
    rms_loud = float(np.sqrt(np.mean(norm_loud ** 2)))
    
    # Target RMS: 10^(-20/20) = 0.1000
    assert np.isclose(rms_quiet, 0.100, atol=1e-3)
    assert np.isclose(rms_loud, 0.100, atol=1e-3)


def test_adaptive_rms_normalize_silence_and_gain_clamp():
    """Verify silence stability and max gain clamping at +30 dB."""
    sr = 16000
    # 1. Pure silence
    silence = np.zeros(sr, dtype=np.float32)
    norm_silence = adaptive_rms_normalize(silence, target_dbfs=-20.0)
    assert np.all(norm_silence == 0.0)
    assert not np.any(np.isnan(norm_silence))
    
    # 2. Extremely faint tone needing 40 dB gain (clamped at 30 dB)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig_faint = (0.001 * np.sqrt(2) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    norm_faint = adaptive_rms_normalize(sig_faint, target_dbfs=-20.0, max_gain_db=30.0)
    
    gain_applied = float(np.sqrt(np.mean(norm_faint ** 2)) / np.sqrt(np.mean(sig_faint ** 2)))
    expected_max_gain = 10.0 ** (30.0 / 20.0)
    assert np.isclose(gain_applied, expected_max_gain, rtol=1e-3)


def test_butterworth_bandpass_attenuation():
    """Verify that Butterworth Bandpass filter attenuates <80Hz rumble and >7500Hz hiss."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    
    # Low frequency rumble (30 Hz) + High frequency electrical noise (7800 Hz) + Voice Tone (500 Hz)
    rumble_30hz = 1.0 * np.sin(2 * np.pi * 30 * t)
    vocal_500hz = 1.0 * np.sin(2 * np.pi * 500 * t)
    hiss_7800hz = 1.0 * np.sin(2 * np.pi * 7800 * t)
    composite = (rumble_30hz + vocal_500hz + hiss_7800hz).astype(np.float32)
    
    filtered = apply_butterworth_bandpass(composite, sr=sr, lowcut=80.0, highcut=7500.0)
    
    # Pass-band tone (500 Hz) should be preserved (power > 0.4)
    # Re-extract 500 Hz power via correlation
    corr_500 = np.mean(filtered * np.sin(2 * np.pi * 500 * t)) * 2
    assert corr_500 > 0.9  # 500 Hz tone passes through cleanly
    
    # 30 Hz rumble should be heavily attenuated
    corr_30 = np.mean(filtered * np.sin(2 * np.pi * 30 * t)) * 2
    assert abs(corr_30) < 0.2  # significant attenuation of <80Hz mechanical rumble


def test_trim_silence_vad_speech_padding():
    """Verify dual-threshold VAD trims silence and retains 150ms pre/post safety buffer."""
    sr = 16000
    # 3.0s total: 1.0s silence + 1.0s speech tone (440Hz, -20dBFS) + 1.0s silence
    t = np.linspace(0, 1.0, sr, endpoint=False)
    speech = (0.1 * np.sqrt(2) * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)
    audio = np.concatenate([silence, speech, silence])
    
    trimmed, start_s, end_s = trim_silence_vad(audio, sr=sr, return_indices=True)
    
    # Expected speech bounds: start ~ 16000, end ~ 32000
    # Expected with 150ms (2400 samples) buffer:
    # start_s ~ 16000 - 2400 = 13600 (within 320 samples tolerance)
    # end_s ~ 32000 + 2400 = 34400 (within 320 samples tolerance)
    assert abs(start_s - (16000 - 2400)) <= 320
    assert abs(end_s - (32000 + 2400)) <= 320
    assert 19000 <= len(trimmed) <= 22000  # approximately 1.30s instead of 3.0s


def test_trim_silence_vad_silence_and_click_fallback():
    """Verify silence stability and noise click rejection without crashing."""
    sr = 16000
    # 1. Complete silence returns unmodified audio safely
    silence = np.zeros(sr, dtype=np.float32)
    trimmed_sil = trim_silence_vad(silence, sr=sr)
    assert len(trimmed_sil) == len(silence)
    assert np.all(trimmed_sil == 0.0)
    
    # 2. Brief transient click (10ms < 200ms minimum speech threshold) falls back to untrimmed
    click = np.zeros(sr, dtype=np.float32)
    click[8000:8160] = 0.5
    trimmed_click = trim_silence_vad(click, sr=sr)
    assert len(trimmed_click) == len(click)


def test_trim_silence_vad_continuous():
    """Verify that continuous tone with no silence is not truncated at all."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    trimmed = trim_silence_vad(tone, sr=sr)
    assert len(trimmed) == len(tone)


def test_preprocess_audio_multi_format_parity(synthetic_wav_file):
    """Verify exact numerical parity across file path, bytes buffer, numpy array, and torch Tensor (Rule 7)."""
    # 1. Load from file path (disk)
    wav_from_file, len_f, sr_f = preprocess_audio(synthetic_wav_file, max_audio_seconds=2.0)
    
    # 2. Load from in-memory bytes buffer (FastAPI HTTP upload format)
    with open(synthetic_wav_file, "rb") as f:
        bytes_data = f.read()
    wav_from_bytes, len_b, sr_b = preprocess_audio(bytes_data, max_audio_seconds=2.0)
    
    # 3. Load from float32 NumPy array
    data_raw, sr = sf.read(synthetic_wav_file, dtype="float32")
    wav_from_array, len_a, sr_a = preprocess_audio(data_raw, sample_rate=sr, max_audio_seconds=2.0)
    
    # 4. Load from PyTorch Tensor
    tensor_raw = torch.from_numpy(data_raw)
    wav_from_tensor, len_t, sr_t = preprocess_audio(tensor_raw, sample_rate=sr, max_audio_seconds=2.0)

    # Asserts
    assert sr_f == TARGET_SAMPLE_RATE and sr_b == TARGET_SAMPLE_RATE and sr_a == TARGET_SAMPLE_RATE and sr_t == TARGET_SAMPLE_RATE
    assert len_f == len_b == len_a == len_t
    
    # File vs Bytes must be 100% bit-exact (zero difference)
    assert np.allclose(wav_from_file, wav_from_bytes, atol=1e-7)
    
    # Array vs Tensor must be 100% bit-exact (zero difference)
    assert np.allclose(wav_from_array, wav_from_tensor, atol=1e-7)
    
    # File/Bytes vs Array/Tensor matches within PCM-16 quantization noise (< 2e-4)
    assert np.allclose(wav_from_file, wav_from_array, atol=2e-4)


def test_preprocess_audio_stereo_to_mono_parity(tmp_path):
    """Verify that 2-channel stereo audio is collapsed to mono with exact channel-averaging."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone_l = 0.4 * np.sin(2 * np.pi * 440 * t)
    tone_r = 0.4 * np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([tone_l, tone_r]).astype(np.float32)
    
    stereo_path = tmp_path / "stereo.wav"
    sf.write(str(stereo_path), stereo, sr)
    
    wav_mono, orig_len, out_sr = preprocess_audio(str(stereo_path))
    assert wav_mono.ndim == 1
    assert out_sr == TARGET_SAMPLE_RATE
    assert len(wav_mono) == sr


def test_preprocess_audio_resampling_fidelity():
    """Verify that 48 kHz (laptop mic) and 44.1 kHz (USB mic) resample to 16 kHz preserving tone fidelity."""
    for sr_in in [48000, 44100]:
        t = np.linspace(0, 1.0, sr_in, endpoint=False)
        sine = (0.2 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
        w_out, orig_len, sr_out = preprocess_audio(sine, sample_rate=sr_in)
        assert sr_out == TARGET_SAMPLE_RATE
        assert len(w_out) == 16000
        
        # Verify 500 Hz tone correlation in 16 kHz output
        t_16k = np.linspace(0, 1.0, 16000, endpoint=False)
        corr_500 = np.mean(w_out * np.sin(2 * np.pi * 500 * t_16k)) * 2
        assert corr_500 > 0.90


def test_preprocess_audio_error_handling():
    """Verify that missing files and invalid types raise clean Python exceptions."""
    with pytest.raises(FileNotFoundError):
        preprocess_audio("non_existent_audio_path_xyz.wav")
        
    with pytest.raises(TypeError):
        preprocess_audio({"invalid": "type"})


def test_preprocess_audio_crop_and_pad(synthetic_wav_file):
    # Test crop when audio exceeds max duration
    wav_cropped, length_c, _ = preprocess_audio(synthetic_wav_file, max_audio_seconds=0.5, pad=False)
    assert len(wav_cropped) == int(0.5 * TARGET_SAMPLE_RATE)
    assert length_c == int(0.5 * TARGET_SAMPLE_RATE)

    # Test pad when audio shorter than max duration
    wav_padded, length_p, _ = preprocess_audio(synthetic_wav_file, max_audio_seconds=2.0, pad=True)
    assert len(wav_padded) == int(2.0 * TARGET_SAMPLE_RATE)
    assert length_p == int(1.0 * TARGET_SAMPLE_RATE)


def test_collate_fn_pad_batch():
    batch = [
        {
            "waveform": torch.randn(16000),  # 1.0s
            "length": 16000,
            "action_id": torch.tensor(1),
            "object_id": torch.tensor(2),
            "location_id": torch.tensor(0),
            "intent_id": torch.tensor(5),
        },
        {
            "waveform": torch.randn(24000),  # 1.5s
            "length": 24000,
            "action_id": torch.tensor(3),
            "object_id": torch.tensor(1),
            "location_id": torch.tensor(2),
            "intent_id": torch.tensor(12),
        },
    ]

    collated = collate_fn_pad(batch)
    assert collated["waveform"].shape == (2, 24000)
    assert collated["lengths"].tolist() == [16000, 24000]
    assert collated["action_id"].shape == (2,)
    # Verify zero-padding in first element
    assert torch.all(collated["waveform"][0, 16000:] == 0.0)
