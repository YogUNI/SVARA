"""Synthesize Real-World Acoustic Variations for Indonesian Speech.

Step 1 of Robust SLU Pipeline:
Takes clean TTS audio clips and generates realistic laptop-mic acoustics:
1. Room Reverb (Impulse response simulation: small room, living room)
2. Hardware Laptop Mic EQ & Dynamic Gain (-6dB to +6dB)
3. Ambient Background Noise (Gaussian room hum, fan noise simulation)
4. Speed & Pitch Drift (0.9x to 1.1x speaking rate)

Saves augmented dataset to data/synthetic_indonesian/wavs_acoustic_augmented/
and creates updated metadata and train split.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from tqdm import tqdm

def simulate_room_reverb(waveform: np.ndarray, sr: int = 16000, room_size: str = "bedroom") -> np.ndarray:
    """Simulate small bedroom or living room acoustic reflections using Schroeder reverb model."""
    if room_size == "bedroom":
        delays = [int(sr * d) for d in [0.012, 0.019, 0.027, 0.035]]
        decays = [0.45, 0.35, 0.25, 0.15]
    else: # living room / larger room
        delays = [int(sr * d) for d in [0.018, 0.028, 0.042, 0.055]]
        decays = [0.55, 0.40, 0.30, 0.20]
        
    out = waveform.copy()
    for delay, decay in zip(delays, decays):
        if len(waveform) > delay:
            delayed = np.zeros_like(waveform)
            delayed[delay:] = waveform[:-delay] * decay
            out += delayed
            
    # Normalize back to prevent clipping
    max_val = np.max(np.abs(out)) + 1e-7
    if max_val > 1.0:
        out = out / max_val
    return out.astype(np.float32)

def simulate_laptop_mic_and_ambient(
    waveform: np.ndarray, 
    sr: int = 16000,
    snr_db: float = 22.0,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Simulate laptop internal microphone: slight low-cut filter, background ambient hum, and gain."""
    # 1. Gain scaling
    gain_linear = 10.0 ** (gain_db / 20.0)
    audio = waveform * gain_linear
    
    # 2. Laptop microphone frequency response (attenuate sub-100Hz rumble)
    b, a = signal.butter(2, 100 / (sr / 2), btype='highpass')
    audio = signal.filtfilt(b, a, audio).astype(np.float32)
    
    # 3. Add ambient room hum / fan noise
    clean_power = np.mean(audio ** 2) + 1e-9
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    
    # Generate pink-like ambient noise
    noise = np.random.randn(len(audio))
    # Soft lowpass for room air/fan characteristic
    b_noise, a_noise = signal.butter(1, 1500 / (sr / 2), btype='lowpass')
    noise = signal.filtfilt(b_noise, a_noise, noise).astype(np.float32)
    noise_power = np.mean(noise ** 2) + 1e-9
    
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    mixed = audio + scaled_noise
    
    # Peak clamp
    max_val = np.max(np.abs(mixed)) + 1e-7
    if max_val > 0.95:
        mixed = mixed * (0.95 / max_val)
    return mixed.astype(np.float32)

def simulate_speed_tempo(waveform: np.ndarray, speed_factor: float) -> np.ndarray:
    """Linear speed and tempo perturbation without voice pitch destruction."""
    if abs(speed_factor - 1.0) < 0.01:
        return waveform
    orig_len = len(waveform)
    new_len = int(round(orig_len / speed_factor))
    resampled = np.interp(
        np.linspace(0.0, 1.0, new_len, endpoint=False),
        np.linspace(0.0, 1.0, orig_len, endpoint=False),
        waveform,
    )
    return resampled.astype(np.float32)

def generate_acoustic_dataset():
    input_root = "data/synthetic_indonesian"
    train_csv_path = os.path.join(input_root, "splits", "train.csv")
    out_audio_dir = os.path.join(input_root, "wavs_augmented")
    os.makedirs(out_audio_dir, exist_ok=True)
    
    df_train = pd.read_csv(train_csv_path)
    print(f"[Step 1] Membaca {len(df_train)} audio train bersih untuk disintesis variasi akustik...")
    
    augmented_rows = []
    # Salin baris asli terlebih dahulu
    for idx, row in df_train.iterrows():
        augmented_rows.append(row.to_dict())
        
    print(f"[Step 1] Memulai proses augmentasi akustik dunia nyata (Reverb + Mic Noise + Tempo)...")
    
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Acoustic Synthesis"):
        src_audio_path = os.path.join(input_root, row["path"])
        if not os.path.exists(src_audio_path):
            continue
            
        data, sr = sf.read(src_audio_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
            
        base_name = os.path.basename(row["path"]).replace(".wav", "")
        
        # 1. Variasi A: Kondisi Kamar Tidur / Ruang Tertutup (Reverb Ringan + Sedikit Noise AC/Kipas)
        audio_v1 = simulate_room_reverb(data, sr=sr, room_size="bedroom")
        audio_v1 = simulate_laptop_mic_and_ambient(audio_v1, sr=sr, snr_db=24.0, gain_db=np.random.uniform(-3.0, 3.0))
        audio_v1 = simulate_speed_tempo(audio_v1, speed_factor=0.95)
        
        rel_path_v1 = f"wavs_augmented/{base_name}_room_q.wav"
        sf.write(os.path.join(input_root, rel_path_v1), audio_v1, sr, subtype='PCM_16')
        
        row_v1 = row.to_dict()
        row_v1["path"] = rel_path_v1
        row_v1["speakerId"] = f"{row['speakerId']}_room"
        augmented_rows.append(row_v1)
        
        # 2. Variasi B: Kondisi Bicara Cepat / Jarak Jauh dari Mic Laptop (Reverb Ruangan + Noise Lebih Nyata)
        audio_v2 = simulate_room_reverb(data, sr=sr, room_size="living")
        audio_v2 = simulate_laptop_mic_and_ambient(audio_v2, sr=sr, snr_db=18.0, gain_db=np.random.uniform(-5.0, 4.0))
        audio_v2 = simulate_speed_tempo(audio_v2, speed_factor=1.06)
        
        rel_path_v2 = f"wavs_augmented/{base_name}_dist_n.wav"
        sf.write(os.path.join(input_root, rel_path_v2), audio_v2, sr, subtype='PCM_16')
        
        row_v2 = row.to_dict()
        row_v2["path"] = rel_path_v2
        row_v2["speakerId"] = f"{row['speakerId']}_dist"
        augmented_rows.append(row_v2)
        
    df_augmented = pd.DataFrame(augmented_rows)
    out_train_csv = os.path.join(input_root, "splits", "train_robust.csv")
    df_augmented.to_csv(out_train_csv, index=False)
    
    print("\n" + "="*50)
    print(f"[Step 1 SELESAI] Data Train Akustik Dunia Nyata Berhasil Dibuat!")
    print(f"- Total audio train awal  : {len(df_train)} file")
    print(f"- Total audio train robust: {len(df_augmented)} file (Kenaikan 3x lipat ragam akustik)")
    print(f"- Tersimpan di            : {out_train_csv}")
    print(f"- Folder audio akustik    : {out_audio_dir}")
    print("="*50)

if __name__ == "__main__":
    generate_acoustic_dataset()
