"""Inspect Fluent Speech Commands dataset per PART 2 step 3.

Reference: docs/01 §1, docs/11 V-01..V-04
"""

import glob
import os
import random
import wave
import pandas as pd

dataset_dir = r"data/raw/fluent_speech_commands_dataset"

print("=== 1. STRUKTUR FILE & FOLDER ===")
pdf_path = os.path.join(dataset_dir, "Fluent Speech Commands Public License.pdf")
print("License PDF path:", pdf_path, "| Exists:", os.path.exists(pdf_path))
print("Top-level entries:", os.listdir(dataset_dir))

print("\n=== 2. HEADER CSV REAL ===")
csv_files = {
    "train": os.path.join(dataset_dir, "data", "train_data.csv"),
    "valid": os.path.join(dataset_dir, "data", "valid_data.csv"),
    "test": os.path.join(dataset_dir, "data", "test_data.csv"),
    "demo": os.path.join(dataset_dir, "data", "speaker_demographics.csv"),
}
for name, p in csv_files.items():
    with open(p, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    print(f"{name} header: {header}")

print("\n=== 3. JUMLAH BARIS VS TARGET PAPER ===")
dfs = {}
for name, p in csv_files.items():
    # Use index_col=0 for splits since first column is unnamed index
    if name != "demo":
        dfs[name] = pd.read_csv(p, index_col=0)
    else:
        dfs[name] = pd.read_csv(p)
    print(f"{name} rows: {len(dfs[name])}")
total_splits = len(dfs["train"]) + len(dfs["valid"]) + len(dfs["test"])
print(f"Total 3 splits: {total_splits} (Paper target: 30043 | Selisih: {total_splits - 30043})")

print("\n=== 4. FILE AUDIO WAV DI DISK & INTEGRITAS PATH ===")
raw_wav_list = glob.glob(os.path.join(dataset_dir, "wavs", "speakers", "*", "*.wav"))
wav_on_disk = set(os.path.normpath(p) for p in raw_wav_list)
print("Total file .wav di disk:", len(wav_on_disk))

all_audio_df = pd.concat([dfs["train"], dfs["valid"], dfs["test"]], ignore_index=True)
missing_paths = []
for idx, row in all_audio_df.iterrows():
    full_p = os.path.normpath(os.path.join(dataset_dir, str(row["path"])))
    if full_p not in wav_on_disk:
        missing_paths.append(row["path"])
print("Jumlah path CSV yang tidak ditemukan di disk:", len(missing_paths))

print("\n=== 5. UNIQUE SPEAKER, TRANSCRIPTION, INTENT TUPLES ===")
text_col = "transcription" if "transcription" in all_audio_df.columns else "transcript"
print(f"Kolom transkripsi teks yang digunakan: '{text_col}'")

def inspect_unique(df, label):
    intents = df[["action", "object", "location"]].drop_duplicates()
    n_spk = df["speakerId"].nunique()
    n_txt = df[text_col].nunique()
    print(f"[{label}] Speakers: {n_spk} | Transcriptions: {n_txt} | Intents: {len(intents)}")

inspect_unique(dfs["train"], "Train")
inspect_unique(dfs["valid"], "Valid")
inspect_unique(dfs["test"], "Test")
inspect_unique(all_audio_df, "Overall (Train+Valid+Test)")

print("\n=== 6. SET NILAI LABELS VS README ===")
actions = sorted(list(all_audio_df["action"].dropna().unique()))
objects = sorted(list(all_audio_df["object"].dropna().unique()))
locations = sorted(list(all_audio_df["location"].dropna().unique()))
print("Observed Actions  :", actions)
print("Observed Objects  :", objects)
print("Observed Locations:", locations)

readme_actions = sorted(["change language", "activate", "deactivate", "increase", "decrease", "bring"])
readme_objects = sorted(["none", "music", "lights", "volume", "heat", "lamp", "newspaper", "juice", "socks", "shoes", "Chinese", "Korean", "English", "German"])
readme_locations = sorted(["none", "kitchen", "bedroom", "washroom"])

print("Actions match README?", actions == readme_actions)
print("Objects match README?", objects == readme_objects)
print("Locations match README?", locations == readme_locations)

print("\n=== 7. SAMPLE-RATE & CHANNELS 20 FILE WAV ACAK ===")
random.seed(42)
sample_wavs = random.sample(sorted(list(wav_on_disk)), 20)
results = []
all_16k_mono = True
for sw in sample_wavs:
    with wave.open(sw, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        dur = wf.getnframes() / sr
        results.append((os.path.basename(sw), sr, ch, dur))
        if sr != 16000 or ch != 1:
            all_16k_mono = False

for r in results:
    print(f"  {r[0]} -> {r[1]} Hz, {r[2]} ch, {r[3]:.2f} s")
print("Semua 20 sampel 16000 Hz Mono?", all_16k_mono)
