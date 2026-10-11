"""Build a Perfectly Balanced Bilingual Dataset (English FSC + Robust Indonesian).

Step 2 of Robust SLU Pipeline:
1. Loads Indonesian Robust Train Split (1,953 acoustic-augmented clips).
2. Samples an evenly balanced subset of English FSC Native Speaker clips (~2,000 clips)
   covering all 31 intents and diverse speakers.
3. Merges them into a unified Bilingual Split: data/bilingual_dataset/splits/
4. Retains pure Indonesian validation/test sets to measure real-world Indonesian SLU generalization,
   and standard FSC validation/test sets to guarantee English capability is not degraded.
"""

import os
import json
import pandas as pd
import numpy as np

def make_bilingual_balanced_splits():
    print("[Step 2] Memulai peracikan Balanced Bilingual Dataset (English + Indonesia)...")
    
    # 1. Load Intent Map
    with open("configs/intent_map.json", "r", encoding="utf-8") as f:
        intent_map_data = json.load(f)
    intent_map = intent_map_data["intent_map"]
    
    # Mapping intent tuple to ID
    intent_lookup = {
        (v["action"], v["object"], v["location"]): int(k)
        for k, v in intent_map.items()
    }
    
    # 2. Load Indonesian Robust Train Data
    id_train_csv = "data/synthetic_indonesian/splits/train_robust.csv"
    if not os.path.exists(id_train_csv):
        raise FileNotFoundError(f"File {id_train_csv} tidak ditemukan. Jalankan Step 1 terlebih dahulu!")
    
    df_id_train = pd.read_csv(id_train_csv)
    # Ensure audio prefix path is absolute or referenced properly
    df_id_train["dataset_source"] = "indonesian"
    df_id_train["audio_full_path"] = df_id_train["path"].apply(
        lambda p: os.path.normpath(os.path.join("data/synthetic_indonesian", p))
    )
    if "intent_id" not in df_id_train.columns:
        df_id_train["intent_id"] = df_id_train.apply(
            lambda r: intent_lookup.get((r["action"], r["object"], r["location"]), -1), axis=1
        )
    print(f"- Indonesian Robust Train Clips: {len(df_id_train)} audio")
    
    # 3. Load English FSC Train Data
    en_train_csv = "data/raw/fluent_speech_commands_dataset/data/train_data.csv"
    if not os.path.exists(en_train_csv):
        raise FileNotFoundError(f"File {en_train_csv} tidak ditemukan!")
        
    df_en_all = pd.read_csv(en_train_csv)
    # Rename 'transcription' -> 'transcript' if needed
    if "transcription" in df_en_all.columns and "transcript" not in df_en_all.columns:
        df_en_all["transcript"] = df_en_all["transcription"]
        
    df_en_all["intent_id"] = df_en_all.apply(
        lambda r: intent_lookup.get((r["action"], r["object"], r["location"]), -1), axis=1
    )
    df_en_all["dataset_source"] = "english_fsc"
    df_en_all["audio_full_path"] = df_en_all["path"].apply(
        lambda p: os.path.normpath(os.path.join("data/raw/fluent_speech_commands_dataset", p))
    )
    
    # Sample FSC to match Indonesian size (~1,953 clips): ~65 clips per intent across all 31 intents
    target_per_intent = int(np.ceil(len(df_id_train) / 31))
    en_sampled_list = []
    
    for intent_id, group in df_en_all.groupby("intent_id"):
        n_sample = min(len(group), target_per_intent)
        en_sampled_list.append(group.sample(n=n_sample, random_state=42))
        
    df_en_sampled = pd.concat(en_sampled_list, ignore_index=True)
    print(f"- English FSC Sampled Clips (Balanced across 31 intents): {len(df_en_sampled)} audio")
    
    # 4. Merge into Balanced Bilingual Train Set
    common_cols = ["audio_full_path", "transcript", "action", "object", "location", "intent_id", "dataset_source", "speakerId"]
    
    df_bilingual_train = pd.concat([
        df_id_train[common_cols],
        df_en_sampled[common_cols]
    ], ignore_index=True)
    
    # Shuffle bilingual dataset
    df_bilingual_train = df_bilingual_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # 5. Save Splits
    out_dir = "data/bilingual_dataset/splits"
    os.makedirs(out_dir, exist_ok=True)
    
    bilingual_train_path = os.path.join(out_dir, "train_bilingual.csv")
    df_bilingual_train.to_csv(bilingual_train_path, index=False)
    
    # Also save separate evaluation benchmarks for Indonesian and English
    id_test_csv = "data/synthetic_indonesian/splits/test.csv"
    df_id_test = pd.read_csv(id_test_csv)
    df_id_test["audio_full_path"] = df_id_test["path"].apply(
        lambda p: os.path.normpath(os.path.join("data/synthetic_indonesian", p))
    )
    df_id_test["dataset_source"] = "indonesian"
    df_id_test["intent_id"] = df_id_test.apply(
        lambda r: intent_lookup.get((r["action"], r["object"], r["location"]), -1), axis=1
    )
    id_test_path = os.path.join(out_dir, "test_indonesian.csv")
    df_id_test[common_cols].to_csv(id_test_path, index=False)
    
    # English test benchmark (FSC test set sampled)
    en_test_csv = "data/raw/fluent_speech_commands_dataset/data/test_data.csv"
    df_en_test = pd.read_csv(en_test_csv)
    if "transcription" in df_en_test.columns and "transcript" not in df_en_test.columns:
        df_en_test["transcript"] = df_en_test["transcription"]
    df_en_test["audio_full_path"] = df_en_test["path"].apply(
        lambda p: os.path.normpath(os.path.join("data/raw/fluent_speech_commands_dataset", p))
    )
    df_en_test["dataset_source"] = "english_fsc"
    df_en_test["intent_id"] = df_en_test.apply(
        lambda r: intent_lookup.get((r["action"], r["object"], r["location"]), -1), axis=1
    )
    en_test_path = os.path.join(out_dir, "test_english.csv")
    df_en_test[common_cols].to_csv(en_test_path, index=False)
    
    print("\n" + "="*50)
    print(f"[Step 2 SELESAI] Balanced Bilingual Dataset Berhasil Dirancang!")
    print(f"- Total Training Bilingual : {len(df_bilingual_train)} audio")
    print(f"  * Indonesian (Acoustic)  : {len(df_id_train)} audio (50%)")
    print(f"  * English (Native FSC)   : {len(df_en_sampled)} audio (50%)")
    print(f"- Test Set Evaluasi:")
    print(f"  * Test Indonesia Unseen  : {len(df_id_test)} audio -> {id_test_path}")
    print(f"  * Test English FSC       : {len(df_en_test)} audio -> {en_test_path}")
    print(f"- File Training Bilingual  : {bilingual_train_path}")
    print("="*50)

if __name__ == "__main__":
    make_bilingual_balanced_splits()
