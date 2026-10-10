"""Create stratified train/valid/test splits for the Indonesian synthetic speech dataset.

Follows SVARA guidelines:
- Stratified by intent_id to ensure balanced coverage across all 31 intents.
- Distributes male (tts_ardi) and female (tts_gadis) speakers proportionally.
- Default split: 70% Train, 15% Validation, 15% Test.
"""

import os
import pandas as pd
import numpy as np

def make_indonesian_splits():
    meta_path = "data/synthetic_indonesian/metadata.csv"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)
    print(f"Total baris data: {len(df)}")

    # Set seed for exact reproducibility
    np.random.seed(42)

    # Shuffling per intent group to ensure balanced train/val/test
    train_rows = []
    valid_rows = []
    test_rows = []

    for intent_id, group in df.groupby("intent_id"):
        shuffled = group.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        
        train_rows.append(shuffled.iloc[:n_train])
        valid_rows.append(shuffled.iloc[n_train:n_train + n_val])
        test_rows.append(shuffled.iloc[n_train + n_val:])

    train_df = pd.concat(train_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    valid_df = pd.concat(valid_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_rows).sample(frac=1.0, random_state=42).reset_index(drop=True)

    output_dir = "data/synthetic_indonesian/splits"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    valid_path = os.path.join(output_dir, "valid.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Split berhasil dibuat!")
    print(f"- Train: {len(train_df)} sampel ({len(train_df)/len(df)*100:.1f}%) -> {train_path}")
    print(f"- Valid: {len(valid_df)} sampel ({len(valid_df)/len(df)*100:.1f}%) -> {valid_path}")
    print(f"- Test:  {len(test_df)} sampel ({len(test_df)/len(df)*100:.1f}%) -> {test_path}")

if __name__ == "__main__":
    make_indonesian_splits()
