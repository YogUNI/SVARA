"""CLI script to generate and save all FSC split protocols.

Generates:
- Split A (Original Benchmark)
- Split B1, B2, B3 (Unseen-Utterance draws with seeds 42, 43, 44)
- Split C (Speaker-Disjoint)
- split_manifest.json with integrity logs

Task: P2-01, P2-02, P2-03
Reference: docs/01 §4, docs/10 §6
"""

import argparse
import json
import os

import pandas as pd
import yaml

from svara.data.fsc import FSCPaths, load_fsc_splits
from svara.data.splits import (
    create_split_a,
    create_split_b,
    create_split_c,
    load_intent_mapping,
    save_split_csvs,
)


def make_all_splits(
    dataset_root: str = "data/raw/fluent_speech_commands_dataset",
    output_base_dir: str = "data/processed/splits",
    config_splits_path: str = "configs/splits.yaml",
    intent_map_path: str = "configs/intent_map.json",
):
    print(f"[01_make_splits] Loading splits config from {config_splits_path}...")
    with open(config_splits_path, "r", encoding="utf-8") as f:
        splits_cfg = yaml.safe_load(f)["splits"]

    intent_mapping = load_intent_mapping(intent_map_path)
    print(f"[01_make_splits] Loaded {len(intent_mapping)} intent mappings.")

    paths = FSCPaths.from_root(dataset_root)
    train_df, valid_df, test_df, demo_df = load_fsc_splits(paths.root_dir)
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    manifest = {"dataset_root": paths.root_dir, "splits": {}}

    # ---------------------------------------------------------
    # Split A: Original Benchmark
    # ---------------------------------------------------------
    print("[01_make_splits] Creating Split A (Original)...")
    tr_a, val_a, te_a = create_split_a(train_df, valid_df, test_df, intent_mapping)
    dir_a = os.path.join(output_base_dir, "A")
    save_split_csvs(tr_a, val_a, te_a, dir_a)
    manifest["splits"]["A"] = {
        "rule": "original_fsc_split",
        "counts": {"train": len(tr_a), "valid": len(val_a), "test": len(te_a), "total": len(tr_a) + len(val_a) + len(te_a)},
    }
    print(f"  Split A saved to {dir_a} (train: {len(tr_a)}, val: {len(val_a)}, test: {len(te_a)})")

    # ---------------------------------------------------------
    # Split B1, B2, B3: Unseen-Utterance (Honest Protocol)
    # ---------------------------------------------------------
    b_draws = splits_cfg["B"]["draws"]
    test_frac = splits_cfg["B"]["holdout_fractions"]["test"]
    val_frac = splits_cfg["B"]["holdout_fractions"]["valid"]
    min_phrases = splits_cfg["B"]["min_train_phrasings_per_intent"]

    for split_name, seed in b_draws.items():
        print(f"[01_make_splits] Creating Split {split_name} (Unseen-Utterance, seed {seed})...")
        tr_b, val_b, te_b, b_info = create_split_b(
            all_df=all_df,
            intent_mapping=intent_mapping,
            seed=seed,
            test_ratio=test_frac,
            valid_ratio=val_frac,
            min_train_phrasings=min_phrases,
        )
        dir_b = os.path.join(output_base_dir, split_name)
        save_split_csvs(tr_b, val_b, te_b, dir_b)
        manifest["splits"][split_name] = b_info
        print(f"  Split {split_name} saved to {dir_b} (train: {len(tr_b)}, val: {len(val_b)}, test: {len(te_b)})")

    # ---------------------------------------------------------
    # Split C: Speaker-Grouped Stratified
    # ---------------------------------------------------------
    c_seed = splits_cfg["C"]["seed"]
    tr_frac = splits_cfg["C"]["target_fractions"]["train"]
    val_c_frac = splits_cfg["C"]["target_fractions"]["valid"]
    print(f"[01_make_splits] Creating Split C (Speaker-Disjoint, seed {c_seed})...")
    tr_c, val_c, te_c, c_info = create_split_c(
        all_df=all_df,
        demo_df=demo_df,
        intent_mapping=intent_mapping,
        seed=c_seed,
        train_ratio=tr_frac,
        valid_ratio=val_c_frac,
    )
    dir_c = os.path.join(output_base_dir, "C")
    save_split_csvs(tr_c, val_c, te_c, dir_c)
    manifest["splits"]["C"] = c_info
    print(f"  Split C saved to {dir_c} (train: {len(tr_c)}, val: {len(val_c)}, test: {len(te_c)})")

    # ---------------------------------------------------------
    # Save split_manifest.json
    # ---------------------------------------------------------
    manifest_path = os.path.join(output_base_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[01_make_splits] Saved split manifest to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate and save FSC dataset splits.")
    parser.add_argument("--root", default="data/raw/fluent_speech_commands_dataset")
    parser.add_argument("--output-dir", default="data/processed/splits")
    parser.add_argument("--config", default="configs/splits.yaml")
    parser.add_argument("--intent-map", default="configs/intent_map.json")
    args = parser.parse_args()

    make_all_splits(
        dataset_root=args.root,
        output_base_dir=args.output_dir,
        config_splits_path=args.config,
        intent_map_path=args.intent_map,
    )


if __name__ == "__main__":
    main()
