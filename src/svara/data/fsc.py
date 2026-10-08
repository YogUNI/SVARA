"""Fluent Speech Commands (FSC) dataset loading and metadata utilities.

Task: P1-01
Reference: docs/01 §1-§2, docs/03 §4, AGENTS.md rule 3
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass(frozen=True)
class FSCPaths:
    """Standard filesystem paths for Fluent Speech Commands dataset."""

    root_dir: str
    train_csv: str
    valid_csv: str
    test_csv: str
    demographics_csv: str
    wavs_dir: str
    license_pdf: str
    readme_md: str

    @classmethod
    def from_root(cls, root_dir: str) -> "FSCPaths":
        root = os.path.abspath(root_dir)
        return cls(
            root_dir=root,
            train_csv=os.path.join(root, "data", "train_data.csv"),
            valid_csv=os.path.join(root, "data", "valid_data.csv"),
            test_csv=os.path.join(root, "data", "test_data.csv"),
            demographics_csv=os.path.join(root, "data", "speaker_demographics.csv"),
            wavs_dir=os.path.join(root, "wavs", "speakers"),
            license_pdf=os.path.join(root, "Fluent Speech Commands Public License.pdf"),
            readme_md=os.path.join(root, "readme.md"),
        )


def read_fsc_split_csv(csv_path: str, dataset_root: Optional[str] = None) -> pd.DataFrame:
    """Read an FSC split CSV with robust column and path resolution.

    Handles real CSV quirks:
    - First unnamed column used as index.
    - Resolves 'transcription' column and aliases to 'transcript' for consistency.
    - Resolves relative wav paths to absolute paths if dataset_root is provided.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"FSC split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Handle unnamed leading index column if present
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:].copy()

    # Align transcription/transcript column name
    if "transcription" in df.columns and "transcript" not in df.columns:
        df["transcript"] = df["transcription"]
    elif "transcript" in df.columns and "transcription" not in df.columns:
        df["transcription"] = df["transcript"]

    # Verify expected columns
    expected_cols = {"path", "speakerId", "transcript", "action", "object", "location"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing expected columns: {missing}")

    if dataset_root:
        root = os.path.abspath(dataset_root)
        df["abs_path"] = df["path"].apply(lambda p: os.path.normpath(os.path.join(root, str(p))))

    return df


def read_fsc_demographics_csv(csv_path: str) -> pd.DataFrame:
    """Read FSC speaker demographics CSV and clean column headers."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Demographics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Strip whitespace from column names (e.g. 'Self-reported fluency level ')
    df.columns = [c.strip() for c in df.columns]

    if "speakerId" not in df.columns:
        raise ValueError(f"Demographics CSV missing 'speakerId' column: {csv_path}")

    return df


def load_fsc_splits(
    dataset_root: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, valid, test, and demographics dataframes from an FSC dataset root."""
    paths = FSCPaths.from_root(dataset_root)
    train_df = read_fsc_split_csv(paths.train_csv, dataset_root=paths.root_dir)
    valid_df = read_fsc_split_csv(paths.valid_csv, dataset_root=paths.root_dir)
    test_df = read_fsc_split_csv(paths.test_csv, dataset_root=paths.root_dir)
    demo_df = read_fsc_demographics_csv(paths.demographics_csv)
    return train_df, valid_df, test_df, demo_df


def extract_intent_vocabularies(
    df: pd.DataFrame,
) -> Tuple[Dict[int, Dict[str, str]], Dict[str, List[str]]]:
    """Extract unique intents (action, object, location) and slot vocabularies from data.

    Returns:
        intent_map: {intent_id: {'action': ..., 'object': ..., 'location': ...}}
        slot_vocab: {'action': [...], 'object': [...], 'location': [...]}
    """
    unique_intents = (
        df[["action", "object", "location"]].drop_duplicates().sort_values(by=["action", "object", "location"])
    )

    intent_map: Dict[int, Dict[str, str]] = {}
    for intent_id, (_, row) in enumerate(unique_intents.iterrows()):
        intent_map[intent_id] = {
            "action": str(row["action"]),
            "object": str(row["object"]),
            "location": str(row["location"]),
        }

    slot_vocab: Dict[str, List[str]] = {
        "action": sorted(df["action"].dropna().unique().tolist()),
        "object": sorted(df["object"].dropna().unique().tolist()),
        "location": sorted(df["location"].dropna().unique().tolist()),
    }

    return intent_map, slot_vocab


def analyze_split_overlaps(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """Compute empirical speaker and transcript overlaps between splits (docs/01 §2 check 7)."""
    train_spk: Set[str] = set(train_df["speakerId"])
    valid_spk: Set[str] = set(valid_df["speakerId"])
    test_spk: Set[str] = set(test_df["speakerId"])

    train_txt: Set[str] = set(train_df["transcript"])
    valid_txt: Set[str] = set(valid_df["transcript"])
    test_txt: Set[str] = set(test_df["transcript"])

    test_total = len(test_df)
    test_leak_count = int(test_df["transcript"].isin(train_txt).sum())
    test_leak_pct = (test_leak_count / test_total * 100.0) if test_total > 0 else 0.0

    return {
        "speakers": {
            "train_unique": len(train_spk),
            "valid_unique": len(valid_spk),
            "test_unique": len(test_spk),
            "train_valid_overlap": len(train_spk & valid_spk),
            "train_test_overlap": len(train_spk & test_spk),
            "valid_test_overlap": len(valid_spk & test_spk),
        },
        "transcripts": {
            "train_unique": len(train_txt),
            "valid_unique": len(valid_txt),
            "test_unique": len(test_txt),
            "train_valid_overlap": len(train_txt & valid_txt),
            "train_test_overlap": len(train_txt & test_txt),
            "valid_test_overlap": len(valid_txt & test_txt),
            "test_utterances_in_train_count": test_leak_count,
            "test_utterances_in_train_pct": round(test_leak_pct, 2),
        },
    }
