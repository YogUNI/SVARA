"""PyTorch Dataset and dynamic collator with length bucketing for SVARA.

Task: P2-06
Reference: docs/02 §1 (wav2vec gotchas: no attention_mask, zero padding, length computation)
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from svara.data.audio import preprocess_audio


class FSCDataset(Dataset):
    """PyTorch Dataset for Fluent Speech Commands splits."""

    def __init__(
        self,
        split_csv_path: str,
        dataset_root: Optional[str] = "data/raw/fluent_speech_commands_dataset",
        max_audio_seconds: float = 5.0,
        is_training: bool = False,
    ):
        if not os.path.exists(split_csv_path):
            raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")

        self.df = pd.read_csv(split_csv_path)
        self.dataset_root = os.path.abspath(dataset_root) if dataset_root else ""
        self.max_audio_seconds = max_audio_seconds
        self.is_training = is_training

        # Precompute vocabulary sets and mappings for slot labels
        # Use global intent_map.json if present to ensure train and val share identical IDs
        intent_map_path = "configs/intent_map.json"
        if os.path.exists(intent_map_path):
            with open(intent_map_path, "r", encoding="utf-8") as f:
                imap = json.load(f)
            sv = imap["slot_vocab"]
            self.actions = sv["action"] if isinstance(sv["action"], list) else list(sv["action"].keys())
            self.objects = sv["object"] if isinstance(sv["object"], list) else list(sv["object"].keys())
            self.locations = sv["location"] if isinstance(sv["location"], list) else list(sv["location"].keys())

            self.action_to_id = {act: i for i, act in enumerate(self.actions)} if isinstance(sv["action"], list) else sv["action"]
            self.object_to_id = {obj: i for i, obj in enumerate(self.objects)} if isinstance(sv["object"], list) else sv["object"]
            self.location_to_id = {loc: i for i, loc in enumerate(self.locations)} if isinstance(sv["location"], list) else sv["location"]
        else:
            self.actions = sorted(self.df["action"].dropna().unique().tolist())
            self.objects = sorted(self.df["object"].dropna().unique().tolist())
            self.locations = sorted(self.df["location"].dropna().unique().tolist())
            self.action_to_id = {act: i for i, act in enumerate(self.actions)}
            self.object_to_id = {obj: i for i, obj in enumerate(self.objects)}
            self.location_to_id = {loc: i for i, loc in enumerate(self.locations)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = str(row["path"])
        abs_path = os.path.normpath(os.path.join(self.dataset_root, rel_path))

        # Shared unified audio preprocessing
        waveform, original_len, sr = preprocess_audio(
            abs_path,
            max_audio_seconds=self.max_audio_seconds,
            normalize=True,
            pad=False,  # Padding done dynamically in Collate
            is_training=self.is_training,
        )

        action_id = self.action_to_id.get(row["action"], 0)
        object_id = self.object_to_id.get(row["object"], 0)
        location_id = self.location_to_id.get(row["location"], 0)
        intent_id = int(row["intent_id"]) if "intent_id" in row else 0

        return {
            "waveform": torch.from_numpy(waveform),
            "length": original_len,
            "action_id": torch.tensor(action_id, dtype=torch.long),
            "object_id": torch.tensor(object_id, dtype=torch.long),
            "location_id": torch.tensor(location_id, dtype=torch.long),
            "intent_id": torch.tensor(intent_id, dtype=torch.long),
            "transcript": row.get("transcript", ""),
            "speakerId": row.get("speakerId", ""),
            "path": rel_path,
        }


def collate_fn_pad(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Dynamic padding collate function for variable-length audio batches.

    Zero-pads waveforms to the maximum length in the current batch.
    Does NOT produce attention_mask (per docs/02 gotchas for wav2vec2-base).
    """
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)

    batch_size = len(batch)
    padded_waveforms = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, item in enumerate(batch):
        wav = item["waveform"]
        padded_waveforms[i, : len(wav)] = wav

    return {
        "waveform": padded_waveforms,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "action_id": torch.stack([item["action_id"] for item in batch]),
        "object_id": torch.stack([item["object_id"] for item in batch]),
        "location_id": torch.stack([item["location_id"] for item in batch]),
        "intent_id": torch.stack([item["intent_id"] for item in batch]),
        "path": [item.get("path", "") for item in batch],
        "speakerId": [item.get("speakerId", "") for item in batch],
        "transcript": [item.get("transcript", "") for item in batch],
    }
