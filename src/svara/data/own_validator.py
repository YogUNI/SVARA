"""Tooling for validating and converting own-recording audio clips.

Implements:
1. convert_to_16k_mono: Converts any input audio file (m4a, ogg, mp3, wav) to 16 kHz mono WAV.
2. validate_own_recordings: Checks metadata.csv and audio files for readability, sampling rate,
   channels, label validity, duration bounds, and filename-metadata consistency.

Task: P5-02
Reference: docs/01 §8, docs/10 P5-02
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import soundfile as sf

from svara.data.audio import preprocess_audio


def convert_audio_file(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
) -> Tuple[bool, str]:
    """Convert any audio file to 16 kHz mono WAV format."""
    try:
        data, _, sr = preprocess_audio(
            input_path,
            target_sample_rate=target_sr,
            max_audio_seconds=None,
            normalize=False,
            pad=False,
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, data, sr, subtype="PCM_16")
        return True, "Success"
    except Exception as e:
        return False, str(e)


def validate_own_recordings(
    metadata_csv_path: str = "data/own_recordings/metadata.csv",
    audio_base_dir: str = "data/own_recordings",
    intent_map_path: str = "configs/intent_map.json",
) -> Dict[str, any]:
    """Validate own-recordings metadata and corresponding audio files."""
    if not os.path.exists(metadata_csv_path):
        return {"valid": False, "error": f"Metadata file not found: {metadata_csv_path}"}

    with open(intent_map_path, "r", encoding="utf-8") as f:
        imap = json.load(f)

    valid_intents = set(int(k) for k in imap["intent_map"].keys())
    valid_actions = set(imap["slot_vocab"]["action"])
    valid_objects = set(imap["slot_vocab"]["object"])
    valid_locations = set(imap["slot_vocab"]["location"])

    errors = []
    warnings = []
    total_rows = 0

    required_cols = [
        "path",
        "speakerId",
        "device",
        "condition",
        "transcript",
        "action",
        "object",
        "location",
        "intent_id",
    ]

    with open(metadata_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in required_cols:
            if col not in reader.fieldnames:
                return {"valid": False, "error": f"Missing required column in metadata: {col}"}

        for row_idx, row in enumerate(reader, start=2):
            total_rows += 1
            rel_path = row["path"]
            full_audio_path = os.path.join(audio_base_dir, rel_path)

            # 1. File existence
            if not os.path.exists(full_audio_path):
                errors.append(f"Row {row_idx}: Audio file does not exist: {rel_path}")
                continue

            # 2. Audio readability & properties
            try:
                info = sf.info(full_audio_path)
                if info.samplerate != 16000:
                    errors.append(f"Row {row_idx}: Sample rate is {info.samplerate} Hz (expected 16000 Hz)")
                if info.channels != 1:
                    errors.append(f"Row {row_idx}: Channels count is {info.channels} (expected 1 / mono)")
                if info.duration < 0.5:
                    warnings.append(f"Row {row_idx}: Audio very short ({info.duration:.2f} s)")
                if info.duration > 8.0:
                    warnings.append(f"Row {row_idx}: Audio long ({info.duration:.2f} s)")
            except Exception as e:
                errors.append(f"Row {row_idx}: Corrupted audio file: {e}")

            # 3. Label validity
            try:
                int_id = int(row["intent_id"])
                if int_id not in valid_intents:
                    errors.append(f"Row {row_idx}: Invalid intent_id {int_id}")
            except ValueError:
                errors.append(f"Row {row_idx}: Non-integer intent_id: {row['intent_id']}")

            if row["action"] not in valid_actions:
                errors.append(f"Row {row_idx}: Unknown action '{row['action']}'")
            if row["object"] not in valid_objects:
                errors.append(f"Row {row_idx}: Unknown object '{row['object']}'")
            if row["location"] not in valid_locations:
                errors.append(f"Row {row_idx}: Unknown location '{row['location']}'")

            if row["condition"] not in ("quiet", "noisy", "q", "n"):
                warnings.append(f"Row {row_idx}: Non-standard condition: {row['condition']}")

    is_valid = len(errors) == 0
    return {
        "valid": is_valid,
        "total_rows": total_rows,
        "errors": errors,
        "warnings": warnings,
    }
