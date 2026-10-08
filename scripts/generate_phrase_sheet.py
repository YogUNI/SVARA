"""Generate printable phrase sheet with 31 FSC intent phrasings for own-recording sessions.

Task: P5-01
Reference: docs/01 §8, docs/10 P5-01
"""

import json
import os
import pandas as pd


def generate_phrase_sheet(
    train_csv_path: str = "data/processed/splits/A/train.csv",
    intent_map_path: str = "configs/intent_map.json",
    output_path: str = "data/own_recordings/phrase_sheet.md",
):
    with open(intent_map_path, "r", encoding="utf-8") as f:
        imap = json.load(f)

    df = pd.read_csv(train_csv_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# SVARA — Lembar Panduan Rekaman Suara (Phrase Sheet)",
        "",
        "Petunjuk untuk Pembicara (Haikal & Responden):",
        "1. Bacalah setiap kalimat perintah di bawah ini dengan intonasi natural dan jelas.",
        "2. Setiap pembicara membaca seluruh 31 kalimat dalam 2 kondisi:",
        "   - **Kondisi Tenang (q / quiet)**: Ruangan hening/kamar.",
        "   - **Kondisi Berisik (n / noisy)**: Dengan latar belakang suara TV, kipas angin, atau dapur.",
        "3. Format penamaan file WAV: `own_<speakerId>_<kondisi>_<intent_id:02d>.wav`",
        "   - Contoh: `own_s01_q_00.wav`, `own_s01_n_00.wav`",
        "",
        "| No | Intent ID | Kalimat Perintah yang Dibaca | Action | Object | Location |",
        "|---|---|---|---|---|---|",
    ]

    for intent_id in range(31):
        sub = df[df["intent_id"] == intent_id]
        if len(sub) > 0:
            sample = sub.iloc[0]
            transcript = sample["transcript"]
            action = sample["action"]
            obj = sample["object"]
            loc = sample["location"]
        else:
            info = imap["intent_map"][str(intent_id)]
            transcript = f"{info['action']} {info['object']} {info['location']}"
            action = info["action"]
            obj = info["object"]
            loc = info["location"]

        lines.append(
            f'| {intent_id + 1} | {intent_id:02d} | **"{transcript}"** | `{action}` | `{obj}` | `{loc}` |'
        )

    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[P5-01] Successfully generated phrase sheet at: {output_path}")


if __name__ == "__main__":
    generate_phrase_sheet()
