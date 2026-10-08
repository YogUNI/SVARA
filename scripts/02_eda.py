"""Exploratory Data Analysis (EDA) script for Fluent Speech Commands.

Generates 300 dpi figures and CSV tables per docs/01 §3 and docs/07 §2.
Applies consistent SVARA palette tokens (docs/06 §3).

Task: P1-05, P1-06
Reference: docs/01 §3, docs/06 §3, docs/07 §2
"""

import argparse
import json
import math
import os
import wave

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from svara.data.fsc import FSCPaths, load_fsc_splits

# SVARA Palette Tokens (docs/06 §3)
COLOR_PINE = "#2F6B5E"
COLOR_INK = "#16303A"
COLOR_BRICK = "#C4492F"
COLOR_LAMP = "#F2B33D"
COLOR_MIST = "#A9B8B2"
COLOR_PLASTER = "#E8ECE6"
COLOR_PAPER = "#F6F8F4"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": COLOR_INK,
        "axes.linewidth": 1.0,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.color": COLOR_INK,
        "ytick.color": COLOR_INK,
        "figure.titlesize": 14,
    }
)


def compute_durations(df: pd.DataFrame) -> list[float]:
    """Read durations of wav files from disk using standard wave module."""
    durations = []
    for p in df["abs_path"]:
        with wave.open(p, "rb") as wf:
            durations.append(wf.getnframes() / wf.getframerate())
    return durations


def run_eda(
    dataset_root: str,
    output_fig_dir: str = "reports/figures",
    output_tab_dir: str = "reports/tables",
    configs_base_path: str = "configs/base.yaml",
):
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(output_tab_dir, exist_ok=True)

    paths = FSCPaths.from_root(dataset_root)
    print(f"[02_eda] Loading FSC data from {paths.root_dir}...")
    train_df, valid_df, test_df, demo_df = load_fsc_splits(paths.root_dir)
    all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)

    # -------------------------------------------------------------
    # 1. Class distributions per slot and per intent
    # -------------------------------------------------------------
    print("[02_eda] Generating class distributions (slots & intents)...")
    slots = ["action", "object", "location"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(COLOR_PAPER)

    for i, slot in enumerate(slots):
        counts = all_df[slot].value_counts().sort_values(ascending=False)
        counts.to_frame(name="count").to_csv(
            os.path.join(output_tab_dir, f"eda_dist_{slot}.csv")
        )
        ax = axes[i]
        ax.set_facecolor(COLOR_PAPER)
        bars = ax.bar(
            counts.index, counts.values, color=COLOR_PINE, edgecolor=COLOR_INK, linewidth=0.8
        )
        ax.set_title(f"Distribusi Slot: {slot.capitalize()}")
        ax.set_ylabel("Jumlah Contoh (Clips)" if i == 0 else "")
        ax.tick_params(axis="x", rotation=45 if slot != "location" else 0)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color=COLOR_MIST)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "eda_slots_distribution.png"), dpi=300)
    plt.close(fig)

    # 31 Intents distribution
    intent_series = all_df.apply(
        lambda r: f"{r['action']}|{r['object']}|{r['location']}", axis=1
    ).value_counts()
    intent_series.to_frame(name="count").to_csv(
        os.path.join(output_tab_dir, "eda_dist_intents.csv")
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(COLOR_PAPER)
    ax.set_facecolor(COLOR_PAPER)
    intent_series.plot(kind="barh", ax=ax, color=COLOR_PINE, edgecolor=COLOR_INK)
    ax.set_title("Distribusi 31 Kombinasi Intent (Action | Object | Location)")
    ax.set_xlabel("Jumlah Contoh (Clips)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4, color=COLOR_MIST)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "eda_intents_distribution.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. Clips per speaker & Phrases per intent
    # -------------------------------------------------------------
    print("[02_eda] Generating speaker and phrasing distributions...")
    speaker_counts = all_df["speakerId"].value_counts()
    speaker_counts.to_frame(name="count").to_csv(
        os.path.join(output_tab_dir, "eda_clips_per_speaker.csv")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLOR_PAPER)
    ax.set_facecolor(COLOR_PAPER)
    ax.hist(
        speaker_counts.values,
        bins=25,
        color=COLOR_PINE,
        edgecolor=COLOR_INK,
        linewidth=0.8,
    )
    ax.set_title("Histogram Jumlah Audio per Speaker (Total 97 Speakers)")
    ax.set_xlabel("Jumlah Audio per Speaker")
    ax.set_ylabel("Frekuensi (Jumlah Speaker)")
    ax.grid(axis="y", linestyle="--", alpha=0.4, color=COLOR_MIST)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "eda_clips_per_speaker.png"), dpi=300)
    plt.close(fig)

    phrases_per_intent = all_df.groupby(["action", "object", "location"])[
        "transcript"
    ].nunique()
    phrases_per_intent.to_frame(name="num_phrasings").to_csv(
        os.path.join(output_tab_dir, "eda_phrases_per_intent.csv")
    )

    # -------------------------------------------------------------
    # 3. Audio duration histogram & percentiles (p95, p99)
    # -------------------------------------------------------------
    print("[02_eda] Computing audio durations for 30,043 clips...")
    durations = compute_durations(all_df)
    all_df["duration_s"] = durations

    dur_min = float(np.min(durations))
    dur_mean = float(np.mean(durations))
    dur_p50 = float(np.percentile(durations, 50))
    dur_p95 = float(np.percentile(durations, 95))
    dur_p99 = float(np.percentile(durations, 99))
    dur_max = float(np.max(durations))

    # Recommended max_audio_seconds: round up p99
    recommended_max_sec = math.ceil(dur_p99)

    duration_stats = {
        "min_s": round(dur_min, 3),
        "mean_s": round(dur_mean, 3),
        "p50_s": round(dur_p50, 3),
        "p95_s": round(dur_p95, 3),
        "p99_s": round(dur_p99, 3),
        "max_s": round(dur_max, 3),
        "recommended_max_audio_seconds": recommended_max_sec,
    }
    with open(
        os.path.join(output_tab_dir, "eda_duration_stats.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(duration_stats, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLOR_PAPER)
    ax.set_facecolor(COLOR_PAPER)
    n, bins, patches = ax.hist(
        durations, bins=40, color=COLOR_PINE, edgecolor=COLOR_INK, linewidth=0.8
    )
    ax.axvline(
        dur_mean,
        color=COLOR_INK,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {dur_mean:.2f} s",
    )
    ax.axvline(
        dur_p95,
        color=COLOR_LAMP,
        linestyle="-.",
        linewidth=1.8,
        label=f"p95: {dur_p95:.2f} s",
    )
    ax.axvline(
        dur_p99,
        color=COLOR_BRICK,
        linestyle=":",
        linewidth=2.0,
        label=f"p99: {dur_p99:.2f} s",
    )
    ax.axvline(
        recommended_max_sec,
        color=COLOR_PINE,
        linestyle="-",
        linewidth=2.0,
        label=f"Cutoff (P1-06): {recommended_max_sec}.0 s",
    )
    ax.set_title("Histogram Durasi Audio Fluent Speech Commands (30.043 clips)")
    ax.set_xlabel("Durasi (detik)")
    ax.set_ylabel("Jumlah Contoh")
    ax.legend(frameon=True, facecolor=COLOR_PAPER, edgecolor=COLOR_MIST)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color=COLOR_MIST)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "eda_duration_histogram.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 4. Demographic distributions
    # -------------------------------------------------------------
    print("[02_eda] Generating demographic distributions...")
    demo_df.to_csv(
        os.path.join(output_tab_dir, "eda_demographics_summary.csv"), index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(COLOR_PAPER)

    demo_features = [
        ("gender", "Gender Speaker"),
        ("ageRange", "Rentang Usia (Age Range)"),
        ("Self-reported fluency level", "Kemampuan Berbahasa (Fluency)"),
        ("First Language spoken", "Bahasa Ibu (First Language)"),
    ]

    for idx, (col, title) in enumerate(demo_features):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor(COLOR_PAPER)
        if col in demo_df.columns:
            counts = demo_df[col].value_counts()
            # If too many categories (like first language), take top 6 + other
            if len(counts) > 6:
                top_counts = counts.head(5)
                other_sum = counts.iloc[5:].sum()
                counts = pd.concat([top_counts, pd.Series({"Other": other_sum})])
            bars = ax.bar(
                counts.index,
                counts.values,
                color=COLOR_PINE,
                edgecolor=COLOR_INK,
                linewidth=0.8,
            )
            ax.set_title(title)
            ax.set_ylabel("Jumlah Speaker")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", linestyle="--", alpha=0.4, color=COLOR_MIST)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_fig_dir, "eda_demographic_distribution.png"), dpi=300
    )
    plt.close(fig)

    # -------------------------------------------------------------
    # 5. Waveform + Log-Mel spectrogram examples
    # -------------------------------------------------------------
    print("[02_eda] Generating waveform and spectrogram examples...")
    example_intents = ["activate|lights|kitchen", "decrease|heat|bedroom", "change language|English|none"]
    selected_clips = []
    for ei in example_intents:
        act, obj, loc = ei.split("|")
        sample_row = all_df[
            (all_df["action"] == act)
            & (all_df["object"] == obj)
            & (all_df["location"] == loc)
        ].iloc[0]
        selected_clips.append(sample_row)

    fig, axes = plt.subplots(len(selected_clips), 2, figsize=(14, 3.2 * len(selected_clips)))
    fig.patch.set_facecolor(COLOR_PAPER)

    for i, row in enumerate(selected_clips):
        with wave.open(row["abs_path"], "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        time_axis = np.linspace(0, len(signal) / sr, num=len(signal))

        # Waveform
        ax_wave = axes[i, 0]
        ax_wave.set_facecolor(COLOR_PAPER)
        ax_wave.plot(time_axis, signal, color=COLOR_PINE, linewidth=0.8)
        ax_wave.set_title(f'Waveform: "{row["transcript"]}" ({row["action"]}/{row["object"]}/{row["location"]})')
        ax_wave.set_xlabel("Waktu (detik)")
        ax_wave.set_ylabel("Amplitudo")
        ax_wave.set_ylim(-1.05, 1.05)
        ax_wave.grid(True, linestyle="--", alpha=0.3, color=COLOR_MIST)

        # Spectrogram (STFT with standard window)
        ax_spec = axes[i, 1]
        ax_spec.set_facecolor(COLOR_PAPER)
        Pxx, freqs, bins, im = ax_spec.specgram(signal, NFFT=512, Fs=sr, noverlap=256, cmap="viridis")
        ax_spec.set_title("Spektrogram Frekuensi (STFT)")
        ax_spec.set_xlabel("Waktu (detik)")
        ax_spec.set_ylabel("Frekuensi (Hz)")

    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "eda_audio_spectrogram_examples.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 6. Original split overlap analysis chart (Phrase leakage proof)
    # -------------------------------------------------------------
    print("[02_eda] Generating split overlap comparison chart...")
    train_speakers = set(train_df["speakerId"])
    test_speakers = set(test_df["speakerId"])
    train_phrases = set(train_df["transcript"])

    test_total = len(test_df)
    test_phrases_in_train_count = int(test_df["transcript"].isin(train_phrases).sum())
    test_phrases_in_train_pct = (test_phrases_in_train_count / test_total) * 100.0

    speaker_overlap_pct = (len(train_speakers & test_speakers) / len(test_speakers)) * 100.0

    overlap_df = pd.DataFrame(
        [
            {
                "Entitas": "Speaker Overlap (Train vs Test)",
                "Persentase": speaker_overlap_pct,
                "Keterangan": "0 dari 10 speaker uji ada di data latih (Disjoint)",
            },
            {
                "Entitas": "Phrase Leakage (Test clips in Train)",
                "Persentase": test_phrases_in_train_pct,
                "Keterangan": "100% frasa kalimat uji sudah pernah didengar saat training!",
            },
        ]
    )
    overlap_df.to_csv(os.path.join(output_tab_dir, "eda_split_overlap_leakage.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor(COLOR_PAPER)
    ax.set_facecolor(COLOR_PAPER)
    bars = ax.bar(
        overlap_df["Entitas"],
        overlap_df["Persentase"],
        color=[COLOR_PINE, COLOR_BRICK],
        edgecolor=COLOR_INK,
        width=0.45,
    )
    ax.set_ylabel("Persentase (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Bukti Empiris 'Phrase Leakage' pada Split Asli (Split A)")
    ax.grid(axis="y", linestyle="--", alpha=0.4, color=COLOR_MIST)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "eda_split_overlap_comparison.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 7. Update configs/base.yaml with max_audio_seconds (Task P1-06)
    # -------------------------------------------------------------
    print(f"[02_eda] Updating {configs_base_path} with max_audio_seconds: {recommended_max_sec}...")
    with open(configs_base_path, "r", encoding="utf-8") as f:
        config_lines = f.readlines()

    updated_lines = []
    for line in config_lines:
        if line.strip().startswith("max_audio_seconds:"):
            updated_lines.append(f"  max_audio_seconds: {recommended_max_sec}\n")
        else:
            updated_lines.append(line)

    with open(configs_base_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    print("[02_eda] EDA selesai dengan sukses!")
    return duration_stats


def main():
    parser = argparse.ArgumentParser(description="Run complete EDA on FSC dataset.")
    parser.add_argument("--root", default="data/raw/fluent_speech_commands_dataset")
    parser.add_argument("--output-figures", default="reports/figures")
    parser.add_argument("--output-tables", default="reports/tables")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    run_eda(
        dataset_root=args.root,
        output_fig_dir=args.output_figures,
        output_tab_dir=args.output_tables,
        configs_base_path=args.config,
    )


if __name__ == "__main__":
    main()
