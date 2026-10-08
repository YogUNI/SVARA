"""Unit tests for Robustness and Fairness evaluation modules (Tasks P6-03, P6-04)."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from svara.eval.fairness import evaluate_demographic_fairness
from svara.eval.robustness import evaluate_model_at_snr, run_robustness_sweep
from svara.models.crnn_baseline import CRNNBaseline


class MockAudioDataset(Dataset):
    def __init__(self, size=8):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "waveform": torch.randn(16000),
            "lengths": torch.tensor(16000, dtype=torch.long),
            "action_id": torch.tensor(1, dtype=torch.long),
            "object_id": torch.tensor(2, dtype=torch.long),
            "location_id": torch.tensor(0, dtype=torch.long),
        }


def test_robustness_sweep():
    model = CRNNBaseline(
        n_action=6,
        n_object=14,
        n_location=4,
        conv_channels=16,
        rnn_hidden=32,
        rnn_layers=1,
    )
    loader = DataLoader(MockAudioDataset(size=4), batch_size=2)
    noise_clip = np.random.normal(0, 0.1, 16000).astype(np.float32)

    df_sweep = run_robustness_sweep(model, loader, noise_clip, snrs=(None, 20.0, 0.0))
    assert len(df_sweep) == 3
    assert "exact_match_acc" in df_sweep.columns
    assert "snr_db" in df_sweep.columns


def test_fairness_evaluation():
    with tempfile.TemporaryDirectory() as tmpdir:
        demo_csv = os.path.join(tmpdir, "speaker_demographics.csv")
        with open(demo_csv, "w", encoding="utf-8") as f:
            f.write("speakerId,gender,age_range,firstLanguage\n")
            f.write("spk_01,female,20-29,English\n")
            f.write("spk_02,male,30-39,English\n")

        preds_df = pd.DataFrame([
            {"speakerId": "spk_01", "correct_exact": True},
            {"speakerId": "spk_01", "correct_exact": True},
            {"speakerId": "spk_02", "correct_exact": False},
        ])

        fairness_res = evaluate_demographic_fairness(preds_df, demographics_csv_path=demo_csv)
        assert "gender" in fairness_res
        gender_df = fairness_res["gender"]
        assert "subgroup" in gender_df.columns
        assert "low_confidence_flag" in gender_df.columns
