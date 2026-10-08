"""Robustness evaluation module: tests SLU models under varying SNR noise conditions.

Simulates acoustic noise at SNR {clean, 20dB, 10dB, 5dB, 0dB}
Outputs accuracy vs SNR curves and CSV summary for Chapter 6.

Task: P6-03
Reference: docs/01 §6, docs/10 P6-03
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from svara.data.augment import mix_noise_at_snr
from svara.eval.metrics import compute_exact_match


def evaluate_model_at_snr(
    model: torch.nn.Module,
    loader: DataLoader,
    noise_clip: np.ndarray,
    snr_db: Optional[float],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate accuracy on a dataloader with additive noise injected at specified SNR."""
    model.eval()
    correct_exact = []
    correct_act = []
    correct_obj = []
    correct_loc = []

    with torch.no_grad():
        for batch in loader:
            wavs = batch["waveform"].numpy()
            lens = batch["lengths"].to(device)

            # Inject noise if snr_db is provided (None means clean)
            if snr_db is not None:
                noisy_wavs = np.zeros_like(wavs)
                for i in range(len(wavs)):
                    noisy_wavs[i] = mix_noise_at_snr(wavs[i], noise_clip, snr_db)
                wav_tensor = torch.from_numpy(noisy_wavs).to(device)
            else:
                wav_tensor = batch["waveform"].to(device)

            logits = model(wav_tensor, lengths=lens)

            pred_act = torch.argmax(logits["action"], dim=-1).cpu().numpy()
            pred_obj = torch.argmax(logits["object"], dim=-1).cpu().numpy()
            pred_loc = torch.argmax(logits["location"], dim=-1).cpu().numpy()

            true_act = batch["action_id"].numpy()
            true_obj = batch["object_id"].numpy()
            true_loc = batch["location_id"].numpy()

            act_m = pred_act == true_act
            obj_m = pred_obj == true_obj
            loc_m = pred_loc == true_loc
            exact_m = act_m & obj_m & loc_m

            correct_exact.extend(exact_m)
            correct_act.extend(act_m)
            correct_obj.extend(obj_m)
            correct_loc.extend(loc_m)

    return {
        "snr_db": "clean" if snr_db is None else f"{snr_db}dB",
        "exact_match_acc": float(np.mean(correct_exact)),
        "action_acc": float(np.mean(correct_act)),
        "object_acc": float(np.mean(correct_obj)),
        "location_acc": float(np.mean(correct_loc)),
        "num_samples": len(correct_exact),
    }


def run_robustness_sweep(
    model: torch.nn.Module,
    loader: DataLoader,
    noise_clip: np.ndarray,
    snrs: Tuple[Optional[float], ...] = (None, 20.0, 10.0, 5.0, 0.0),
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Run full SNR robustness evaluation sweep."""
    dev = device or torch.device("cpu")
    rows = []
    for snr in snrs:
        res = evaluate_model_at_snr(model, loader, noise_clip, snr, dev)
        rows.append(res)
    return pd.DataFrame(rows)
