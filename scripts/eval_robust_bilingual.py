"""Evaluate Trained Robust Bilingual SLU Model on Unseen Sets.

Computes exact-match and per-slot accuracies on:
1. data/bilingual_dataset/splits/test_indonesian.csv (expanded 341 unseen casual Indonesian clips)
2. data/bilingual_dataset/splits/test_english.csv (3,793 FSC native English clips)
"""

import os
import json
import torch
from torch.utils.data import DataLoader

from svara.data.collate import FSCDataset, collate_fn_pad
from svara.models.wav2vec_slu import Wav2VecSLU
from svara.train.losses import compute_batch_accuracies

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for Evaluation: {device}")

    with open("configs/intent_map.json", "r", encoding="utf-8") as f:
        intent_map = json.load(f)
    slot_vocab = intent_map["slot_vocab"]

    id_test_csv = "data/bilingual_dataset/splits/test_indonesian.csv"
    en_test_csv = "data/bilingual_dataset/splits/test_english.csv"

    id_test_dataset = FSCDataset(id_test_csv)
    en_test_dataset = FSCDataset(en_test_csv)

    id_test_loader = DataLoader(id_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_pad)
    en_test_loader = DataLoader(en_test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_pad)

    ckpt_path = "reports/runs/robust_bilingual_w2v2/best.ckpt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = Wav2VecSLU(
        pretrained_model_name_or_path="facebook/wav2vec2-base",
        n_action=len(slot_vocab["action"]),
        n_object=len(slot_vocab["object"]),
        n_location=len(slot_vocab["location"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def eval_loader(loader):
        hits, act_hits, obj_hits, loc_hits, total = 0, 0, 0, 0, 0
        with torch.no_grad():
            for b in loader:
                w = b["waveform"].to(device)
                l = b["lengths"].to(device)
                t = {
                    "action_id": b["action_id"].to(device),
                    "object_id": b["object_id"].to(device),
                    "location_id": b["location_id"].to(device),
                }
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    logits = model(w, l)
                accs = compute_batch_accuracies(logits, t)
                hits += int(round(accs["acc_exact_match"] * len(l)))
                act_hits += int(round(accs["acc_action"] * len(l)))
                obj_hits += int(round(accs["acc_object"] * len(l)))
                loc_hits += int(round(accs["acc_location"] * len(l)))
                total += len(l)
        return {
            "exact_match": hits / total,
            "action_acc": act_hits / total,
            "object_acc": obj_hits / total,
            "location_acc": loc_hits / total,
            "total_samples": total,
        }

    print("\nMengevaluasi Test Set Bahasa Indonesia Baru (Unseen Utterance)...")
    id_results = eval_loader(id_test_loader)
    print(f"- Exact-Match: {id_results['exact_match']*100:.2f}%")
    print(f"- Action:      {id_results['action_acc']*100:.2f}%")
    print(f"- Object:      {id_results['object_acc']*100:.2f}%")
    print(f"- Location:    {id_results['location_acc']*100:.2f}%")

    print("\nMengevaluasi Test Set Bahasa Inggris Native FSC...")
    en_results = eval_loader(en_test_loader)
    print(f"- Exact-Match: {en_results['exact_match']*100:.2f}%")
    print(f"- Action:      {en_results['action_acc']*100:.2f}%")
    print(f"- Object:      {en_results['object_acc']*100:.2f}%")
    print(f"- Location:    {en_results['location_acc']*100:.2f}%")

    metrics = {
        "indonesian": id_results,
        "english": en_results,
    }
    with open("reports/runs/robust_bilingual_w2v2/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics tersimpan di: reports/runs/robust_bilingual_w2v2/test_metrics.json")

if __name__ == "__main__":
    evaluate()
