# 02 — Model and Experiments

> Detailed technical spec, code sketch, gotchas and HuggingFace usage plan: **docs/12**. Extra optional experiments E11-E15 are defined there.

## 1. Main model: `Wav2VecSLU`
```
waveform (16 kHz mono, normalized) 
  -> wav2vec2 feature encoder (CNN, frozen)
  -> wav2vec2 transformer encoder (12 layers, base; configurable truncation)
  -> length-aware mean pooling over time (ignore padded frames)
  -> dropout
  -> head_action   Linear(768 -> n_action)     # 6 classes
  -> head_object   Linear(768 -> n_object)     # 14 classes
  -> head_location Linear(768 -> n_location)   # 4 classes
```
- Pretrained checkpoints (HuggingFace): `facebook/wav2vec2-base` (self-supervised, default) and
  `facebook/wav2vec2-base-960h` (ASR fine-tuned, variant). Verify names exist before use.
- Loss: sum of three cross-entropies (equal weights; try label smoothing 0.05 as a tunable).
- Joint-head variant: single `Linear(768 -> 31)` over `intent_id`; slots derived from the intent map.
- Prediction: argmax per head. Intent = tuple of slot predictions. Check the tuple is a **valid** intent in
  `intent_map.json`; log invalid-combination rate (a nice analysis: multi-head can output impossible intents).
  Optional decoding: constrained decoding to the best valid intent by summed log-probabilities. Compare both.

### wav2vec2 gotchas (must implement correctly)
- Input normalization: zero-mean, unit-variance per utterance (the feature extractor does this for base models).
- For `wav2vec2-base` style models (group-norm feature extractor) **do not pass `attention_mask`**; pad with zeros
  and prefer length-bucketed batches to limit padding. For length-aware pooling compute frame lengths with
  `model._get_feat_extract_output_lengths(lengths)` and build the pooling mask yourself.
- Keep `max_audio_seconds` from the EDA; crop or pad deterministically for eval, random crop only in training.
- Freeze the convolutional feature encoder (`freeze_feature_encoder()`).
- SpecAugment-style masking is built in via config (`mask_time_prob`, `mask_time_length`); enable only for training.
- Mixed precision (fp16/bf16) on GPU; gradient checkpointing only if memory-bound.

## 2. Baselines
- **B0 Majority/prior**: trivial sanity check.
- **B1 CRNN from scratch**: 64–80-bin log-mel (25 ms / 10 ms) -> 2–3 conv blocks -> 2-layer BiGRU -> mean pool ->
  same 3 heads. Shows what pretraining buys (RQ2). Keep it small and well-tuned, not a strawman.
- **B2 wav2vec2 frozen**: encoder fully frozen, train heads only (linear probe). Cheap, strong reference.

## 3. Initial hyperparameters (starting point, tune on validation)
| Item | Starting value |
|---|---|
| Optimizer | AdamW, weight decay 0.01 |
| LR | encoder 3e-5, heads 1e-3 (two param groups) |
| Schedule | 10% warm-up, linear decay |
| Batch | 16 (use grad accumulation to reach effective 32) |
| Epochs | up to 15, early stopping on validation exact-match (patience 3) |
| Precision | fp16 on Colab T4 |
| Dropout | 0.1 before heads |
| Seeds | 42, 43, 44 for final configs; single seed 42 while exploring |
Benchmark one epoch first on Colab to measure time per epoch, then plan the experiment budget (docs/04).
These are starting values only. Tune with the small, fair protocol in docs/11 §4 (validation of Split B1 only), then freeze `M0`.
Use gradient clipping (max-norm 1.0) by default.

## 4. Augmentation (training only)
- SpecAugment-style time masking (model-internal) and optional frequency-style masking for the CRNN.
- Additive noise from the **training noise categories** with random SNR in [0, 20] dB, probability 0.5.
- Speed perturbation {0.9, 1.0, 1.1} probability 0.3 (optional).
- Random gain ±6 dB. Optional simulated reverb.
All augmentation toggles live in YAML (`configs/aug_*.yaml`) and must be logged in the run config.

## 5. Experiment matrix
Run on **Split B** as the primary protocol (hardest, most informative), with Split A for reference when cheap.
One-factor-at-a-time from the main config `M0`.

| ID | Variation | Answers |
|---|---|---|
| M0 | wav2vec2-base, full fine-tune (CNN frozen), 3 heads, SpecAugment on, no noise aug | main model |
| E1 | CRNN from scratch (B1) | RQ2 |
| E2 | wav2vec2 frozen, heads only (B2) | RQ2 |
| E3 | M0 with joint 31-way head | RQ3 |
| E4 | M0 initialised from `wav2vec2-base-960h` | pretraining type |
| E5 | M0 truncated to top-8 / 6 / 4 transformer layers removed (keep first N layers) | RQ4 |
| E6 | M0 + noise augmentation | RQ5 |
| E7 | M0 without SpecAugment | augmentation value |
| E8 | M0 on Split A (original) | RQ1 (A vs B) |
| E9 | M0 on Split C | RQ6 (if time) |
| E10 | Best config + constrained decoding | valid-intent decoding |
Final candidates (best accuracy; best small model) get 3 seeds each, and Split B1–B3 draws.

Stretch experiments (only after everything else is done):
- XLS-R (`facebook/wav2vec2-xls-r-300m`) for cross-lingual transfer; heavy, likely needs fp16 + small batch.
- Few-shot adaptation: fine-tune on a small subset of own recordings, test on held-out own speakers.
- Distilled/small encoders (e.g., DistilHuBERT) as an alternative compression route.

## 6. Layer truncation (RQ4)
Keep the first N transformer layers (N ∈ {12, 8, 6, 4}) and fine-tune. Record accuracy, parameters, latency, size.
Expected pattern is a trade-off curve; **report whatever is measured**. Plot accuracy vs latency and vs size.

## 7. Confidence and rejection
Implement `src/svara/eval/reject.py`: confidence scores (min-head, product, joint-head probability), threshold
selection on validation for a target selective accuracy (e.g., ≥99% on accepted validation samples),
risk–coverage curve, FAR on OOD set. Optional temperature scaling on validation to calibrate; report ECE.

## 8. Training outputs (per run, `reports/runs/<run_id>/`)
- `config.yaml` (resolved), `git_commit.txt`, `env.txt` (library versions, GPU)
- `train_log.csv` (epoch, train_loss, val_loss, val exact/slot acc, lr, time)
- `best.ckpt` (gitignored) + `last.ckpt` for resume
- `metrics.json` (valid + test: all metrics in docs/01), `predictions.csv` (path, true slots, pred slots, confidences)
- `confusion_*.csv`, figures auto-generated by `scripts/make_report_assets.py`

## 9. Resumability (Colab)
Trainer saves optimizer, scheduler, scaler, epoch, RNG state each epoch to Drive. `--resume auto` picks the last
checkpoint of that run id. Data loading must not depend on local ephemeral paths other than a cached copy of the
dataset on `/content` (copy from Drive at start for speed).

## 10. Tests (`tests/`)
- Split tests: no leakage per protocol; every intent present in train.
- Model smoke test: forward pass on random audio returns three logit tensors of right shapes.
- Pooling mask test: padded input gives same output as unpadded (within tolerance).
- Train/serve parity: preprocessing function gives identical tensors from the same wav in both paths.
- ONNX parity: ONNX outputs match PyTorch within tolerance on 20 clips; int8 reports its accuracy delta.
