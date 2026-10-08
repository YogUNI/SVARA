# 01 — Dataset and Evaluation

## 1. Dataset facts (from the dataset README; verify with code, do not trust blindly)
- 97 speakers, 248 distinct phrases mapped to 31 unique intents, 3 slots: action / object / location.
- Collected via crowdsourcing; US and Canada only; each participant said each phrase twice; clips judged
  noisy, inaudible, unintelligible or wrong-phrase by validators were removed.
- Audio: 16 kHz, single channel `.wav` in `wavs/speakers/<speakerId>/`.
- Files: `data/train_data.csv`, `data/valid_data.csv`, `data/test_data.csv`, `data/speaker_demographics.csv`.
- CSV columns (per README): `path, speakerId, transcript, action, object, location`.
- Label sets (per README):
  - action: change language, activate, deactivate, increase, decrease, bring
  - object: none, music, lights, volume, heat, lamp, newspaper, juice, socks, shoes, Chinese, Korean, English, German
  - location: none, kitchen, bedroom, washroom
- Demographics: self-reported speaking ability, first language, current language for work/school, gender, age range.
- Paper split sizes: train 23,132 / valid 3,118 / test 3,793 (sum 30,043).
- License: "Fluent Speech Commands Public License" (PDF in dataset folder). Read it. Cite it in the report.
  Do not publish the audio.

## 2. Data audit script: `scripts/00_check_data.py`
Must verify and write `reports/data_audit.json`:
1. Dataset root exists; all CSVs present; print real header of each CSV.
2. Row counts per split; compare with the paper sizes above and flag differences (do not fail hard).
3. Every `path` exists on disk; list missing files.
4. Audio sanity on a sample (or all if fast): sample rate == 16000, channels == 1, duration stats
   (min / mean / p95 / max), clipping, silence ratio.
5. Label sets observed vs README label sets.
6. Unique intents = unique (action, object, location) tuples. Expect 31. Write `configs/intent_map.json`:
   `{ "intent_id": {"action":..., "object":..., "location":...}, ... }` plus `slot_vocab` for each slot.
7. Overlap analysis between original train/valid/test: speakers shared (count), transcripts shared (count
   and %), and the fraction of test utterances whose transcript appears in train. This empirically
   reproduces the "phrase leakage" finding for the report.
8. Demographic join: every speakerId in CSV has a demographics row.
9. Label consistency: each `transcript` maps to exactly one (action, object, location); list conflicts (docs/11 V-01).
10. Duplicate audio detection by content hash across splits; wav-folder speaker equals `speakerId` (V-03, V-04).
11. Phrasings per intent in every split (V-02); class imbalance ratios per slot and intent (V-07).

## 3. EDA outputs (`scripts/02_eda.py`, viewed in `notebooks/01_eda.ipynb`)
Save figures to `reports/figures/eda_*.png` (300 dpi) and tables to `reports/tables/eda_*.csv`:
- Class distribution: per slot and per intent (log-scale if needed).
- Clips per speaker; phrases per intent; duration histogram (+ percentile lines).
- Demographic distributions (gender, age range, first language, speaking ability).
- Waveform + log-mel examples for a few intents.
- Original-split overlap bar chart (speaker overlap vs transcript overlap).
- Choose `max_audio_seconds` from the duration p99 (rounded up); record it in `configs/base.yaml`.

## 4. Split protocols (`scripts/01_make_splits.py`, config `configs/splits.yaml`)
All splits are saved as CSVs in `data/processed/splits/<name>/{train,valid,test}.csv` with the same columns
as the original plus `intent_id`. A `split_manifest.json` records seed, rule, counts and overlap checks.

### Split A — Original (baseline, comparable to the literature)
Use the provided train/valid/test CSVs unchanged.

### Split B — Unseen-utterance (primary honest protocol)
- Group by `transcript`. For each intent, hold out a fraction (default ~20%) of its distinct phrasings for test
  and a smaller fraction (~10%) for valid; the rest go to train.
- Constraints: every intent keeps at least 2 training phrasings (if impossible, reduce held-out for that intent
  and log it). Speakers may appear in all partitions.
- Assertion in code: `set(test.transcript) ∩ set(train.transcript) == ∅`, same for valid.
- Because the sentence set is small (248), hold-out choice affects variance: generate **3 hold-out draws**
  (`B1, B2, B3`) with different seeds and report mean ± std for the main model.

### Split C — Speaker-grouped, demographically stratified
- Group by `speakerId`; allocate ~70/15/15 speakers while trying to balance gender, age range and first language
  across partitions (greedy stratification is fine).
- First check (from the audit) whether the original split is already speaker-disjoint. If it is, Split C is a
  re-sampled, better-balanced version and has **lower priority**; if time is short, drop C (see docs/04 cut list).
- Assertion: no speaker in more than one partition.

### Never do
Random file-level splits (leaks duplicate recordings of the same phrase by the same speaker).

## 5. Metrics (`src/svara/eval/metrics.py`)
Primary:
- **Exact-match intent accuracy**: action, object and location all correct.
Secondary:
- Per-slot accuracy (action / object / location).
- Macro-F1 over the 31 intents; per-intent precision/recall.
- Confusion matrices: per slot, and 31×31 joint (normalized rows).
Statistics:
- Mean ± std over seeds (default 3: 42, 43, 44).
- 95% bootstrap confidence interval on test accuracy (1,000 resamples) per run.
- For model comparisons on the same test set, use McNemar's test (or paired bootstrap) and report p-values.
Systems metrics (`src/svara/export/benchmark.py`):
- Model size on disk (MB), parameter count.
- CPU latency p50/p95 for a fixed-length input (warm-up excluded), batch size 1, thread count recorded.
- Real-time factor = latency / audio duration.
- Peak RAM during inference.

## 6. Robustness evaluation (`src/svara/eval/robustness.py`)
- Noise source: ESC-50 environmental sounds (or own household recordings). **Partition the noise categories**:
  categories used for training augmentation must differ from categories used for test noise. Document the lists.
- Mix noise into clean test clips at SNR in {20, 10, 5, 0} dB (+ clean). Plot accuracy vs SNR for each model.
- Optional reverb: convolve with simulated or recorded room impulse responses; report separately.
- Output: `reports/tables/robustness_<run>.csv`, `reports/figures/robustness_snr.png`.

## 7. Fairness analysis (`src/svara/eval/fairness.py`)
Join test predictions with `speaker_demographics.csv`. Report exact-match accuracy per group for gender, age
range, first language, speaking ability (merge tiny groups, mark groups with n < 30 as low-confidence).
Add bootstrap CIs. Interpret cautiously; do not over-claim causal explanations.

## 8. Own-recording set ("ID-accent English set")
Purpose: measure domain shift to Indonesian-accented English and real rooms. **Evaluation only** (never in training,
except an optional clearly-labelled adaptation experiment).
Protocol:
- Target ≥ 10 speakers (more is better), anonymised IDs (`own_s01`...). Written consent that audio is used only for
  this course project and is not published.
- Each speaker reads a fixed list: one phrasing for each of the 31 intents (take phrasings from the FSC
  transcripts), in two conditions: **quiet room** and **with background noise** (TV/fan/kitchen sounds), if possible
  using a phone and a laptop mic.
- Record 16 kHz mono WAV (convert if needed). File naming: `own_s01_q_<intent_id>.wav` (`q`=quiet, `n`=noisy).
- Metadata CSV `data/own_recordings/metadata.csv`: `path, speakerId, device, condition, transcript, action, object,
  location, intent_id, gender(optional), age_range(optional)`.
- Report exact-match accuracy overall, quiet vs noisy, per device, per speaker (anonymised). Report n explicitly:
  this is a small field test, not a benchmark.

## 9. Out-of-domain / rejection evaluation
Build an OOD set: (a) non-command speech (own recordings of random sentences, 100+ clips), (b) silence and
room noise, (c) ESC-50 clips. Evaluate false-accept rate at several confidence thresholds.
Confidence score: minimum over the three heads of the max softmax probability (also compare product and
joint-head probability). Choose threshold on **validation** only. Plot risk–coverage and report
accuracy-at-coverage on test + FAR on the OOD set.

## 10. Stretch: SLURP stress test
SLURP (≈58 h, 18 scenarios, 46 actions, 56 entities, 177 speakers) is much harder and heavier. If time allows,
evaluate only a smart-home-like scenario subset zero-shot as a qualitative stress test; do not make it a
dependency. Check its license before use.
