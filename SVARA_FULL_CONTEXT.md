# SVARA — FULL CONTEXT (auto-concatenated from AGENTS.md and docs/00-11)

> Generated file. Edit the source files, not this one.


<!-- ===== BEGIN AGENTS.md ===== -->

# AGENTS.md — SVARA (Smart Voice Assistant for Residential Automation)

## What this project is
University Deep Learning final project (Universitas Mercu Buana, Informatics Engineering), team of 2.
Goal: train, evaluate and deploy an **end-to-end Spoken Language Understanding (SLU)** model.
Pipeline: speech audio -> predicted slots (`action`, `object`, `location`) -> simulated smart-home action.
The **model is the graded core**. The web app is a demo wrapper around a mature model.

Deliverables: (1) trained model + reproducible code, (2) report in **Indonesian** following the course
template (8 chapters + appendices, IEEE references), (3) demo video, (4) GitHub repo.

Assigned by lecturer: domain Speech Recognition, task "Smart Home Voice Control", method
"Transformer SLU, wav2vec2", tools Python + HuggingFace, dataset **Fluent Speech Commands (FSC)**,
main paper Lugosch et al., 2019.

## Team and ownership
- **Yoga (Y)**: data pipeline, model, training, evaluation, export, backend, web build, technical report chapters (4-6).
- **Haikal (H)**: own-recording dataset and household-noise clips, QA of the web demo, report chapters 1-3 and references (IEEE), demo video, field-test sessions.
- Collaboration is through the GitHub repo (see docs/09). Work in feature branches, open PRs, the other person reviews.
- Task IDs in docs/10 (e.g. `P3-07`) are used in branch names, commits and PR titles. Check the owner before editing someone's area.
- Haikal is less experienced with ML code: write clear READMEs/comments, keep commands copy-pasteable, and never assume he can debug a stack trace.

## Read these docs before working (they are the source of truth)
| File | Read when |
|---|---|
| docs/00_PROJECT_BRIEF.md | always, first |
| docs/01_DATASET_AND_EVALUATION.md | touching data, splits, metrics |
| docs/02_MODEL_AND_EXPERIMENTS.md | touching model, training, ablations |
| docs/03_TECH_STACK_AND_REPO.md | creating files, choosing libs, folder layout |
| docs/04_ROADMAP.md | planning, deciding what to do next |
| docs/05_BACKEND_AND_DEPLOYMENT.md | export, API, Docker, hosting |
| docs/06_UI_UX_SPEC.md | any frontend work |
| docs/07_REPORT_MAPPING.md | producing figures/tables for the report |
| docs/08_ANTIGRAVITY_PROMPTS.md | ready-made prompts per phase |
| docs/09_GITHUB_COLLABORATION.md | branches, PRs, ownership, what goes in git |
| docs/10_DETAILED_TASKS.md | the full task list with IDs, owners, acceptance criteria |
| docs/11_DL_VERIFICATION_CHECKLIST.md | sanity checks, test-set discipline, tuning protocol; read before any training or reporting |

## Non-negotiable rules
1. **Never fabricate results.** No invented accuracy, latency, loss curves or table values. Every number
   shown anywhere (report assets, web UI) must come from a logged run in `reports/`. If a result does not
   exist yet, show an explicit empty state or `TODO`, never a plausible-looking placeholder.
2. **Never commit data.** `data/`, raw audio, own recordings and model checkpoints are gitignored. FSC has its
   own license (PDF inside the dataset folder): do not redistribute audio publicly.
3. **Do not hardcode dataset facts.** Read column names and label sets from the CSV headers and generate
   `configs/intent_map.json` from data. The dataset README says CSV has `path, speakerId, transcript,
   action, object, location`, but some libraries expect `transcription`: always inspect the real header.
4. **Splits are grouped, never random per file.** Each speaker recorded each phrase twice; random file-level
   splitting leaks. Group by `speakerId` and/or `transcript` as specified in docs/01.
5. **Reproducibility.** Fixed seeds (default 42, 43, 44), config-driven (YAML), every run writes
   `config.yaml`, `metrics.json`, `train_log.csv`, git commit hash into `reports/runs/<run_id>/`.
6. **Logic lives in `.py`, not notebooks.** Notebooks are thin wrappers (EDA, Colab runner, results viewer).
7. **Keep train/serve preprocessing identical.** Same normalization, sample rate (16 kHz mono), max length.
   Share one function in `src/svara/data/audio.py`.
8. **Ask before** adding a heavy dependency, changing the model family, changing split definitions,
   or deleting files in `reports/`.
9. **Language:** code, comments, commit messages, docs in English. Report text and user-facing
   UI copy for the demo in Indonesian unless docs/06 says otherwise.
10. **Test-set discipline** (docs/11 §1): choose hyperparameters, variants, thresholds and the deployment model on validation only; never edit a config after seeing test numbers; log every test evaluation.
11. Small, reviewable steps. After each task: run tests/lint, summarise what changed, list what is NOT done.

## Stack (details in docs/03)
- Python 3.10/3.11, PyTorch, torchaudio, HuggingFace `transformers`, pandas, scikit-learn, matplotlib.
- Training on Google Colab GPU (user has no local GPU assumed): the repo is cloned in Colab, data and
  checkpoints live on Google Drive, training scripts must be **resumable**.
- Export: ONNX + onnxruntime (int8 dynamic quantization evaluated, not assumed).
- Backend: FastAPI + onnxruntime. Frontend: React + Vite + TypeScript. One Docker image serves both.

## Domain glossary
- **Intent** = the unique combination (action, object, location). FSC has 31 unique intents, 248 phrasings.
- **Exact-match accuracy** = all three slots correct. This is the headline metric.
- **Unseen-utterance split** = test phrases never appear in training (hard, honest generalization test).
- **Reject / abstain** = model refuses to act when confidence is below a threshold chosen on validation.
- **Own set** = our recordings of FSC-style English commands by Indonesian-accented speakers (eval only).

## Commands (once scaffolded; keep this section accurate)
```
pip install -r requirements.txt
python scripts/00_check_data.py --root data/raw/fluent_speech_commands_dataset
python scripts/01_make_splits.py --config configs/splits.yaml
python scripts/sanity_checks.py --config configs/model_w2v2_3head.yaml --split B1   # docs/11 V-10..V-20
python scripts/train.py --config configs/model_w2v2_3head.yaml --split A --seed 42
python scripts/evaluate.py --run reports/runs/<run_id> --split A
python scripts/export_onnx.py --run reports/runs/<run_id> --quantize int8
uvicorn svara.serve.app:app --reload          # backend
cd web && npm install && npm run dev               # frontend
pytest -q
```

## Definition of done (any task)
- Runs end-to-end from a clean checkout following the commands above.
- Has a minimal test or a smoke-run command documented.
- Writes artifacts to the paths in docs/03 and docs/07 so the report can consume them.
- No secrets, no data, no checkpoints in git.
- AGENTS.md "Commands" section updated if commands changed.

## Working style
Be direct about trade-offs. If a requirement in docs conflicts with reality (library behavior, Colab limits,
data surprises), stop and report it with evidence instead of silently working around it.
Prefer boring, well-tested solutions over clever ones. Cut scope in the order listed in docs/04.

<!-- ===== END AGENTS.md ===== -->

<!-- ===== BEGIN docs/00_PROJECT_BRIEF.md ===== -->

# 00 — Project Brief

## 1. One-paragraph summary
We build **SVARA** (Smart Voice Assistant for Residential Automation): an end-to-end SLU model that maps a short spoken English command directly to a
structured intent (`action`, `object`, `location`) without producing a transcript first, then drives a
simulated smart home (lights, lamp, heat, music/volume in kitchen / bedroom / washroom). We fine-tune a
pretrained **wav2vec 2.0** encoder with classification heads on **Fluent Speech Commands (FSC)**, evaluate it
far more strictly than the standard benchmark, test it on our own Indonesian-accented recordings and noisy
conditions, compress it for CPU inference, and ship it as a web demo.

## 2. Course constraints (from the lecturer's spreadsheet, row 12)
| Field | Value |
|---|---|
| Domain / Task | Speech Recognition / Smart Home Voice Control |
| Problem statement | Voice-based home control |
| Method | Transformer SLU, wav2vec2 |
| Tools | Python, HuggingFace |
| Dataset | Fluent Speech Commands (lecturer supplied a GitHub mirror) |
| Main paper | Lugosch et al., 2019 (Speech Model Pre-training for End-to-End SLU) |
| Hard requirement | The **model must be mature first**; website/mobile integration is free choice |
| Report | Template with 8 chapters + appendices (see docs/07), IEEE references |

## 3. Why the naive version is not enough (critical context)
- FSC is **saturated**: public models reach ~99% on the original test split. Matching that is a baseline,
  not an achievement.
- In the original split, test *sentences* also appear in training, so a model can memorise phrases. Published
  re-evaluation shows accuracy dropping substantially (roughly 78–88% in their setup) when test phrases are unseen.
- Audio is clean (noisy/unintelligible clips were removed), speakers are from US/Canada, language is English only.
- Not everything in FSC is "smart home": `bring` (newspaper, juice, socks, shoes) and `change language`
  (Chinese, Korean, English, German) are assistant/robot tasks. Device-like objects: lights, lamp, heat, music, volume.
- Our lecturer's mirror is third-party. Verify integrity against the paper's split sizes (train 23,132 /
  valid 3,118 / test 3,793 = 30,043 total).

## 4. Our value-add (what makes this project stand out)
1. **Honest evaluation protocols**: original split (A), unseen-utterance split (B), speaker-grouped split (C).
2. **Multi-head vs joint-head** formulation and a structured ablation study.
3. **Robustness**: household noise at several SNRs, with noise categories held out between train and test.
4. **Fairness analysis** using `speaker_demographics.csv` (first language, gender, age range, self-reported ability).
5. **Own-recording field test**: Indonesian-accented speakers reading FSC-style English commands (domain shift).
6. **Safe actuation**: confidence-based rejection + out-of-domain false-accept test (risk–coverage curve).
7. **Edge-minded deployment**: layer truncation + ONNX + int8, with measured accuracy/latency/size trade-offs.
8. **Working web demo** with a floor-plan smart-home simulator.

## 5. Research questions (each maps to an experiment and a report section)
- RQ1: How much of FSC accuracy is phrase memorisation? (A vs B)
- RQ2: What does pretraining buy us? (CRNN from scratch vs wav2vec2 frozen vs fine-tuned)
- RQ3: Do three slot heads beat one joint 31-way head, especially on unseen phrases?
- RQ4: How much of the encoder do we actually need? (12 vs 8 vs 6 transformer layers)
- RQ5: How does accuracy degrade with noise, and does noise augmentation fix it?
- RQ6: Does performance differ across speaker groups (L1, gender, age) and on Indonesian-accented speech?
- RQ7: Can confidence thresholding stop the system from acting on speech it does not understand?

## 6. Scope
**In scope (must):** FSC pipeline, splits A and B, wav2vec2 3-head model, CRNN baseline, core ablations,
confusion matrices, noise test, export + latency benchmark, FastAPI + web demo, report, video.
**Should:** split C, fairness analysis, own-recording set (target ≥10 speakers), reject mechanism, int8 quantization.
**Could (stretch):** SLURP subset stress test, Indonesian-language commands with a multilingual encoder (XLS-R),
on-device Raspberry Pi test, mobile PWA.
**Out of scope:** wake-word detection, multi-turn dialogue, real IoT hardware control (simulated only unless
time remains), training an encoder from scratch.

## 7. Success criteria
- Reproducible repo: one command per stage, results regenerate from configs.
- Split A result is in the same ballpark as literature (sanity check, not the goal). Splits B/C and
  robustness tables are complete and honestly reported, including failures.
- Demo runs on CPU with end-to-end latency we have measured and report.
- Report chapters filled only with evidence from `reports/`.

## 8. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Colab session limits / disconnects | Resumable training, checkpoint every epoch to Drive, small batches |
| Compute budget too small for many runs | Run ablations on a fixed protocol (B), 1 seed first, add seeds if time remains |
| Mirror dataset differs from official | `00_check_data.py` verifies counts, header, missing wavs |
| Own recordings: low quantity / consent issues | Start recording early, written consent, anonymised IDs, never public |
| Quantization hurts accuracy | Evaluate int8 vs fp32 explicitly; fall back to fp32 or truncated model |
| Web scope creep | Web is phase 8; UI spec is fixed; model gets priority |
| Overclaiming | AGENTS.md rule 1: no fabricated numbers; report limitations |

## 9. Team
Two people, agreed split (details in docs/09 and docs/10):
- **Yoga**: data pipeline, model, training, evaluation, export, backend, web build, report chapters 4-6.
- **Haikal**: own-recording set and household-noise clips, QA/testing of the demo, report chapters 1-3 and references, video, field-test sessions.
Both must be able to explain the whole pipeline (data -> training -> model -> evaluation -> demo) at presentation time.

<!-- ===== END docs/00_PROJECT_BRIEF.md ===== -->

<!-- ===== BEGIN docs/01_DATASET_AND_EVALUATION.md ===== -->

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

<!-- ===== END docs/01_DATASET_AND_EVALUATION.md ===== -->

<!-- ===== BEGIN docs/02_MODEL_AND_EXPERIMENTS.md ===== -->

# 02 — Model and Experiments

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

<!-- ===== END docs/02_MODEL_AND_EXPERIMENTS.md ===== -->

<!-- ===== BEGIN docs/03_TECH_STACK_AND_REPO.md ===== -->

# 03 — Tech Stack and Repository

## 1. Decision: Python scripts (.py) as the core, notebooks (.ipynb) as thin wrappers
Reasons: agentic IDEs edit, diff, test and refactor `.py` files reliably; `.ipynb` JSON is brittle for diffs and
hard to test; training must be resumable and runnable headless on Colab; the report needs reproducible commands.

| Use `.py` (src/ + scripts/) | Use `.ipynb` (notebooks/) |
|---|---|
| Data loading, splits, augmentation | `01_eda.ipynb`: look at plots from EDA outputs |
| Models, training loop, evaluation | `90_colab_runner.ipynb`: mount Drive, clone repo, pip install, call scripts |
| Export, benchmark, serving | `91_results_explorer.ipynb`: load `reports/` and browse metrics |
| Report asset generation | demos/screenshots for the video |

Rule: a notebook may import from `svara` and call scripts, but must not contain unique logic.

## 2. Libraries
| Area | Choice | Notes |
|---|---|---|
| Language | Python 3.10 or 3.11 | match Colab |
| DL | PyTorch, torchaudio | |
| Models | HuggingFace `transformers` (Wav2Vec2Model) | no `datasets` required; local CSV loader |
| Audio | soundfile, torchaudio, librosa (EDA only) | |
| Data/stats | pandas, numpy, scikit-learn, scipy, statsmodels (McNemar) | |
| Plots | matplotlib (+ seaborn optional) | 300 dpi PNG for report |
| Config | PyYAML + argparse (or OmegaConf) | keep simple, avoid heavy frameworks |
| Logging | CSV + TensorBoard | training log is a report appendix |
| Export | onnx, onnxruntime, onnxruntime tools (quantization) | |
| Serving | FastAPI, uvicorn, python-multipart, pydantic | |
| Frontend | React + Vite + TypeScript, Recharts, CSS variables | see docs/06 |
| Tests | pytest, ruff (lint/format) | |
| Packaging | Docker (single image) | docs/05 |
Pin versions in `requirements.txt` after the first working run. Keep `requirements-serve.txt` minimal (no torch if
the server only needs onnxruntime + numpy + soundfile).

## 3. Compute plan
- Training: Google Colab (GPU, likely T4). Drive layout: `MyDrive/svara/{data,runs,models}`.
- Copy dataset from Drive to `/content/data` at session start (much faster IO).
- Local machine: CPU is enough for serving, EDA, tests, and the frontend.
- Always benchmark one epoch before launching a batch of experiments; write measured numbers to `docs/04` notes.

## 4. Repository layout
```
svara/
├── AGENTS.md
├── README.md
├── .github/                       # CODEOWNERS, PR template, issue templates
├── requirements.txt / requirements-serve.txt / requirements.lock / pyproject.toml
├── Dockerfile  .dockerignore  .gitignore
├── configs/
│   ├── base.yaml                 # paths, sample rate, max_audio_seconds, seeds
│   ├── splits.yaml
│   ├── model_w2v2_3head.yaml     # M0
│   ├── model_w2v2_joint.yaml     # E3
│   ├── model_w2v2_frozen.yaml    # E2
│   ├── model_crnn.yaml           # E1
│   ├── aug_none.yaml  aug_specaug.yaml  aug_noise.yaml
│   ├── noise_split.yaml          # which noise categories are train-noise vs test-noise
│   ├── serve.yaml                # confidence threshold etc. (chosen on validation)
│   ├── device_map.yaml           # intent -> simulated device effect (docs/05)
│   └── intent_map.json           # GENERATED from data
├── data/                          # gitignored
│   ├── raw/fluent_speech_commands_dataset/
│   ├── processed/splits/{A,B1,B2,B3,C}/
│   ├── own_recordings/{wavs_raw/, wavs_16k/, metadata.csv}   # consent/ stays on Drive, never in git
│   ├── noise/esc50/
│   └── noise_own/                 # our household-noise clips + categories.csv
├── src/svara/
│   ├── data/   fsc.py splits.py audio.py augment.py collate.py own_set.py
│   ├── models/ wav2vec_slu.py crnn_baseline.py heads.py decode.py
│   ├── train/  trainer.py losses.py schedulers.py
│   ├── eval/   metrics.py evaluate.py robustness.py fairness.py reject.py plots.py
│   ├── export/ to_onnx.py quantize.py benchmark.py
│   └── serve/  app.py inference.py device_state.py schemas.py
├── scripts/
│   ├── 00_check_data.py  01_make_splits.py  02_eda.py
│   ├── sanity_checks.py  lr_range_test.py  hparam_search.py       # docs/11 V-10..V-20, tuning protocol
│   ├── train.py  evaluate.py  run_ablation.py
│   ├── eval_robustness.py  eval_fairness.py  eval_reject.py  eval_own_set.py
│   ├── export_onnx.py  benchmark.py  make_report_assets.py
├── notebooks/  01_eda.ipynb  90_colab_runner.ipynb  91_results_explorer.ipynb
├── web/                           # React app (docs/06)
├── reports/
│   ├── NOTES.md  CLAIMS_AUDIT.md  RESULTS_INDEX.md   # running notes, claim→evidence table, figure/table index
│   ├── notes/                     # eda_captions.md, listening_check.md ...
│   ├── runs/<run_id>/             # per-run artifacts (docs/02 §8)
│   ├── metrics/                   # aggregated JSON consumed by the web "Lab" page
│   ├── figures/  tables/  logs/
├── models/                        # exported .onnx (small ones via release/LFS), checkpoints gitignored
├── tests/
└── docs/                          # these files + DECISIONS.md
    ├── notes/                     # Haikal's notes: license_summary, project_in_my_words, training_explained, literature_notes, recording_guide, qa_plan
    └── report/                    # report chapters as Markdown (bab1.md ... bab8.md), references.md, img/
```

## 5. Conventions
- Naming: `snake_case.py`, run ids `YYYYMMDD-HHMM_<config>_<split>_s<seed>`.
- Type hints and docstrings on public functions; `ruff` clean; `pytest -q` green before declaring a task done.
- One config file fully determines a run; CLI flags only override paths/seed/split/resume.
- Git: feature branches per phase, conventional commit messages (`feat:`, `fix:`, `docs:`, `exp:`).
- `.gitignore` must include: `data/`, `models/*.ckpt`, `reports/runs/**/*.ckpt`, `node_modules/`, `.env`, `*.wav`.

## 6. Aggregation contract for the web "Lab" page
`scripts/make_report_assets.py` writes one file `reports/metrics/summary.json` (schema versioned) containing:
protocol results (A/B1-3/C), ablation table, per-slot accuracies, confusion matrices (joint + slots), SNR curve,
risk–coverage curve, fairness table, own-set results, systems metrics (size, latency, RTF), and run ids.
The web app only reads this file (served at `/api/results`). If the file or a section is missing the UI shows
its documented empty state; it never invents numbers.

<!-- ===== END docs/03_TECH_STACK_AND_REPO.md ===== -->

<!-- ===== BEGIN docs/04_ROADMAP.md ===== -->

# 04 — Roadmap

Timeline is expressed in relative weeks because the deadline is not fixed in these docs. **Compress or stretch to
the real deadline** (fill in below). The critical path is data -> main model -> experiments -> report; recordings
start early and run in parallel; the web app starts only when a decent exported model exists.

- Deadline: `TODO (fill in)`   - Weeks available: `TODO`
- Yoga (Y): data, model, training, evaluation, export, backend, web build
- Haikal (H): recordings, noise clips, QA, report ch. 1-3, references, video, field-test sessions
- Task-level breakdown with IDs: docs/10_DETAILED_TASKS.md

## Phase overview
| Phase | Name | Rel. time | Owner | Depends on |
|---|---|---|---|---|
| P0 | Setup & scaffolding | 0.5 w | Y (H onboarding) | — |
| P1 | Data audit & EDA | 0.5 w | Y (H writes dataset section) | P0 |
| P2 | Splits + baselines | 0.5 w | Y | P1 |
| P3 | Main model (M0) | 1 w | Y (H runs a small training for literacy) | P2 |
| P4 | Experiments & ablations | 1.5 w | Y (H drafts methodology text) | P3 |
| P5 | Own recordings + noise clips (parallel from W1) | 1 w spread | **H** (Y builds tooling, reviews) | P0 |
| P6 | Robustness, fairness, reject | 1 w | Y (H supplies own set + noise clips) | P3, P5 |
| P7 | Export, quantize, benchmark | 0.5 w | Y | P3 |
| P8 | Backend + web demo | 1.5 w | Y builds, **H does QA** | P7 |
| P9 | Field test & polish | 0.5 w | **H leads sessions**, Y fixes | P8 |
| P10 | Report, video, repo release | 1.5 w (starts at P4) | split by chapter (docs/10), H video | all |
Suggested total: about 6–8 weeks of part-time work. P10 overlaps with P4–P9: write chapters as results land.

## P0 — Setup & scaffolding
Tasks: create repo from the layout in docs/03; `.gitignore`; `requirements.txt`; `pyproject`/ruff; empty modules with
docstrings; CI-less `pytest` smoke test; Colab runner notebook that clones the repo, mounts Drive, installs deps,
copies the dataset. Obtain the dataset (lecturer mirror) and read its license PDF.
Done when: `pytest -q` passes; Colab notebook can run `00_check_data.py` on the real dataset.

## P1 — Data audit & EDA
Tasks: implement `00_check_data.py` and `02_eda.py` per docs/01; generate `intent_map.json`; choose `max_audio_seconds`.
Done when: `reports/data_audit.json`, EDA figures and the overlap analysis exist; discrepancies vs paper sizes noted.
Report feeds: Bab 3 (dataset), Bab 1 (problem), research-gap evidence.

## P2 — Splits & baselines
Tasks: `01_make_splits.py` with leakage tests; dataset/collate classes; CRNN baseline (E1) and prior baseline;
verify the pipeline on a tiny subset (overfit 64 samples sanity check).
Done when: split tests pass; baseline trains and logs; split manifests saved.

## P3 — Main model (M0)
Tasks: `Wav2VecSLU`, trainer with resume, metrics, evaluate script; benchmark time/epoch; first full runs on A and B1.
Done when: M0 results with metrics.json on A and B1; train/serve preprocessing parity test passes; time budget
for P4 computed from measured epoch time.
Checkpoint review: compare A result with literature as a sanity check. If far lower, debug (normalization, padding,
LR, frozen encoder) before moving on.

## P4 — Experiments & ablations
Tasks: implement configs E2–E10 (docs/02 §5); `run_ablation.py` queues runs, skips finished ones; 3 seeds for finalists.
Done when: ablation table (mean ± std where applicable), confusion matrices, A-vs-B comparison, significance tests.

## P5 — Own recordings (start in week 1, finish before P6)
Tasks: write consent text; build the phrase sheet (one phrasing per intent, from FSC transcripts); recording guide
(distance, device, quiet vs noise condition); collect ≥10 speakers; convert to 16 kHz mono; `metadata.csv`; validation script.
Done when: dataset validated by script (all files readable, labels valid), counts documented.

## P6 — Robustness, fairness, reject
Tasks: noise mixing evaluation with held-out noise categories; fairness by demographics; own-set evaluation;
confidence thresholds, risk–coverage, OOD false accepts.
Done when: tables/figures per docs/01 §6–9 exist, threshold chosen on validation, chosen value written to
`configs/serve.yaml` for the backend.

## P7 — Export, quantize, benchmark
Tasks: ONNX export with dynamic time axis; parity tests; int8 dynamic quantization; accuracy delta on test and own set;
CPU latency/RTF/size table for fp32, int8, truncated variants; pick the deployment model with justification.
Done when: `models/svara.onnx` (+ meta JSON with label maps, normalization, threshold) and `reports/tables/systems.csv`.

## P8 — Backend + web demo
Tasks per docs/05 and docs/06: FastAPI app; device state machine; frontend (Live view, Lab view, About view);
browser recording -> 16 kHz WAV; Docker image; deployment to a free host.
Done when: demo works from a phone and a laptop browser; end-to-end latency measured; Lab page reads `summary.json`.

## P9 — Field test & polish
Tasks: run the live demo with new people in a real room (log outcomes); fix UX issues; record the demo video script;
finalize README with exact reproduction steps.
Done when: field-test log exists (anonymised), video recorded, README verified on a clean machine/Colab.

## P10 — Report, video, repo release
Map per docs/07. Write each chapter when its evidence exists. Final pass: figures numbered and referenced,
IEEE references verified, appendices (code listing/log excerpts/GitHub link/video link), limitations section honest.

## Cut list (drop in this order if time runs out)
1. SLURP stress test, XLS-R, Indonesian-language experiments, Raspberry Pi
2. Split C (and B2/B3 draws: keep B1 only, state it as a limitation)
3. Constrained decoding (E10), speed perturbation, reverb
4. int8 quantization (keep fp32 + truncated model)
5. Fairness breakdown beyond gender and first language
6. Mobile polish of the UI (desktop-first still must work)
**Never cut**: Split A vs B comparison, ablation core (E1–E3, E5, E6), confusion matrix, honest limitations, working demo.

## Weekly checkpoints (self-review)
- End of W1: P0–P2 done, recordings started.
- End of W3: M0 on A and B1 with trusted numbers; ablation queue running.
- End of W5: all experiments finished; export done; web skeleton working.
- End of W7: full demo, report draft complete; remaining time = revision and video.

<!-- ===== END docs/04_ROADMAP.md ===== -->

<!-- ===== BEGIN docs/05_BACKEND_AND_DEPLOYMENT.md ===== -->

# 05 — Backend, Export and Deployment

## 1. Export pipeline
1. Choose the deployment run (best trade-off from docs/02 §5–6, justified by measured numbers).
2. `scripts/export_onnx.py --run <run_dir> [--quantize int8]`
   - Wrapper module: input `waveform: float32 [batch, samples]` (already normalized), output three logit tensors
     (plus joint logits if the joint head is used). Dynamic axes for batch and samples.
   - Export with a recent opset supported by onnxruntime; test with different lengths (1 s, 2 s, 4 s).
   - Avoid attention-mask inputs (see docs/02 gotchas). Pooling must be inside the graph or replicated exactly in numpy.
3. Parity test: PyTorch vs ONNX logits on 20+ clips (max abs diff within tolerance, same argmax).
4. Quantization: `onnxruntime.quantization.quantize_dynamic` (QInt8 weights). Measure accuracy delta on the test
   split and the own set; accept only if the drop is small and documented. Keep fp32 as fallback.
5. Write `models/svara.onnx` and `models/svara.meta.json`:
```json
{
  "schema": 1, "sample_rate": 16000, "max_audio_seconds": "<from base.yaml>",
  "normalize": "zero_mean_unit_var",
  "slots": {"action": ["..."], "object": ["..."], "location": ["..."]},
  "intent_map": "<embedded copy of configs/intent_map.json>",
  "confidence": {"method": "min_head_softmax", "threshold": "<from validation>"},
  "model": {"run_id": "...", "precision": "fp32|int8", "layers": 12},
  "license_note": "Trained on Fluent Speech Commands; academic use."
}
```
Never hardcode label lists in the server: always load from meta JSON.

## 2. Backend (FastAPI + onnxruntime)
Layout: `src/svara/serve/{app.py, inference.py, device_state.py, schemas.py}`.

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + model loaded flag |
| GET | `/api/model-info` | contents of meta JSON (without secrets), model size, precision |
| POST | `/api/predict` | multipart `audio` (WAV 16 kHz mono preferred) -> prediction JSON |
| GET | `/api/results` | serves `reports/metrics/summary.json` (empty-state JSON if absent) |
| GET | `/api/devices` | current simulated home state (per session) |
| POST | `/api/devices/reset` | reset simulated state |
| GET | `/` and assets | serves built frontend (`web/dist`) |

### `/api/predict` response (schema v1)
```json
{
  "accepted": true,
  "intent": {"action":"activate","object":"lights","location":"kitchen"},
  "confidence": {"action":0.99,"object":0.98,"location":0.97,"overall":0.97},
  "threshold": 0.9,
  "top_k": [{"intent_id": 12, "slots": {"...":"..."}, "score": 0.97}],
  "valid_intent": true,
  "effect": {"type":"device","changes":[{"device":"lights","room":"kitchen","state":"on"}], "message":"Kitchen lights on"},
  "timing_ms": {"decode":4,"preprocess":2,"inference":85,"total":95},
  "audio": {"duration_s": 1.8, "sample_rate": 16000}
}
```
- `accepted=false` when overall confidence < threshold or the slot tuple is not a valid intent: then `effect` is
  `{"type":"none","message":"Not sure what you meant"}` and the state is not changed. Numbers above are illustrative
  shapes only, not real results.
- Validation: reject files > ~1 MB, duration > max + margin, non-audio; return 422 with a clear error code
  (`AUDIO_TOO_LONG`, `AUDIO_UNREADABLE`, `AUDIO_TOO_SHORT`, `SAMPLE_RATE_UNSUPPORTED`).
- Resample on the server if the client sends another rate (soundfile + a simple resampler); the web client already sends 16 kHz.
- Concurrency: one onnxruntime session loaded at startup; set intra-op threads from env; a semaphore limits parallel inference.
- CORS: allow the dev origin only (`localhost:5173`); in production the app is same-origin.
- Privacy: do not store uploaded audio by default. Optional env `SAVE_DEMO_AUDIO=false`.
- Logging: structured log line per request (id, durations, accepted flag), no audio content.

## 3. Simulated home state (`device_state.py`, `configs/device_map.yaml`)
Rooms: `kitchen`, `bedroom`, `washroom`, and `global` (for location `none`). State per session (cookie/session id) in memory.

| Object | State model | Effects |
|---|---|---|
| lights | per room: on/off, brightness 0–100 (step 20) | activate/deactivate; increase/decrease = brightness step |
| lamp | per room: on/off | activate/deactivate |
| heat | per room: on/off, target temp °C (step 1, 16–30) | activate/deactivate; increase/decrease = ±1 °C |
| music | global: playing/stopped | activate/deactivate |
| volume | global: 0–10 | increase/decrease = ±1 step |
| newspaper / juice / socks / shoes with `bring` | no device | `effect.type = "assistant_task"`, message "Fetch request: <object>" |
| `change language` + language object | UI badge only | `effect.type = "ui_language"`; shows chosen language chip; no real translation |

Rules:
- `location = none` means **all rooms** for lights/lamp/heat, and global scope for music/volume. This is a design
  decision; make it a config flag.
- Generate the list of valid (action, object, location) combos from `intent_map.json`; for each combo, look up its
  handler in `device_map.yaml`. A combo without a handler returns `effect.type="none"` with message "Recognized, no simulated effect" (log it, then add a handler).
- State transitions must be pure functions (state, intent) -> (new_state, effect) to be unit-tested.

## 4. Docker
Single multi-stage image:
1. Node stage builds `web/` -> `web/dist`.
2. Python slim stage installs `requirements-serve.txt` (onnxruntime, fastapi, uvicorn, numpy, soundfile, pyyaml), copies
   `src/`, `models/svara.onnx`, `models/svara.meta.json`, `reports/metrics/summary.json`, `web/dist`.
3. `CMD uvicorn svara.serve.app:app --host 0.0.0.0 --port ${PORT:-7860}`.
Image must run without GPU and without internet. Keep image size small; do not include PyTorch in the serve image.
(`soundfile` needs libsndfile; install it in the image.)

## 5. Hosting options (pick one, verify current free-tier limits before committing)
| Option | Pros | Cons / checks |
|---|---|---|
| Hugging Face Spaces (Docker SDK) | free tier, natural fit for HF-based project, public URL | verify CPU/RAM limits and cold start; port conventions |
| Render / Railway / Fly.io | simple Docker deploys | free tiers change; check RAM (ONNX model + runtime) and sleep behavior |
| Run locally on a laptop for the live presentation | zero risk of hosting limits | needs a recorded backup video |
Always prepare a **local fallback** (`docker run -p 7860:7860 ...`) and a recorded demo video for the presentation.
Browser microphone requires HTTPS (or `localhost`): make sure the hosted URL is HTTPS.

## 6. Performance targets (to be measured, then reported; do not assume)
Measure on the target CPU: model size, p50/p95 inference latency for 1–3 s clips, end-to-end latency in the browser
(record stop -> result visible). Include network latency separately from model latency in the report.

## 7. Optional extensions
- Gradio fallback demo (`app_gradio.py`) for quick sharing if the custom web app slips.
- Raspberry Pi or ESP32-relay prototype (stretch): same ONNX model on a Pi, relay toggled by GPIO.
- Streaming/VAD-based endpointing (stretch): current design is push-to-talk with a fixed max length.

## 8. Tests
- `tests/test_device_state.py`: every intent in `intent_map.json` maps to a handler or a documented fallback; transitions pure.
- `tests/test_api.py`: FastAPI TestClient with synthetic WAV (sine/silence) checks schema, errors, reject path.
- `tests/test_inference_parity.py`: server preprocess equals training preprocess (docs/02 §10).

<!-- ===== END docs/05_BACKEND_AND_DEPLOYMENT.md ===== -->

<!-- ===== BEGIN docs/06_UI_UX_SPEC.md ===== -->

# 06 — UI / UX Specification

## 1. Product framing
- **Subject:** a voice-controlled home you can talk to, plus a research view that proves the model works.
- **Audience:** lecturer and classmates during a demo (first-time users), and report readers who want evidence.
- **Primary job of the UI:** let a stranger press a button, say one command, and *see the house react* within a
  second or two. Secondary job: let a reader inspect results honestly (the "Lab").
- **Memorable element (spend the boldness here, keep everything else quiet):** a top-down **floor plan** of a small
  home where devices physically respond: lights glow, the heater shows its temperature, music bars pulse.
  Nothing else on the page competes with it.
- Language of UI copy: Indonesian by default, with an EN toggle only if time remains. Keep labels plain and literal.

## 2. Information architecture
| View | Route | Purpose |
|---|---|---|
| Rumah (Live) | `/` | microphone, floor plan, prediction, event log |
| Lab | `/lab` | evidence: protocols, ablation, confusion matrices, noise curve, reject curve, fairness, own set, systems metrics |
| Tentang | `/about` | dataset, method in 5 sentences, limitations, license/consent note, team |
Top bar: product name, three links, a small status chip (`Model siap` / `Memuat model` / `Server tidak terjangkau`).

## 3. Visual system (tokens)
Avoid the common generated-page defaults (cream + terracotta, near-black + acid accent, identical rounded cards,
tracked ALL-CAPS eyebrows, "→" on every link, monospace micro-labels). Use these deliberate choices instead:

| Token | Name | Hex | Use |
|---|---|---|---|
| `--plaster` | Plaster | `#E8ECE6` | page background |
| `--ink` | Ink | `#16303A` | text, walls of the floor plan |
| `--pine` | Pine | `#2F6B5E` | primary actions, selected states |
| `--lamp` | Lamp | `#F2B33D` | **only** for "device is on / active" glow |
| `--brick` | Brick | `#C4492F` | rejected / error / recording |
| `--mist` | Mist | `#A9B8B2` | borders, off-state devices, dividers |
| `--paper` | Paper | `#F6F8F4` | raised surfaces (sparingly) |
Dark theme (optional): swap `--plaster` -> `#0F2128`, `--ink` -> `#E8ECE6`; keep `--lamp` and `--brick`.
Contrast: body text >= 4.5:1; check `--lamp` text on `--plaster` (do not put amber text on light backgrounds).

Typography (Google Fonts, provide system fallbacks):
- UI and headings: **Familjen Grotesk** (weights 400/500/700).
- Long text in Lab/About (captions, interpretations): **Newsreader** (400/500), slightly larger line-height than sans.
- Scale: 14 / 16 / 20 / 28 / 44 px; line length < 72 characters; sentence case everywhere; no all-caps labels.
- Numbers in tables use tabular figures (`font-variant-numeric: tabular-nums`).

Shape and space: 8-px spacing grid. Two radii only: 4 px (inputs, chips) and 16 px (floor plan frame, main panels).
Vary hierarchy by size and spacing rather than by wrapping everything in identical cards. No gradient washes.
Motion: one orchestrated moment only (page load: floor plan walls draw in, ~600 ms). Everything else is
action-driven and short (120–200 ms): a device lighting up, a chip appearing. Respect `prefers-reduced-motion`.

## 4. View: Rumah (Live)
Desktop wireframe (>= 1024 px):
```
+----------------------------------------------------------------------------+
| SVARA            Rumah   Lab   Tentang                  [Model siap]   |
+------------------------------------------+---------------------------------+
|                                          |  Perintah terakhir              |
|   +----------------+----------------+    |  action   [ activate  ] ####    |
|   |  Dapur         |  Kamar tidur   |    |  object   [ lights    ] ####    |
|   |  (lampu, pemanas)| (lampu, lamp,|    |  location [ kitchen   ] ###     |
|   |                |   pemanas)     |    |  Diterima | 95 ms               |
|   +----------------+----------------+    |  "Lampu dapur menyala"          |
|   |  Kamar mandi   |  Lorong        |    +---------------------------------+
|   |  (lampu, pemanas)| musik/volume |    |  Riwayat                        |
|   +----------------+----------------+    |  12:01 activate lights kitchen  |
|        FLOOR PLAN (SVG)                  |  12:00 ditolak (ragu)           |
+------------------------------------------+---------------------------------+
|  [  Tahan untuk bicara  (atau tekan Spasi)  ]   ~~~ waveform ~~~   [Unggah] |
+----------------------------------------------------------------------------+
```
Mobile (< 700 px): floor plan on top, result below, mic button fixed at the bottom (thumb reachable), history in a collapsible sheet.

### Floor plan (SVG, the hero)
- Four zones: Dapur (kitchen), Kamar tidur (bedroom), Kamar mandi (washroom), Lorong (hall, holds music/volume).
  Map zones to `location` values; `none` highlights the whole plan.
- Walls in `--ink`, room floors in `--paper`. Devices:
  - Lights: ceiling circle; on = `--lamp` fill + soft radial glow across the room floor; brightness controls glow opacity.
  - Lamp: floor-lamp icon with a small cone of light when on.
  - Heat: radiator glyph with the target temperature as text (e.g., `22 °C`), tinted `--brick` only while on.
  - Music/volume: speaker in the hall; 10 volume ticks; animated bars only while music is playing.
- When a command is accepted: the affected device animates (≤ 200 ms) and its room gets a brief outline in `--pine`.
- Clicking a device toggles it manually (useful for resetting the demo and for judges to explore).
- Rejected command: no device change; the result panel shows the muted best guess, labelled as uncertain.
- Unsupported-but-recognized (`bring`, `change language`): show an "assistant task" ribbon above the plan
  (e.g., "Permintaan antar: koran") with no device effect; language chip updates for `change language`.

### Microphone control
- Large push-to-talk button: hold to record (mouse, touch) or hold Space; release to send. Max recording length = model max seconds;
  show a thin progress ring that fills toward the limit.
- States and labels: `Tahan untuk bicara` (idle), `Merekam…` (recording, button turns `--brick`), `Memproses…` (spinner, button disabled), `Selesai`.
- Waveform: live `AnalyserNode` bars while recording; after recording, a static mini waveform of what was sent.
- Permission denied: explain how to allow the microphone and offer **file upload** as the fallback (WAV/MP3/WebM accepted client-side, converted to 16 kHz WAV).
- Suggested commands: a quiet row of 5–6 example phrases (text only, no audio files) that rotate through different intents.
  They are suggestions of what to say, not buttons.

### Result panel
- Three slot rows (action, object, location). Each row: value, confidence bar (width only, no gradient), numeric %.
- Status line: `Diterima` (Pine) or `Ditolak: kurang yakin` (Brick) + the threshold used + total latency and model latency.
- One plain sentence describing the effect (from the API `effect.message`).
- "Detail" disclosure: top-k intents, valid-intent flag, audio duration. Collapsed by default.
- No transcript is shown: the model is end-to-end and does not produce text. Say so in the About page, not in the main UI.

### Event log
Latest first, max 50, each line: time, slots, accepted/rejected, latency. "Reset rumah" button (confirm not needed; undo toast).

## 5. View: Lab (evidence)
Purpose: make the evaluation readable in under 5 minutes. All data from `/api/results`. No hardcoded numbers.
Sections (anchored, sticky side index on desktop):
1. **Ringkasan**: what was trained, on which split, the headline exact-match accuracy with CI and the model size/latency chosen. Large type, no decoration.
2. **Split A vs B**: grouped bars for exact-match and slot accuracies; caption explains phrase leakage in 2 sentences, with the measured overlap percentage.
3. **Ablation**: sortable table (variant, split, exact-match mean ± std, params, size, latency) + accuracy-vs-latency scatter with labelled points.
4. **Confusion matrices**: slot tabs (action/object/location/joint); joint 31×31 heatmap with hover tooltip; sequential single-hue colour scale (Pine); toggle row-normalised/raw.
5. **Noise**: line chart accuracy vs SNR per model (clean at the right end).
6. **Reject curve**: risk–coverage chart with the chosen threshold marked; OOD false-accept rate beside it.
7. **Fairness**: horizontal bars per group with CIs; groups with n < 30 hatched and labelled "sampel kecil".
8. **Uji lapangan (own set)**: quiet vs noisy, per device; show n.
9. **Sistem**: size/latency/RTF table for fp32, int8, truncated.
Each section: short human-written caption (Newsreader) stating what to notice, plus a "Sumber" line with the run id(s).
Empty state per section: "Hasil belum tersedia. Jalankan `python scripts/make_report_assets.py`." (plain, no apology).
Charts: Recharts; every chart has a text alternative (table toggle) and colour-blind-safe encodings (not colour alone).

## 6. View: Tentang
Short, plain sections: What this is (3 sentences), How it works (diagram: audio -> wav2vec2 -> three heads -> intent -> home),
Dataset and license notice, Limitations (English-only commands, clean-ish training data, small field test, simulated devices),
Privacy (audio is processed in memory and not stored, if true — keep this sentence accurate to the deployed config), Team.

## 7. Components (suggested)
`TopBar`, `StatusChip`, `FloorPlan` (+ `Room`, `DeviceLights`, `DeviceLamp`, `DeviceHeat`, `DeviceMusic`), `PushToTalk`,
`Waveform`, `ResultPanel`, `SlotRow`, `EventLog`, `LabSection`, `Charts/*`, `EmptyState`, `Toast`.
State: React state + small context for session/home state; server is the source of truth for device state per session.
Data layer: typed API client (`web/src/api.ts`) mirroring the schemas in docs/05.

## 8. Audio capture details (client)
- `getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: false}})`; keep noise suppression off by default
  so the model sees audio similar to training (make this a documented setting; test both).
- Record with MediaRecorder, decode via `AudioContext.decodeAudioData`, resample to 16 kHz mono with `OfflineAudioContext`, encode PCM16 WAV in JS, POST as multipart.
- Trim leading/trailing silence conservatively (energy threshold) and cap to the model max length; never send > limit.
- Handle iOS Safari quirks (user gesture to start AudioContext; MediaRecorder MIME differences).

## 9. Accessibility and quality floor
Keyboard: Space/Enter operate push-to-talk, Tab order follows visual order, visible focus ring (2 px Pine offset).
Screen readers: live region announces results ("Perintah diterima: aktifkan lampu dapur"). Colour is never the only
signal (icons/text for on/off and accepted/rejected). Touch targets ≥ 44 px. Responsive 360–1440 px. Reduced motion respected.
Lighthouse accessibility ≥ 90 as a target, measured and reported.

## 10. Copy rules
Plain verbs, sentence case, no filler. A button names its action ("Reset rumah", "Unggah rekaman"). Errors state what happened and what to do
("Mikrofon diblokir. Izinkan akses mikrofon di pengaturan browser, atau unggah file rekaman.") and never apologise. The same action keeps the same name everywhere.

## 11. Not in v1
Authentication, accounts, wake-word, multi-user rooms, real IoT control, saving user audio, i18n beyond ID/EN.

<!-- ===== END docs/06_UI_UX_SPEC.md ===== -->

<!-- ===== BEGIN docs/07_REPORT_MAPPING.md ===== -->

# 07 — Report Mapping

The report follows the course file `Template_Laporan_TB.docx`: identity page, abstract, 8 chapters, IEEE references,
appendices. **Keep the lecturer's template as the source of truth** (re-open it for exact headings and numbering;
the mapping below is based on the headings we read). Report text is in Indonesian.

## 1. Chapter -> evidence map
| Template section | What goes in | Source artifacts |
|---|---|---|
| Identitas | names, NIM, class, lecturer, GitHub and video links | fill manually |
| Abstrak | problem, method, key results (3–4 numbers), conclusion | `reports/metrics/summary.json` |
| Bab 1 Pendahuluan | background of voice-controlled homes, problem, objectives, scope, contributions | docs/00 |
| Bab 2 Studi Literatur + Research Gap | SLU vs ASR+NLU pipeline, wav2vec 2.0, FSC and its limits, SLURP, noise robustness, fairness; the gap: honest unseen-phrase evaluation + accented/noisy field test + safe actuation | docs/00 §3–4, references below |
| Bab 3 Dataset & Sistem | dataset description, license, preprocessing, system architecture diagram, hardware/software requirements | `reports/data_audit.json`, EDA figures/tables, docs/03 |
| Bab 4 Metodologi | splits A/B/C and leakage argument, model, losses, augmentation, training setup, metrics, statistics, experiment design | docs/01, docs/02, `reports/runs/*/config.yaml` |
| Bab 5 Implementasi | code structure, training procedure, hyperparameters, environment (Colab GPU, versions), challenges | `env.txt`, configs, `train_log.csv` |
| Bab 6 Hasil & Evaluasi | main results, A vs B, ablation study, confusion matrix, noise, fairness, own set, reject analysis, error analysis, discussion | `reports/tables/*`, `reports/figures/*`, `summary.json` |
| Bab 7 Deployment (and field test) | export, quantization, latency/size table, API, web UI screenshots, field test protocol and outcomes | docs/05, docs/06, `systems.csv`, field-test log |
| Bab 8 Kesimpulan | answers to RQ1–RQ7, limitations, future work | all of the above |
| Daftar Pustaka | IEEE style | section 3 below |
| Lampiran | code listings (key files), training logs, GitHub link, demo video link, consent form template (no personal data) | repo, `reports/logs/` |

## 2. Auto-generated report assets (`scripts/make_report_assets.py`)
Produces, from `reports/runs/**` only, and fails loudly if a required run is missing:
- Figures (PNG, 300 dpi, consistent style, axis labels with units, readable at column width):
  `fig_eda_*`, `fig_split_ab.png`, `fig_ablation.png`, `fig_confusion_<slot|joint>.png`, `fig_noise_snr.png`,
  `fig_risk_coverage.png`, `fig_fairness_<attr>.png`, `fig_systems_tradeoff.png`, `fig_architecture.png` (diagram can be drawn separately).
- Tables (CSV + Markdown + LaTeX-friendly): dataset stats, split sizes, hyperparameters, ablation, systems metrics, fairness, own set.
- `reports/metrics/summary.json` for the web Lab.
- A `RESULTS_INDEX.md` that lists each figure/table with its source run ids, so every number in the report is traceable.

## 3. Candidate references (VERIFY every entry — authors, venue, year, pages — before citing; IEEE format)
- Lugosch, Ravanelli, Ignoto, Tomar, Bengio. Speech Model Pre-training for End-to-End Spoken Language Understanding. Interspeech 2019. (assigned main paper; this introduced Fluent Speech Commands)
- Baevski, Zhou, Mohamed, Auli. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS 2020.
- Bastianelli, Vanzo, Swietojanski, Rieser. SLURP: A Spoken Language Understanding Resource Package. EMNLP 2020.
- Park et al. SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. Interspeech 2019.
- Piczak. ESC: Dataset for Environmental Sound Classification. ACM Multimedia 2015.
- Yang et al. SUPERB: Speech processing Universal PERformance Benchmark. Interspeech 2021.
- Ravanelli et al. SpeechBrain: A General-Purpose Speech Toolkit. arXiv 2021.
- Evaluation of FSC splits and phrase overlap: "Rethinking End-to-End Evaluation of Decomposable Tasks: A Case Study on Spoken Language Understanding" (arXiv 2106.15065; confirm authors and venue).
- Lee et al. Speech-MASSIVE: A Multilingual Speech Dataset for SLU and Other Tasks. (confirm venue/year)
- Wolf et al. Transformers: State-of-the-Art Natural Language Processing. EMNLP 2020 (System Demonstrations).
- Fluent Speech Commands dataset page and license (cite the official source named in the license PDF).
Also add the lecturer's spreadsheet only if the template requires it. Do not cite blog posts for technical claims when a paper exists.

## 4. Writing rules
- Every claim of performance cites a table/figure; every table/figure is referenced in the text.
- Report negative or surprising results and discuss them; do not tune the test set. Choose thresholds and hyperparameters on validation only.
- State sample sizes, seeds, and confidence intervals. For the own set say clearly it is a small field test.
- Error analysis: show the 10 most frequent confusions and 5 qualitative failure cases (with intents, not raw audio, if consent is limited).
- Limitations section must include: English-only FSC, clean training data, US/Canada speakers, simulated devices, small own set, compute limits.
- Plagiarism and AI-use: write the discussion in the team's own words; keep notes of how tools were used if the course requires disclosure.
- Figures: consistent palette (reuse UI tokens from docs/06 where sensible), no screenshots of terminals as results.

## 5. Report timeline hints
Write Bab 1–3 during P1–P2, Bab 4–5 during P3–P4, Bab 6 as experiments finish, Bab 7 after P8, Bab 8 and Abstrak last.
Keep a running `reports/NOTES.md` with decisions, surprises and failed attempts: it saves hours later.

<!-- ===== END docs/07_REPORT_MAPPING.md ===== -->

<!-- ===== BEGIN docs/08_ANTIGRAVITY_PROMPTS.md ===== -->

# 08 — Ready-to-paste Prompts for Antigravity

How to use: open the repo root in Antigravity (AGENTS.md is read automatically). Paste one prompt at a time,
review the plan/diff, run the stated check, then commit. If an agent proposes something that contradicts the docs,
point to the doc section instead of re-explaining. Ask it to **stop and report** when assumptions fail.

Tip: set agent autonomy to "agent-assisted" (review before risky commands) while you learn how it behaves.

---
## Prompt 0 — Orient the agent (run once)
```
Read AGENTS.md and every file in docs/ (00 to 11, including the GitHub workflow in 09, task list in 10 and verification checklist in 11). Then write a short summary (max 25 lines) of: the project goal, the three
split protocols, the main model, the deliverables, and the rules you must never break. List any contradictions or
ambiguities you found in the docs. Do not write code yet.
```

## Prompt 1 — Scaffold (P0)
```
Create the repository scaffold exactly as specified in docs/03 section 4: folders, empty modules with docstrings,
.gitignore (must ignore data, checkpoints, wav files, node_modules, .env), requirements.txt and requirements-serve.txt
(unpinned for now), pyproject with ruff config, a tests/ folder with one smoke test, and a README with setup steps.
Create notebooks/90_colab_runner.ipynb that mounts Google Drive, clones the repo, installs requirements, copies the
dataset from Drive to /content/data and calls scripts/00_check_data.py. Notebook must contain no unique logic.
Run ruff and pytest and show me the results.
```

## Prompt 2 — Data audit (P1)
```
Implement scripts/00_check_data.py and src/svara/data/fsc.py following docs/01 sections 1-2 (all 11 audit checks, including label consistency, duplicate audio hashes, and phrasings per intent; docs/11 V-01..V-08).
Read column names from the real CSV headers (the dataset README says `transcript`). Produce reports/data_audit.json
and configs/intent_map.json. Include the original-split overlap analysis (speakers and transcripts shared between
train/valid/test). Add tests using a tiny synthetic fixture dataset (no real data in git). Print a human-readable
summary and flag differences from the expected counts (23132/3118/3793) without failing.
```

## Prompt 3 — EDA (P1)
```
Implement scripts/02_eda.py per docs/01 section 3. Save PNG figures (300 dpi) to reports/figures and CSV tables to
reports/tables. Use a consistent matplotlib style. Then create notebooks/01_eda.ipynb that only loads and displays
those outputs. Recommend max_audio_seconds from the duration percentiles and write it to configs/base.yaml.
```

## Prompt 4 — Splits (P2)
```
Implement scripts/01_make_splits.py and src/svara/data/splits.py per docs/01 section 4: Split A (original),
Split B (unseen-utterance; three draws B1-B3, grouped by transcript, per-intent hold-out), Split C (speaker-grouped,
demographically stratified). Write split_manifest.json with seed, rule and counts. Add pytest tests that assert no
transcript leakage for B, no speaker leakage for C, and that every intent still has >= 2 training phrasings.
If a constraint cannot be satisfied for some intent, log it and explain; do not silently relax it.
```

## Prompt 5 — Dataset class, audio utils, baseline (P2)
```
Implement src/svara/data/audio.py (single shared preprocessing function: load, mono, 16 kHz check, normalize,
crop/pad to max_audio_seconds), collate with length bucketing, and the PyTorch Dataset for a split CSV.
Then implement the CRNN baseline (docs/02 section 2) and scripts/train.py with config-driven training, CSV logging,
checkpoint-per-epoch and --resume auto. Include a sanity mode that overfits 64 samples to prove the pipeline works.
Do not implement wav2vec2 yet.
```

## Prompt 6 — Main model (P3)
```
Implement src/svara/models/wav2vec_slu.py per docs/02 section 1: wav2vec2 encoder from HuggingFace,
frozen CNN feature encoder, optional layer truncation, length-aware mean pooling, three heads and an optional joint head.
Respect the wav2vec2 gotchas (no attention_mask for base models, zero padding, frame-length computation).
Add the pooling-mask test and the train/serve preprocessing parity test from docs/02 section 10.
Extend trainer.py with two LR groups, warmup+linear decay, fp16, early stopping on validation exact-match.
Add configs/model_w2v2_3head.yaml. Explain how I should run one epoch on Colab and what to record (time per epoch, GPU memory).
```

## Prompt 7 — Metrics and evaluation (P3)
```
Implement src/svara/eval/metrics.py and scripts/evaluate.py per docs/01 section 5: exact-match, per-slot accuracy,
macro-F1, confusion matrices (slots and 31x31 joint), bootstrap CI, McNemar/paired bootstrap for comparing two runs,
invalid-intent rate, optional constrained decoding. Write metrics.json and predictions.csv into the run directory.
Add unit tests with hand-made predictions where the expected numbers are known.
```

## Prompt 8 — Ablations (P4)
```
Create the configs for experiments E1-E10 in docs/02 section 5 and scripts/run_ablation.py that runs a queue of
configs on a given split and seed, skips finished runs (detected by metrics.json), and is safe to resume after a Colab
disconnect. Add a command to print a comparison table from reports/runs. Do not run training here; show me the
exact Colab commands and an estimate of total GPU time based on the time-per-epoch I will provide.
```

## Prompt 9 — Robustness, fairness, reject (P6)
```
Implement robustness.py (noise mixing at SNR 20/10/5/0 dB with partitioned noise categories, per docs/01 section 6),
fairness.py (per-group accuracy with CIs using speaker_demographics.csv), reject.py (confidence scores, threshold choice on
validation, risk-coverage, OOD false-accept rate, optional temperature scaling and ECE) and scripts for each.
All must write CSV + figures into reports/. Explain any assumption about noise files; do not download data without telling me.
```

## Prompt 10 — Own recordings tooling (P5)
```
Create tooling for the own-recording set: a script that generates the phrase sheet (one FSC phrasing per intent) as a
printable Markdown/CSV, a validator for data/own_recordings (readable, 16 kHz mono after conversion, labels valid, metadata
complete), a converter (any audio -> 16 kHz mono WAV), and scripts/eval_own_set.py reporting quiet vs noisy and per device.
Never put audio or personal data into git. Add a consent-form template under docs/ (no personal data).
```

## Prompt 11 — Export and benchmark (P7)
```
Implement export/to_onnx.py, quantize.py and benchmark.py per docs/05 section 1: ONNX export with dynamic axes, parity
test against PyTorch, int8 dynamic quantization, accuracy delta on test and own set, CPU latency p50/p95, RTF, size,
peak RAM. Output models/svara.onnx, models/svara.meta.json and reports/tables/systems.csv. If quantization hurts
accuracy noticeably, report it and keep fp32 as the default.
```

## Prompt 12 — Backend (P8)
```
Implement the FastAPI backend per docs/05 sections 2-3: endpoints, schemas, onnxruntime inference using meta.json,
validation and error codes, the pure device-state machine driven by configs/device_map.yaml and intent_map.json,
reject path, structured logs without audio. Add tests (TestClient with synthetic WAV, device-state tests covering every
intent). Provide a curl example for each endpoint.
```

## Prompt 13 — Frontend foundation (P8)
```
Scaffold web/ with React + Vite + TypeScript following docs/06. Implement design tokens as CSS variables (docs/06 section 3),
fonts with fallbacks, the app shell (TopBar, routes /, /lab, /about), a typed API client mirroring docs/05 schemas, and an
EmptyState component. Do not build the floor plan yet. Use mocked API responses ONLY behind an explicit dev flag and never
show mocked numbers in the Lab view.
```

## Prompt 14 — Floor plan + push-to-talk (P8)
```
Build the Rumah view per docs/06 section 4: the SVG floor plan (rooms, lights, lamp, heat, music/volume) driven by the
server's device state, the push-to-talk component with 16 kHz WAV encoding in the browser (docs/06 section 8), waveform,
result panel with three slot rows, event log, reject state, unsupported-but-recognized ribbon. Keyboard (Space) and touch
support, reduced-motion support. Manual device toggles. Show me screenshots at 390px and 1280px widths.
```

## Prompt 15 — Lab view (P8)
```
Build the Lab view per docs/06 section 5, reading only /api/results (reports/metrics/summary.json). Implement each
section with the specified chart type and the table-toggle accessibility alternative. Implement empty states exactly as
described. If summary.json lacks a field, show the empty state; never invent values.
```

## Prompt 16 — Docker and deploy (P8)
```
Write the multi-stage Dockerfile per docs/05 section 4, a docker-compose or run script for local use, and step-by-step
deployment notes for the hosting option I choose (I will tell you which). Verify the image runs without GPU and without
internet access, and that the microphone works over localhost and HTTPS. List every environment variable.
```

## Prompt 17 — Report assets (P10)
```
Implement scripts/make_report_assets.py per docs/07 section 2: generate figures and tables from reports/runs only, write
reports/metrics/summary.json and RESULTS_INDEX.md. Fail loudly if required runs are missing. Do not produce placeholders.
```

## Prompt 18 — Review and hardening
```
Act as a strict reviewer. Check the repo against AGENTS.md rules and docs/ (including the docs/11 checklist and test-set discipline): test-set reads outside evaluate.py, leakage in splits, preprocessing parity,
hardcoded labels or numbers, secrets/data in git, missing tests, unreproducible commands, README accuracy.
Produce a prioritized list of issues with file paths and proposed fixes. Do not change code until I approve.
```

## Prompt 19 — Create GitHub issues from the task list (P0-06)
```
Read docs/10_DETAILED_TASKS.md and write scripts/dev/create_issues.py that, using the GitHub CLI (`gh`), creates one issue per
task row: title "[P3-07] <task>", body with owner, effort, depends-on and done-when, labels phase:Pn and owner:yoga|haikal.
Add a --dry-run mode that only prints what would be created and a --phase filter. Do NOT run it for real: show me the dry-run output first.
```

## Prompt 20 — Deep-learning sanity checks (P2-12, docs/11 section 3)
```
Implement scripts/sanity_checks.py and the supporting tests for docs/11 checks V-10 to V-19: batch inspection (save 5 augmented wavs),
initial loss per head vs ln(C), overfit-64, shuffled-label control, silence/noise input control, trainable-parameter count vs config,
gradient check (frozen vs unfrozen), eval-mode determinism, padding invariance, NaN/inf guard with gradient clipping.
Output a single Markdown report reports/sanity_<run>.md with PASS/FAIL per check and the measured values. Do not change thresholds to make checks pass.
```

## Prompt 21 — LR range test and hyperparameter search (P3-13, P3-14)
```
Implement scripts/lr_range_test.py and scripts/hparam_search.py following docs/11 section 4: tune on Split B1 validation only, single seed,
8-12 runs max, same budget for the CRNN baseline, results in reports/tables/hparam_search.csv. The script must refuse to read any test
split (assert in code). Print the Colab commands and an estimate of GPU time from the measured time per epoch I provide.
```

## Prompt 22 — Reproducibility and claims audit (P3-15, P10-25)
```
Add reproducibility utilities per docs/11 section 6 (seeding incl. DataLoader workers, deterministic flags with documented exceptions,
run metadata with split-manifest hash and GPU name). Then create scripts/dev/claims_audit.py that scans docs/report/*.md for numbers and
comparisons and lists sentences that are not yet mapped in reports/CLAIMS_AUDIT.md. Do not auto-fill evidence; only report gaps.
```

---
## Handy follow-up prompts
- "Show me the diff summary and which tests cover it. What is NOT covered?"
- "That contradicts docs/0X section Y. Re-read it and fix the implementation, not the doc."
- "Before you implement, give me two options with trade-offs and a recommendation."
- "Stop. List the assumptions you made in the last task and how to verify each."

<!-- ===== END docs/08_ANTIGRAVITY_PROMPTS.md ===== -->

<!-- ===== BEGIN docs/09_GITHUB_COLLABORATION.md ===== -->

# 09 — GitHub Collaboration (Yoga + Haikal)

Goal: two people, one repo, no overwritten work, no lost data, no leaked recordings. Everything below is simple on purpose.

## 1. What lives where
| Thing | Where | Why |
|---|---|---|
| Code, configs, docs, tests, small result files (`reports/metrics/*.json`, figures, tables) | **GitHub repo** | versioned, reviewable |
| Dataset (FSC wavs), own recordings, household-noise clips | **Google Drive** (`MyDrive/svara/data/...`), never in git | large, licensed, personal |
| Checkpoints (`*.ckpt`, `*.pt`) | **Google Drive** (`MyDrive/svara/runs/`) | large |
| Final deployable model (`svara.onnx`, `svara.meta.json`) | GitHub **Release** asset (or Git LFS / Hugging Face Hub) | needed by Docker build, but too big for normal commits |
| Secrets/tokens | nowhere in the repo; environment variables / platform secrets | security |
Drive layout (shared with both accounts as editor):
```
MyDrive/svara/
├── data/raw/fluent_speech_commands_dataset/
├── data/own_recordings/{wavs_raw/, wavs_16k/, metadata.csv, consent/}   # consent/ is PRIVATE: only Haikal + Yoga
├── data/noise_own/
├── data/noise/esc50/
├── runs/                      # reports/runs mirrors here from Colab
└── models/
```

## 2. Repo setup (task P0-01 .. P0-07)
1. Yoga creates a **private** repo `svara` and invites Haikal as collaborator (write access).
2. Branch protection on `main`: require a pull request, require 1 approval, disallow force-push.
3. Add `.github/` files from this package: `CODEOWNERS`, `PULL_REQUEST_TEMPLATE.md`, `ISSUE_TEMPLATE/task.md`, `ISSUE_TEMPLATE/bug.md`.
4. Create a GitHub **Project** (board) with columns: `Backlog`, `Ready`, `In progress`, `In review`, `Done`, `Blocked`.
5. Create labels: `phase:P0` … `phase:P10`, `owner:yoga`, `owner:haikal`, `type:code`, `type:data`, `type:report`, `type:qa`, `type:video`, `blocked`, `decision`, `cut-candidate`.
6. Create one GitHub Issue per task in docs/10 (title `[P3-07] Run M0 on Split A seed 42`). Paste the task row as the issue body.
   Tip: ask Antigravity to generate a script with the GitHub CLI (`gh issue create`) from docs/10; review it before running.

## 3. Branching and commits
- `main` is always working and demo-able. **Nobody commits directly to main.**
- Branch name: `<type>/<task-id>-<short-name>`, e.g. `feat/P3-01-wav2vec-slu`, `data/P5-09-own-set-metadata`, `docs/P10-03-bab3`.
- Commit messages (conventional): `feat:`, `fix:`, `test:`, `docs:`, `data:` (metadata only), `exp:` (experiment configs/results), `chore:`.
  Include the task id: `feat(model): add length-aware pooling [P3-01]`.
- Small PRs (aim for < 400 changed lines, one task or a few related tasks).
- Merge strategy: **squash and merge**, delete the branch afterwards.
- Pull before you start: `git checkout main && git pull`, then create the branch.
- Never use `git push --force` on shared branches. Never commit files larger than 5 MB (CI/pre-commit check recommended).

## 4. Pull request rules
- Every PR links its issue(s) (`Closes #12`), says what changed, how it was tested, and what is NOT done.
- The **other person reviews**: Yoga's code PRs are reviewed by Haikal for readability (can he follow the README steps?),
  and Yoga reviews Haikal's data/report PRs for correctness. Reviewing is also how Haikal learns the pipeline.
- PR must pass: `ruff`, `pytest -q` (when code is involved). Report PRs: spell-check and figure/table references resolve.
- Review comments are specific ("line 40: what happens if the file is empty?") and answered before merge.
- Antigravity-authored PRs: the human who triggered it reads the diff before opening the PR; summary generated by the agent is a starting point, not proof.

## 5. Ownership map (also encoded in `.github/CODEOWNERS`)
| Path | Primary owner | Reviewer |
|---|---|---|
| `src/`, `scripts/`, `configs/`, `tests/`, `Dockerfile`, `requirements*.txt` | Yoga | Haikal |
| `web/` | Yoga | Haikal (QA + copy) |
| `reports/runs/`, `reports/metrics/`, `reports/figures/`, `reports/tables/` | Yoga | Haikal |
| `data/own_recordings/metadata.csv` (metadata only), `docs/notes/`, `reports/notes/` | Haikal | Yoga |
| `docs/report/` (report chapter drafts in Markdown), `docs/references.bib` or `docs/references.md` | Haikal (ch. 1-3, refs), Yoga (ch. 4-6) | the other |
| `docs/00`-`docs/10`, `AGENTS.md` | Yoga | Haikal (reads and flags unclear parts) |
Editing outside your area is fine if you open a PR and tag the owner.

## 6. Notebooks and merge conflicts
- Notebooks hold no logic (docs/03). Before committing a notebook, **clear outputs** (`jupyter nbconvert --clear-output --inplace`) to avoid giant diffs.
- Do not both edit the same notebook at the same time; announce in chat first.
- Report drafts are Markdown files under `docs/report/` (one file per chapter) so both can work in parallel without conflicts.
  The final Word/PDF is assembled from these files by Haikal using the course template (docx), after Yoga's final review.
- If a merge conflict appears: stop, ask for help in chat (screenshare), don't "accept all".

## 7. Haikal onboarding checklist (task P0-02, P0-05, P0-11)
1. Install Git and VS Code (or use Antigravity), create/confirm GitHub account, accept the repo invite.
2. `git config --global user.name "..."` and `user.email "..."`.
3. Clone: `git clone <repo-url>`; open the folder; read `README.md` and `docs/00_PROJECT_BRIEF.md`.
4. Create a branch, change one line in `docs/notes/hello.md`, commit, push, open a PR, get it merged (practice run, 15 minutes).
5. Open `notebooks/90_colab_runner.ipynb` in Colab, run all cells, confirm the data check prints a summary.
6. Learn five commands: `git status`, `git pull`, `git checkout -b`, `git add -A && git commit -m "..."`, `git push -u origin <branch>`.
Yoga does a 30-minute screenshare for this once; after that Haikal works independently.

## 8. Sharing results from Colab
- Training writes to `/content/svara/reports/runs/<run_id>` and **syncs to Drive** at every epoch.
- After a run finishes, the notebook cell `Export run summary` copies only small files (`config.yaml`, `metrics.json`, `train_log.csv`,
  `predictions.csv`, figures) into the repo working copy; Yoga commits them on a branch `exp/<run_id>` and opens a PR (so results are reviewable).
- Do not commit `predictions.csv` if it exceeds ~5 MB; compress or keep on Drive and commit only the summary.

## 9. Own-recording data protocol (Haikal)
- Raw recordings go to Drive `data/own_recordings/wavs_raw/`, converted files to `wavs_16k/` using the tool built in P5-02.
- Speaker IDs are anonymous (`own_s01` …). The mapping from person to ID is kept in a private note on Haikal's side and **never** in the repo or Drive shared folders others can access.
- Consent confirmations (photo/screenshot/signed text) are stored in Drive `consent/` (private), not in the repo.
- Only `metadata.csv` (no names, no contact info) is committed.
- If a speaker withdraws consent, delete their files everywhere and re-run the affected evaluations (log this in `reports/NOTES.md`).

## 10. Rituals
| When | What | Who | Length |
|---|---|---|---|
| Twice a week (fixed slots) | async status in chat: done / next / blocked | both | 5 min |
| Weekly | sync call: demo what works, move cards on the board, adjust plan, update decision log | both | 30 min |
| Before each milestone gate (docs/10 §3) | checklist review | both | 20 min |
| Before the presentation | mock Q&A: Yoga asks Haikal about the pipeline, then swap | both | 45 min |
Decision log: `docs/DECISIONS.md` (date, decision, reason, owner). Anything that changes a split, metric, model family or scope goes there.

## 11. Working with Antigravity safely
- Always start sessions with: "Read AGENTS.md and the docs relevant to task <ID>."
- One task per agent session/branch. Ask for a plan first, approve, then implement.
- Never let the agent run destructive commands (rm -rf outside the repo, force push, history rewrite) without reading the command.
- After each session: `git diff --stat`, run tests, skim the diff, write the PR description yourself or edit the agent's.
- If the agent invents numbers or results, reject the change and reference AGENTS.md rule 1.

## 12. Release checklist (before submission)
- [ ] `main` builds and runs from a **fresh clone on Colab** following only the README (Haikal verifies — task P10-20).
- [ ] README has: abstract, architecture diagram, results summary (from summary.json), how to reproduce, how to run the demo, license/data notice, team.
- [ ] No data, checkpoints, recordings, tokens in git history (`git log --stat` skim, secret scan).
- [ ] Final model published as a Release asset; Docker image builds and runs locally.
- [ ] Tag `v1.0` on the submitted commit; the report links to that tag.
- [ ] docs/11 phase sign-off table is filled in by both of you.

<!-- ===== END docs/09_GITHUB_COLLABORATION.md ===== -->

<!-- ===== BEGIN docs/10_DETAILED_TASKS.md ===== -->

# 10 — SVARA Detailed Task Breakdown

Every task has an ID. Use the ID in the GitHub issue title, the branch name, commits and PR title (docs/09).
Total ≈ 150 tasks; each is small enough to finish in one sitting (S) up to a few sittings (XL).

## 1. Legend
- **Owner**: `Y` = Yoga, `H` = Haikal, `Y+H` = together. The *reviewer* is always the other person.
- **Effort** (rough, per person): S ≤ 1 h · M 2–4 h · L 5–8 h · XL > 8 h or long GPU/wall-clock time.
- **Depends**: task IDs that must be done first (`—` = none).
- **Done when**: the acceptance check. A task is not done until its check passes and the PR is merged.
- Dates are not fixed: fill the deadline in docs/04 and map W1..W7 to real dates in the first sync (task P0-12).
- Anything marked *(cond.)* depends on a result; decide and log it in `docs/DECISIONS.md`.
- Rule for all tasks: no fabricated numbers (AGENTS.md rule 1).

## 2. Weekly overview (7 working weeks + buffer; stretch or compress to the real deadline)
| Week | Yoga | Haikal |
|---|---|---|
| W1 | P0 setup, P1 audit/EDA code | P0 onboarding, P0-09, P0-15, P5-03..05 start recruiting, P10-01 report skeleton |
| W2 | P1 finish, P2 splits+baseline, P3-01..05 | P1-09..10 captions + Bab 3, P5-06 recordings, P1-12 literature notes |
| W3 | P3 main model runs, P4 configs start | P3-10 literacy training run, P5-06/07 recordings, P2-11 Bab 4.1 |
| W4 | P4 experiment runs, P6-01 noise setup | P5-08..11 noise + OOD clips, metadata, P4-13 Bab 4.2–4.4 |
| W5 | P4 finish, P6 robustness/fairness/reject, P7 export | P5-12, P6-02/09, P10-02/04 Bab 1–2, P10-05 references |
| W6 | P8 backend + web, P8-21 bug fixes | P8-15 copy, P8-19..22 QA, P9-01/02 field sessions |
| W7 | P9-03/04, P10 chapters 4–8 + abstract, README | P9-05, P10 video, P10-23 assemble docx, P10-20 fresh-clone test |
| Buffer | Q&A rehearsal, fixes, submission checks | same |
Parallel by design: Haikal's recordings (P5) run while Yoga builds the model; Haikal's QA (P8-19+) starts as soon as the demo exists.

## 3. Milestone gates (do not skip; review together, 20 min)
| Gate | When | Must be true |
|---|---|---|
| G1 | end W1 | repo + board + labels + issues exist; Colab runner works for both; audit script runs on real data; Haikal merged his practice PR; recruiting started |
| G2 | end W3 | splits A/B1 with passing leakage tests; sanity checks V-10..V-20 green (docs/11); M0 trained on A and B1 with `metrics.json`; time-per-epoch measured; experiment budget written in §14 |
| G3 | end W5 | all planned E-runs done or cut decisions logged; own set validated (≥10 speakers); noise/fairness/reject results exist; ONNX exported + parity-tested |
| G4 | end W6 | demo reachable (local + hosted or fallback); Lab page shows real results; QA pass 1 closed; no P0/P1 bugs open |
| G5 | end W7 | all report chapters drafted; video recorded; fresh-clone test passed; tag `v1.0` created; Q&A rehearsal done |
If a gate fails, stop adding scope and apply the cut list (docs/04).

## 4. Phase P0 — Setup (W1)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P0-01 | Create private GitHub repo `svara`, invite Haikal, enable branch protection on `main` | Y | S | — | Haikal can push a branch; direct push to main is blocked |
| P0-02 | Create/confirm GitHub account, install Git + editor, accept invite, set `user.name/email` | H | S | P0-01 | `git config --list` shows name/email; invite accepted |
| P0-03 | Commit this context package (AGENTS.md, docs/, .github/) to `main` | Y | S | P0-01 | files visible on GitHub |
| P0-04 | Run Prompt 0 and Prompt 1 (orient + scaffold) on `feat/P0-04-scaffold`; open PR | Y | M | P0-03 | docs/03 layout exists; ruff + pytest pass |
| P0-05 | Review the scaffold PR by following README setup; comment on unclear steps; approve | H | M | P0-04 | review comments posted; PR merged |
| P0-06 | Create Project board, labels and one issue per task from this file | Y | M | P0-03 | issues exist with owner + phase labels |
| P0-07 | Create Drive layout (docs/09 §1) and share with Haikal as editor | Y | S | — | both can read/write the folders |
| P0-08 | Put the FSC dataset (lecturer mirror) on Drive; list top-level contents | Y | S | P0-07 | wavs/, data/*.csv, README, license PDF present |
| P0-09 | Read the license PDF; write `docs/notes/license_summary.md` (what is allowed, what is not, how to cite) ≤ 10 lines | H | S | P0-08 | file merged; Yoga confirms nothing is misread |
| P0-10 | Build `notebooks/90_colab_runner.ipynb` (mount Drive, clone repo, install deps, copy data to `/content`, call check script) | Y | M | P0-04, P0-08 | runs top to bottom on Colab |
| P0-11 | Run the Colab runner from scratch with his own account; open issues for any failure | H | M | P0-10 | he reaches the data-check output; issues filed |
| P0-12 | Kickoff call: walk through docs/00; agree weekly sync slots; fill deadline, week-to-date map, names in docs/04 | Y+H | S | — | docs/04 filled; calendar events exist |
| P0-13 | Practice PR: edit `docs/notes/hello.md` via branch → PR → merge | H | S | P0-02 | merged PR in history |
| P0-14 | Create `docs/DECISIONS.md` and `reports/NOTES.md` with headers | Y | S | P0-03 | files exist |
| P0-15 | Write `docs/notes/project_in_my_words.md` (10 lines: what SVARA does, what the model is, what the data is, what we deliver); Yoga corrects | H | M | P0-12 | merged; no major misconceptions left |

## 5. Phase P1 — Data audit and EDA (W1–W2)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P1-01 | Implement `data/fsc.py` loader (reads real CSV headers, resolves wav paths) | Y | M | P0-04 | loads all three CSVs; unit test on a synthetic fixture |
| P1-02 | Implement `scripts/00_check_data.py` (docs/01 §2 checks 1–8) | Y | L | P1-01 | writes `reports/data_audit.json`; readable console summary |
| P1-03 | Generate `configs/intent_map.json` + slot vocabularies; assert 31 intents (log if not) | Y | S | P1-02 | file generated; count logged |
| P1-04 | Overlap analysis: speakers and transcripts shared between original train/valid/test | Y | M | P1-02 | numbers in audit JSON + bar chart |
| P1-05 | Implement `scripts/02_eda.py` (all figures/tables in docs/01 §3) | Y | L | P1-02 | PNGs (300 dpi) + CSVs under `reports/` |
| P1-06 | Choose `max_audio_seconds` from duration p99; write to `configs/base.yaml` | Y | S | P1-05 | value + justification in DECISIONS.md |
| P1-07 | Tests with tiny synthetic dataset fixture (no real data in git) | Y | M | P1-01 | `pytest -q` green |
| P1-08 | Run audit + EDA on the real dataset in Colab; commit small outputs via `exp/` PR | Y | S | P1-05, P0-10 | outputs in repo |
| P1-09 | Write Indonesian captions (2–3 sentences each) for each EDA figure: `reports/notes/eda_captions.md` | H | M | P1-08 | all figures captioned; Yoga verifies facts |
| P1-10 | Draft `docs/report/bab3.md` sections 3.1–3.2 (dataset description, license, statistics) | H | L | P1-08, P0-09 | draft merged; every number traceable to audit/EDA |
| P1-11 | Log any discrepancy vs the paper counts (23,132 / 3,118 / 3,793) and decisions in DECISIONS.md | Y | S | P1-08 | entry written |
| P1-12 | Start `docs/notes/literature_notes.md`: for each key paper (Lugosch 2019, wav2vec 2.0, SLURP, SpecAugment, split-evaluation paper, ESC-50): problem · method · dataset · result · why it matters to SVARA; continue through W3 | H | L | — | ≥ 6 entries, citations verified against the real papers |

## 6. Phase P2 — Splits and baselines (W2)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P2-01 | `splits.py`: Split A from the original CSVs + `intent_id` | Y | S | P1-03 | CSVs written under `data/processed/splits/A` |
| P2-02 | Split B1–B3 (unseen-utterance, per-intent hold-out, ≥ 2 train phrasings per intent) | Y | L | P2-01 | manifests list seeds/counts; constraint violations logged |
| P2-03 | Split C (speaker-grouped, stratified) *(cond. on P1-04: skip if original split is already speaker-disjoint and balanced)* | Y | M | P1-04 | built or decision logged |
| P2-04 | Leakage tests (transcript for B, speaker for C, intent coverage) | Y | M | P2-02 | tests green |
| P2-05 | `data/audio.py`: shared preprocessing (load, mono, 16 kHz check, normalize, crop/pad) | Y | M | P1-06 | unit tests on synthetic audio |
| P2-06 | Dataset + collate with length bucketing | Y | M | P2-05 | batch shapes tested |
| P2-07 | CRNN baseline model (docs/02 §2) | Y | M | P2-06 | forward-pass test |
| P2-08 | `scripts/train.py` (config-driven, CSV log, checkpoint per epoch, `--resume auto`) | Y | L | P2-07 | resume test: stop and continue gives same log continuity |
| P2-09 | Overfit-64-samples sanity check | Y | S | P2-08 | training accuracy → near 100% on 64 samples |
| P2-10 | First CRNN run on Split A (Colab) | Y | M | P2-09 | `metrics.json` + log in `reports/runs/` |
| P2-11 | Draft `docs/report/bab4.md` §4.1 (split protocols + why random file-level splits leak) after a 20-min walkthrough with Yoga | H | L | P2-04 | draft merged; Yoga confirms accuracy |

## 7. Phase P3 — Main model (W2–W3)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P3-01 | `models/wav2vec_slu.py`: encoder, frozen CNN, optional truncation, length-aware pooling, 3 heads + joint option | Y | L | P2-06 | forward test passes |
| P3-02 | Tests: pooling-mask equivalence, train/serve preprocessing parity | Y | M | P3-01 | tests green |
| P3-03 | Trainer upgrades: 2 LR groups, warmup + linear decay, fp16, early stopping on validation exact-match | Y | M | P2-08 | config run completes 1 epoch locally on CPU smoke (tiny subset) |
| P3-04 | `eval/metrics.py` (exact-match, per-slot, macro-F1, confusion, bootstrap CI, McNemar) + known-answer tests | Y | L | P2-01 | tests with hand-made predictions pass |
| P3-05 | `scripts/evaluate.py` writing `metrics.json`, `predictions.csv`, confusion CSVs | Y | M | P3-04 | works on CRNN run output |
| P3-06 | Benchmark 1 epoch of M0 on Colab; record time/epoch, GPU mem, max batch size; fill §14 | Y | S | P3-03 | numbers in §14 |
| P3-07 | Train M0 on Split A, seed 42 | Y | L | P3-06 | run folder complete |
| P3-08 | Train M0 on Split B1, seed 42 | Y | L | P3-06 | run folder complete |
| P3-09 | Sanity check vs literature (A) and A-vs-B gap; debug if far off (normalization, padding, LR, frozen CNN) | Y | M | P3-07, P3-08 | conclusion in DECISIONS.md |
| P3-10 | Literacy run: execute a tiny training (small subset, 1–2 epochs) from the Colab notebook; write `docs/notes/training_explained.md` (what is loss, epoch, validation, overfitting, checkpoint) in his own words | H | M | P3-03 | merged; Yoga corrects |
| P3-11 | Update the experiment budget (P4) with measured times | Y | S | P3-06 | §14 updated |
| P3-12 | Haikal walkthrough (30 min): Yoga shows one prediction end-to-end on a real wav | Y+H | S | P3-07 | Haikal can narrate it back |

## 8. Phase P4 — Experiments (W3–W5)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P4-01 | Configs for E2 (frozen), E3 (joint head), E4 (`-960h` init) | Y | M | P3-03 | configs load; 1-step smoke runs |
| P4-02 | Config + code for layer truncation E5 (12/8/6/4 layers) | Y | M | P3-01 | param counts logged correctly |
| P4-03 | `data/augment.py` (noise mixing, gain, speed) + configs E6/E7 | Y | M | P2-05, P6-01 (noise lists) | augmentation unit tests |
| P4-04 | `scripts/run_ablation.py` queue runner (skips finished runs) | Y | M | P3-05 | dry-run prints plan |
| P4-05 | Run E1–E4 on B1 (seed 42) | Y | XL | P4-01, P4-04 | metrics for each |
| P4-06 | Run E5–E7 on B1 (seed 42) | Y | XL | P4-02, P4-03 | metrics for each |
| P4-07 | Run E8 (A) and optionally E9 (C) *(cond. on P2-03)* | Y | L | P3-07 | metrics |
| P4-08 | Finalists (best accuracy, best small): 3 seeds × B1–B3 | Y | XL | P4-05, P4-06 | mean ± std computed |
| P4-09 | Paired significance tests between key variants | Y | M | P4-08 | table + p-values |
| P4-10 | Constrained decoding (E10) and invalid-intent rate analysis | Y | M | P3-05 | numbers reported |
| P4-11 | Error analysis: top-10 confusions, 5 qualitative failure cases (intent-level) | Y | M | P4-08 | `reports/tables/errors.csv` + notes |
| P4-12 | Pick deployment candidates (accuracy vs size/latency); log in DECISIONS.md | Y | S | P4-08 | decision recorded |
| P4-13 | Draft `docs/report/bab4.md` §4.2–4.4 (model, augmentation, training setup, metrics) from docs/02 and configs | H | L | P2-11, P3-07 | draft merged; Yoga checks every technical statement |
| P4-14 | Draft `docs/report/bab5.md` §5.1–5.2 (environment, tools, versions) from `env.txt` | H | M | P3-07 | draft merged |
| P4-15 | Weekly budget re-plan: GPU hours used/left, cut decisions | Y | S | each week | DECISIONS.md updated |

## 9. Phase P5 — Own recordings and noise clips (W1–W5; Haikal leads)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P5-01 | Phrase-sheet generator: one FSC phrasing per intent (31), printable | Y | M | P1-03 | `data/own_recordings/phrase_sheet.md` generated |
| P5-02 | Tooling: validator (readable, labels valid, metadata complete) + converter (any audio → 16 kHz mono WAV) | Y | M | P1-03 | both scripts + tests |
| P5-03 | Recruit ≥ 12 speakers (target 15), mixed gender if possible, anonymous IDs `own_s01…`; schedule sessions; keep the ID mapping privately | H | M | — | list of confirmed people |
| P5-04 | Consent: adapt the consent template (docs/), collect confirmation from every speaker; store in Drive `consent/` (private) | H | S | P5-03 | every speaker has a stored confirmation |
| P5-05 | Rehearse the recording protocol with 1–2 people (distance, device, quiet vs noisy); fix the guide | H | M | P5-01 | guide v1 in `docs/notes/recording_guide.md` |
| P5-06 | Quiet-room sessions: each speaker reads all 31 phrases | H | XL | P5-04, P5-05 | wavs in `wavs_raw/` |
| P5-07 | Noisy-condition sessions (TV/fan/kitchen sound), same 31 phrases | H | L | P5-06 | wavs in `wavs_raw/` |
| P5-08 | Household-noise clips: ≥ 20 clips ≥ 10 s (fan, TV, kitchen, rain, traffic…); label category per clip; store in `noise_own/` | H | M | — | files + `noise_own/categories.csv` |
| P5-09 | Convert + validate recordings; fill `metadata.csv` (no names); fix all validator errors | H | M | P5-02, P5-07 | validator exits clean |
| P5-10 | Spot-check 10% of clips for label/quality | Y | S | P5-09 | log of issues; fixed or dropped |
| P5-11 | OOD set: ≥ 100 clips of non-command speech + silence/room-tone clips | H | L | P5-02 | files + metadata |
| P5-12 | Draft `docs/report/bab3.md` §3.3 (own set: protocol, n, conditions, limitations) | H | M | P5-09 | draft merged |
Privacy gate: nothing from this phase except `metadata.csv` is ever committed (docs/09 §9).

## 10. Phase P6 — Robustness, fairness, reject (W4–W5)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P6-01 | Get ESC-50 (check license), partition categories into train-noise vs test-noise; include own clips; document lists | Y | M | P5-08 | `configs/noise_split.yaml` |
| P6-02 | Agree with Yoga which household categories go to test noise; update categories file | Y+H | S | P6-01 | decision logged |
| P6-03 | `eval/robustness.py` + script: SNR 20/10/5/0 dB for key models; curves | Y | L | P6-01, P4-08 | CSV + `fig_noise_snr.png` |
| P6-04 | `eval/fairness.py`: per-group accuracy with CIs from `speaker_demographics.csv` | Y | M | P3-05 | CSV + figures; small groups flagged |
| P6-05 | `eval/reject.py`: confidence scores, validation threshold, risk–coverage, ECE (+ optional temperature scaling) | Y | L | P3-05 | curve + threshold JSON |
| P6-06 | `scripts/eval_own_set.py`: overall, quiet vs noisy, per device | Y | M | P5-09, P3-07 | table + figure |
| P6-07 | OOD false-accept evaluation at several thresholds | Y | M | P5-11, P6-05 | table |
| P6-08 | Write the chosen threshold into `configs/serve.yaml` (validation-selected only) | Y | S | P6-05 | file merged |
| P6-09 | Review all new figures/tables; write Indonesian captions and "what to notice" lines | H | M | P6-03..07 | captions merged; Yoga verifies |
| P6-10 | Draft `docs/report/bab6.md` descriptive text for noise, fairness, own set, reject (numbers from Yoga's tables) | H | L | P6-09 | draft merged |

## 11. Phase P7 — Export and benchmark (W5)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P7-01 | `export/to_onnx.py` (dynamic axes, normalized-waveform input) | Y | M | P4-12 | ONNX loads in onnxruntime |
| P7-02 | Parity tests PyTorch vs ONNX (20+ clips, different lengths) | Y | M | P7-01 | argmax identical, max diff within tolerance |
| P7-03 | `export/quantize.py` int8; accuracy delta on test + own set | Y | M | P7-02 | delta documented; keep fp32 if drop is notable |
| P7-04 | `export/benchmark.py`: size, p50/p95 latency, RTF, RAM for fp32/int8/truncated | Y | M | P7-03 | `reports/tables/systems.csv` |
| P7-05 | Choose deployment model; write `models/svara.meta.json` (labels, intent map, threshold, normalization) | Y | S | P7-04, P6-08 | meta JSON validates |
| P7-06 | Publish model as a GitHub Release asset (or HF Hub); document download step | Y | S | P7-05 | download URL works from a clean machine |
| P7-07 | Draft `docs/report/bab7.md` §7.1–7.2 (export, quantization, systems table) | H | M | P7-04 | draft merged |

## 12. Phase P8 — Backend, web demo and QA (W5–W6)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P8-01 | Backend schemas + config loading (meta JSON, serve.yaml) | Y | M | P7-05 | schema tests |
| P8-02 | `inference.py` (shared preprocessing, onnxruntime session, confidence, reject) | Y | M | P8-01 | parity test with training preprocessing |
| P8-03 | `device_state.py` + `device_map.yaml`; every intent has a handler or documented fallback | Y | M | P1-03 | pure-function tests for all intents |
| P8-04 | `app.py` endpoints (`health`, `model-info`, `predict`, `results`, `devices`, `reset`) + error codes | Y | M | P8-02, P8-03 | TestClient tests |
| P8-05 | Backend tests (synthetic WAV, long/short audio, bad file) | Y | M | P8-04 | green |
| P8-06 | README: curl example for each endpoint | Y | S | P8-04 | verified by Haikal |
| P8-07 | Web scaffold (Vite + React + TS), design tokens, fonts | Y | M | docs/06 | `npm run dev` shows empty shell |
| P8-08 | API client, routes `/`, `/lab`, `/about`, top bar + status chip | Y | M | P8-07, P8-04 | status chip reflects server state |
| P8-09 | Floor plan SVG + device components + state-driven animations | Y | L | P8-08 | manual toggles work |
| P8-10 | Push-to-talk: mic permission, recording, 16 kHz WAV encoding, upload fallback | Y | L | P8-08 | works in Chrome desktop + Android Chrome |
| P8-11 | Result panel (3 slot rows, confidences, status line) + event log | Y | M | P8-09, P8-10 | shows real prediction |
| P8-12 | Reject state + unsupported-but-recognized ribbon (`bring`, `change language`) | Y | M | P8-11 | both states demonstrable |
| P8-13 | Lab view: sections 1–9 reading `/api/results`; table toggle; empty states | Y | L | P10-13 | no hardcoded numbers |
| P8-14 | About page (content from docs/06 §6; copy by Haikal) | Y | S | P8-15 | page live |
| P8-15 | Review and write Indonesian UI copy (labels, errors, example phrases, About text); check tone and typos | H | M | P8-07 | copy PR merged |
| P8-16 | Accessibility + responsive pass (keyboard, focus, contrast, 360–1440 px) | Y | M | P8-13 | Lighthouse accessibility measured |
| P8-17 | Dockerfile (multi-stage) + local run script | Y | M | P8-05, P8-13 | `docker run` serves app without internet/GPU |
| P8-18 | Deploy to the chosen host (verify free-tier limits first); HTTPS; document fallback | Y | M | P8-17 | public URL works on a phone |
| P8-19 | QA plan: matrix of devices × browsers × rooms × noise conditions + scripted test cases (docs/notes/qa_plan.md) | H | L | P8-07 | plan reviewed by Yoga |
| P8-20 | QA pass 1: execute plan, file bug issues with steps, screenshot, severity | H | L | P8-18, P8-19 | all cases run; bugs filed |
| P8-21 | Fix bugs by severity (P0/P1 first) | Y | M | P8-20 | all P0/P1 closed |
| P8-22 | QA pass 2: regression on fixed bugs | H | M | P8-21 | confirmed in issues |
| P8-23 | Measure end-to-end latency in the browser (stop → result) separately from model latency | H | M | P8-18 | table in notes; included in Bab 7 |

## 13. Phase P9 — Field test (W6–W7)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P9-01 | Plan ≥ 3 live sessions with ≥ 5 *new* people (not in the recording set), different rooms/devices; logging sheet columns: person id, device, room, noise, spoken command, result, correct?, perceived latency, comment | H | M | P8-18 | plan + sheet ready |
| P9-02 | Run the sessions and fill the log (consent first) | H | L | P9-01 | ≥ 100 logged trials |
| P9-03 | Analyze log: accuracy, reject rate, common failures | Y | M | P9-02 | table + 3 findings |
| P9-04 | Fix the top 3 usability/model-config issues (e.g., threshold, trimming, UI hint) | Y | M | P9-03 | merged; QA re-check |
| P9-05 | Capture screenshots (390 px and 1280 px) and short screen recordings for the report and video | H | M | P9-04 | files in Drive/`docs/report/img` |
| P9-06 | Freeze model and demo; tag `rc1` | Y | S | P9-04 | tag exists |
| P9-07 | Draft `docs/report/bab7.md` §7.3–7.5 (API, UI, field test) from logs and screenshots | H | L | P9-03 | draft merged |

## 14. Compute budget (fill after P3-06; keep honest)
| Item | Measured / planned |
|---|---|
| Time per epoch (M0, Colab GPU) | TODO |
| Epochs per run (typical) | TODO |
| Runs planned (E1–E10 × seeds × splits) | TODO |
| Total GPU hours estimated | TODO |
| Cut triggered? (docs/04 cut list) | TODO |

## 15. Phase P10 — Report, video, release (W2–W7)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P10-01 | Convert the course template headings into `docs/report/` skeleton (one Markdown per chapter + front matter), keep exact headings | H | M | P0-12 | skeleton merged |
| P10-02 | Write Bab 1 (background, problem, objectives, scope, contributions, structure) | H | L | P10-01 | draft merged |
| P10-03 | Finalize Bab 3 (merge 3.1–3.3 + system/architecture + hardware/software requirements) | H | L | P1-10, P5-12 | chapter complete; figures referenced |
| P10-04 | Write Bab 2 (literature + research gap) from literature notes; Yoga reviews the gap claims against the evidence | H | L | P1-12 | draft merged; every claim cited |
| P10-05 | Build the IEEE reference list; verify each entry (authors, venue, year, pages/DOI) against the real source; remove any unverifiable entry | H | M | P10-04 | `docs/references.md` clean |
| P10-06 | Finalize Bab 4 (methodology) | Y | M | P4-13 | reviewed by Haikal for clarity |
| P10-07 | Finalize Bab 5 (implementation) | Y | M | P4-14 | complete |
| P10-08 | Finalize Bab 6 (results, ablation, confusion matrix, discussion, limitations) from `RESULTS_INDEX.md` | Y | XL | P6-10, P4-11 | each claim points to a table/figure |
| P10-09 | Finalize Bab 7 (deployment + field test) | Y | M | P9-07, P7-07 | complete |
| P10-10 | Write Bab 8 (answers to RQ1–RQ7, limitations, future work) | Y | M | P10-08 | RQs each answered with evidence |
| P10-11 | Write Abstrak (last) | Y | S | P10-10 | ≤ limit in template |
| P10-12 | Appendices: key code listings, training log excerpts, GitHub link (tag), video link | H | M | P10-09 | appendix complete |
| P10-13 | `scripts/make_report_assets.py` (+ `summary.json`, `RESULTS_INDEX.md`) | Y | L | P6-08, P7-04 | fails loudly if runs missing |
| P10-14 | Check every figure/table is referenced and every reference is cited; unify captions/numbering | H | M | P10-12 | checklist done |
| P10-15 | Video script (problem → data → model → results → demo → limits); check the course's length requirement | H | M | P9-04 | script approved by Yoga |
| P10-16 | Record screen + voice-over (clean audio, demo takes in a quiet room) | H | L | P10-15, P9-06 | raw recordings |
| P10-17 | Edit video, captions optional, export | H | L | P10-16 | final file |
| P10-18 | Technical review of the video (Yoga checks every claim) | Y | S | P10-17 | corrections applied |
| P10-19 | Mock Q&A both ways (45 min) using the question list in §16 | Y+H | M | P10-08 | both answer all questions without notes |
| P10-20 | Fresh-clone test: new Colab session, follow README only, reproduce one small run + the demo locally | H | M | P9-06 | issues filed/fixed; test passes |
| P10-21 | Final README (abstract, diagram, results from summary.json, reproduce steps, data notice, team) | Y | M | P10-13 | merged |
| P10-22 | Assemble the final Word/PDF from the Markdown chapters using the course template; Yoga reviews | H | L | P10-11, P10-14 | document opens correctly; numbering/figures OK |
| P10-23 | Tag `v1.0`; verify the submission checklist (docs/09 §12) | Y+H | S | P10-20, P10-22 | tag + checklist ticked |

## 16. Haikal minimum-literacy questions (answer without notes before the presentation)
1. What is the input and the output of the model? What is an intent and what are slots?
2. What does the dataset contain, how many speakers/phrases/intents, and why can't we split it randomly by file?
3. What is wav2vec2 and what does fine-tuning mean here?
4. What is training vs validation vs test, and why do we choose thresholds on validation only?
5. Why is accuracy on the original split not enough? What does the unseen-utterance split show?
6. What is the ablation study and what did we learn from it?
7. How do we handle noise and non-command speech? What does the reject threshold do?
8. What did our own recordings test and what are their limits?
9. How does the demo work from microphone to the lamp turning on?
10. What are the limitations of the project?
Yoga keeps a matching list for himself on the demo/QA side (how was QA done, what bugs were found, what did the field test show).

## 17. Risk triggers and fallback rules
| Trigger | Fallback |
|---|---|
| M0 on Split A far below literature by end of W3 | stop adding experiments; debug preprocessing/padding/LR first (P3-09), ask Haikal to re-check the data path |
| Colab quota/disconnects block runs | smaller batch, fewer epochs, run ablations with one seed, cut E9/E10; document |
| Fewer than 10 speakers recorded by end of W4 | extend recruiting with classmates; keep the analysis but label it a pilot with n |
| int8 hurts accuracy | ship fp32 or the truncated model |
| Hosting limits | local Docker + recorded video as the primary demo path |
| Web slips | Gradio fallback (`app_gradio.py`) with the same ONNX model |
| Haikal blocked on Git/Colab | 20-min screenshare with Yoga; never wait more than a day |
| Scope creep | anything not in docs/10 requires a DECISIONS.md entry |

## 18. Verification add-on tasks (from the audit; see docs/11)
| ID | Task | Owner | Effort | Depends | Done when |
|---|---|---|---|---|---|
| P1-13 | Add audit checks 9–11 (transcript→intent consistency, duplicate hashes, folder/speaker consistency, phrasings per intent, imbalance ratios) | Y | M | P1-02 | V-01..V-04, V-07 recorded in `data_audit.json` |
| P1-14 | Listening check: play 30 random clips with their labels, record mismatches in `reports/notes/listening_check.md` | H | M | P1-08 | V-05 done; Yoga reviews mismatches |
| P2-12 | Pipeline sanity tests: initial-loss check, shuffled-label control, trainable-param count, gradient check, eval determinism, padding invariance | Y | L | P2-08, P3-01 | V-11, V-13, V-15..V-18 green |
| P2-13 | Batch inspection: print stats and listen to one augmented batch (save 5 wavs to Drive for Haikal to listen) | Y+H | S | P2-06, P4-03 | V-10 logged |
| P3-13 | LR range test for head and encoder groups; plot saved | Y | M | P3-03 | V-20 done; chosen range in DECISIONS.md |
| P3-14 | Hyperparameter search per docs/11 §4 (B1 validation, one seed, 8–12 runs, same budget for CRNN) | Y | XL | P3-13, P3-08 | `reports/tables/hparam_search.csv`; config M0 frozen |
| P3-15 | Reproducibility utilities (seeding incl. DataLoader workers, deterministic flags, run metadata) + `requirements.lock` after first working Colab run | Y | M | P3-03 | V-17/§6 satisfied; lock file committed |
| P3-16 | Training curves figure per run (train vs val loss, exact-match) and per-class recall in `metrics.json` | Y | M | P3-05 | figures generated automatically |
| P4-16 | Maintain the test-access log in `reports/NOTES.md` (every test evaluation: run id, split, why) | Y | S | P3-07 | log complete at every gate |
| P4-17 | *(optional)* Pooling ablation: mean vs attentive pooling | Y | M | P3-14 | result or cut decision |
| P7-08 | Silence-trimming consistency experiment: with vs without client-side trimming | Y | M | P7-01 | V-42 reported; one policy chosen |
| P8-24 | Web-captured evaluation set: ≥ 100 clips recorded through the demo by consenting testers (local debug flag, not stored in prod) | H | L | P8-18 | V-43 data ready |
| P8-25 | Evaluate the web-captured set vs phone recordings; test browser audio settings (echo cancel / noise suppression / AGC) | Y | M | P8-24 | V-43, V-44 reported |
| P10-24 | Independent re-run: Haikal runs `evaluate.py` on the final checkpoint and gets the same metrics | H | M | P9-06 | match within tolerance; logged |
| P10-25 | Claims audit table `reports/CLAIMS_AUDIT.md` + regenerate assets from a clean checkout + hand-check 5 numbers | Y+H | L | P10-08 | every number/comparison in the report is traceable |
| P10-26 | Verify the "known unverified assumptions" list (docs/11 §10); update docs where reality differs | Y | M | P3-07 | each item marked confirmed/changed in DECISIONS.md |

## 19. Minimum viable path (if time is tight, this is the part that must be excellent)
| Phase | Must (MVP) | Should | Could |
|---|---|---|---|
| P0 | P0-01..P0-04, P0-07, P0-08, P0-10, P0-12 | P0-05, P0-06, P0-09, P0-11, P0-13..P0-15 | — |
| P1 | P1-01..P1-06, P1-08, P1-13 | P1-09..P1-12, P1-14 | — |
| P2 | P2-01, P2-02, P2-04..P2-10, P2-12 | P2-03 (cond.), P2-11, P2-13 | — |
| P3 | P3-01..P3-09, P3-13..P3-16 | P3-10..P3-12 | — |
| P4 | P4-01, P4-02, P4-04..P4-06 (E1, E2, E3, E5, E6), P4-07 (E8), P4-11, P4-12, P4-16 | P4-08, P4-09, P4-13..P4-15 | P4-10, P4-17, E9 |
| P5 | P5-01..P5-07, P5-09, P5-10 (≥ 10 speakers) | P5-08, P5-11, P5-12 | — |
| P6 | P6-01, P6-03, P6-05, P6-06, P6-08 | P6-04, P6-07, P6-09, P6-10 | — |
| P7 | P7-01..P7-05 | P7-06..P7-08 | int8 if accuracy drops |
| P8 | P8-01..P8-05, P8-07..P8-12, P8-17, P8-19..P8-22 | P8-13..P8-16, P8-18, P8-23..P8-25 | Lab polish |
| P9 | P9-01..P9-03, P9-06 | P9-04, P9-05, P9-07 | — |
| P10 | P10-01..P10-11, P10-13, P10-15..P10-17, P10-20, P10-22, P10-23, P10-25 | P10-12, P10-14, P10-18, P10-19, P10-21, P10-24, P10-26 | — |
Rule: finish all "Must" rows to a verified standard before starting any "Could". Quality of verified results beats quantity of experiments.

<!-- ===== END docs/10_DETAILED_TASKS.md ===== -->

<!-- ===== BEGIN docs/11_DL_VERIFICATION_CHECKLIST.md ===== -->

# 11 — Deep Learning Verification Checklist and Experimental Discipline

Purpose: make sure our numbers are **trustworthy**, not just high. Deep learning fails silently: a bug usually produces a
plausible-looking accuracy. This document lists the checks that catch the common silent failures. Each check has an ID
(`V-xx`) and is scheduled by tasks in docs/10 §18. Record outcomes in `reports/NOTES.md` (check, date, result, who).

## 1. Test-set discipline (hard rules)
1. **Selection uses validation only**: hyperparameters, variant choice, early stopping, confidence threshold, deployment model.
2. The test set is evaluated **only** by `scripts/evaluate.py` on a config that is already frozen. Never edit configs after seeing test numbers
   to "improve" them; if you must, treat it as a new experiment and say so in the report.
3. Keep a **test-access log** in `reports/NOTES.md`: date, run id, split, why. The report states how many times each test set was used.
4. Ablation tables show validation and test; the *ranking used for decisions* comes from validation (mean over B1–B3 draws where available).
5. Own recordings, OOD clips, field-test log: never used for training, tuning or threshold selection (except the explicit, labelled adaptation stretch).
6. Normalization is per-utterance (no dataset statistics), so no train/test statistic leakage; if any dataset-level statistic is ever used (e.g., mel mean/std for the CRNN), compute it on train only.

## 2. Data checks (before any training) — owner Y unless noted
| ID | Check | Pass criterion |
|---|---|---|
| V-01 | Each `transcript` maps to exactly one `(action, object, location)` | zero conflicts; otherwise list and resolve |
| V-02 | Each intent has ≥ 2 distinct phrasings in every training split | holds for A, B1–B3 (log exceptions) |
| V-03 | Duplicate audio detection by content hash across splits | none; or duplicates documented |
| V-04 | Filename/speaker consistency: wav path speaker folder == `speakerId` column | 100% |
| V-05 | **Listening check**: 30 random clips played with their labels *(H)* | ≤ 1 mismatch; mismatches logged in `reports/notes/listening_check.md` |
| V-06 | Duration outliers, clipping, near-silent clips | listed; decision on exclusion logged |
| V-07 | Class distribution per slot/intent; imbalance ratio | decision logged: weighting yes/no, with reason |
| V-08 | Splits: leakage assertions (docs/01 §4) | tests green |

## 3. Pipeline sanity checks (before the long runs)
| ID | Check | Expected | If it fails |
|---|---|---|---|
| V-10 | Print and **listen to** one training batch after augmentation (waveform stats, labels) | audio plausible, labels match | fix augment/collate |
| V-11 | Initial loss per head before training (random heads) | ≈ ln(C): action ln6≈1.79, object ln14≈2.64, location ln4≈1.39, joint ln31≈3.43 | wrong init, wrong loss, label index bug |
| V-12 | Overfit 64 samples to ~100% train accuracy | yes within a few hundred steps | model/optimizer/label bug |
| V-13 | **Shuffled-label control**: train on shuffled labels | validation accuracy ≈ chance (about 1/C) | any higher = leakage or bug |
| V-14 | **Input control**: evaluate trained model on silence and on pure noise | not confidently correct; no systematic class | model ignores audio or has bias |
| V-15 | Trainable-parameter count equals what the config says (frozen CNN, truncation, frozen encoder) | matches | freezing bug |
| V-16 | Gradient check: nonzero grad norms for heads and unfrozen layers, zero for frozen modules | yes | detached graph / wrong requires_grad |
| V-17 | Eval-mode determinism: same input twice → identical logits | yes | dropout/augmentation still on |
| V-18 | Padding invariance: same clip padded differently gives same logits (within tolerance) | yes | pooling mask bug |
| V-19 | No NaN/inf in loss or grads; gradient clipping (default max-norm 1.0) enabled; fp16 loss scaling active | yes | lower LR, check fp16 path, fall back to fp32 for the run |
| V-20 | LR range test (few hundred steps, LR 1e-6 → 1e-3, separate sweeps for head and encoder) | clear stable region | pick LR from region; document plot |

## 4. Hyperparameter tuning protocol (fair and small)
- Tune on **Split B1 validation** only, **one seed**, small budget (target 8–12 runs), same budget for the CRNN baseline so the comparison is fair.
- Search space (starting point): encoder LR {1e-5, 3e-5, 1e-4}; head LR {1e-3}; dropout {0.1, 0.3}; label smoothing {0, 0.05};
  effective batch {16, 32}; SpecAugment mask prob {0.0, 0.05}; pooling {mean, attentive} (optional).
  Change one or two things at a time; log everything in `reports/tables/hparam_search.csv`.
- Freeze the chosen config, tag it `M0`, then run all variants with the same settings except the factor under study.
- Report the search table in an appendix, and state that test sets were not used for tuning.

## 5. Training monitoring
- Always plot train vs validation loss and exact-match per epoch (`reports/figures/curves_<run>.png`).
- Inspect the train–validation gap; overfitting symptoms (val loss rising while train falls) → earlier stopping, more regularization/augmentation.
- Look at **per-class recall** each run to catch classes the model ignores (macro-F1 and per-intent recall in `metrics.json`).
- Record best epoch, total time, GPU type and peak memory in `env.txt`/`metrics.json`.

## 6. Reproducibility
- Seed Python, NumPy, PyTorch (CPU and CUDA); DataLoader `worker_init_fn` and generator seeded.
- Enable deterministic algorithms where feasible and **state clearly** that some GPU ops remain non-deterministic: this is why we report mean ± std over 3 seeds.
- Pin dependency versions: after the first working Colab run, `pip freeze > requirements.lock`; install from it in Colab.
- Each run stores: resolved config, git commit, split-manifest hash, library versions, GPU name.
- **Independent re-run (H)**: a teammate re-runs `evaluate.py` on the final checkpoint and obtains the same metrics (task P10-24).

## 7. Evaluation correctness
| ID | Check | Pass criterion |
|---|---|---|
| V-30 | Metrics code tested on hand-made predictions with known answers | tests green |
| V-31 | Confusion matrix row sums equal class supports; total equals test size | yes |
| V-32 | Exact-match computed two independent ways (per-slot AND vs joint intent id) | identical (invalid slot combos counted as wrong) |
| V-33 | Bootstrap CI contains the point estimate; width plausible for n | yes |
| V-34 | Majority/prior baseline reported next to every model table | yes |
| V-35 | Report per-split sample counts and number of distinct phrases in each test set | in every results table |

## 8. Deployment-path checks
| ID | Check | Pass criterion |
|---|---|---|
| V-40 | ONNX vs PyTorch parity on 20+ clips and several lengths | same argmax; small max-diff |
| V-41 | int8 vs fp32 accuracy delta on test and own set | reported; ship int8 only if the drop is small |
| V-42 | Silence-trimming consistency: evaluate with and without the client-side trimming used in the web app | difference reported; choose one policy for train and serve |
| V-43 | Browser-captured evaluation: ≥ 100 clips recorded through the actual web demo (consenting testers, local debug flag) compared with the same phrases from phone recordings | accuracy gap measured and explained |
| V-44 | Browser settings: echo cancellation / noise suppression / auto-gain effect on accuracy (on vs off) | choice documented |
| V-45 | Out-of-domain false-accept check at the shipped threshold | reported in Bab 6/7 |

## 9. Claims audit (before submitting the report)
- Build a table `claim → figure/table/file → run ids` in `reports/CLAIMS_AUDIT.md`. Every sentence in the report that contains a number or a comparison appears here.
- Regenerate all figures and tables with `make_report_assets.py` from a clean checkout; diff against what is in the report.
- Cross-check 5 random numbers against the raw `metrics.json` by hand.
- Search the report for superlatives and causal claims ("proves", "because") and soften to what the evidence supports.

## 10. Known unverified assumptions (this plan was written from knowledge, no code has been executed yet)
Treat each as a hypothesis to confirm in the first sprint; if one is wrong, fix the doc and log it in DECISIONS.md.
1. Exact HuggingFace checkpoint names (`facebook/wav2vec2-base`, `-960h`, XLS-R) and their current availability.
2. wav2vec2-base behavior with padding and without `attention_mask`, and the frame-length helper used for pooling.
3. ONNX export of the full model (with pooling) with dynamic time axis, and onnxruntime int8 dynamic quantization keeping accuracy.
4. Time per epoch and memory on the Colab GPU actually assigned (varies by session).
5. That the lecturer's GitHub mirror equals the official distribution (check counts, headers, license PDF).
6. Numbers taken from literature (split sizes, ~99% on original split, ~78–88% on unseen-utterance split) come from the sources we found; confirm against the original papers before quoting.
7. Whether the original split is speaker-disjoint (decides whether Split C is useful).
8. ESC-50 and FSC license terms for our use; hosting free-tier limits; Antigravity rules-file limits at the time you use it.
9. Initial hyperparameters (LR, batch, epochs, mask probabilities) are common starting points, not tuned values.

## 11. Phase sign-off table (fill in as you go, both initials)
| Phase | Checks that must be green | Y | H | Date |
|---|---|---|---|---|
| P1 | V-01..V-08 | | | |
| P2 | V-10..V-13, V-15..V-18 | | | |
| P3 | V-11..V-20, V-30..V-35 | | | |
| P4 | tuning protocol §4, test-access log §1 | | | |
| P6 | V-14, V-45, threshold chosen on validation only | | | |
| P7 | V-40..V-42 | | | |
| P8–P9 | V-43..V-44, field-test log complete | | | |
| P10 | claims audit §9, independent re-run §6 | | | |

<!-- ===== END docs/11_DL_VERIFICATION_CHECKLIST.md ===== -->
