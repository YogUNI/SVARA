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
