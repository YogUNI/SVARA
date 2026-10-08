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
