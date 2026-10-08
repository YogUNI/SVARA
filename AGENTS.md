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
| docs/12_TRANSFORMER_SLU_HF_TECH_SPEC.md | implementing or explaining the Transformer SLU / wav2vec2 / HuggingFace part (read before models/) |
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
