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
| P8.5 | FPP 3D Walkthrough (Game Mode) | 0.5 w | Y (PBR, Controller, Voice HUD) | P8 |
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

## P8.5 — FPP 3D Walkthrough (Game-Style Interactive Smart Home)
Tasks per docs/13: Kinematic first-person controller (WASD + Mouse 360° PointerLock); eye-level perspective (1.65m);
dynamic head bobbing; collision detection; realistic PBR interior rooms (kitchen, bedroom, washroom, living room);
real point & spot lights; hands presence with smartwatch; Spacebar push-to-talk voice hook; in-game HUD & mini-map.
Done when: user can walk smoothly inside the 3D house at 60 FPS, talk via Spacebar, and observe devices reacting in first-person POV.

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
