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
| P3-17 | HF-native cross-check baseline: `Wav2Vec2ForSequenceClassification` with the 31-way joint head, same split and seed as M0 | Y | M | P3-07 | result next to M0; large unexplained gap triggers a bug hunt in our custom model |
| P3-18 | Test preprocessing parity against `Wav2Vec2FeatureExtractor` normalization on sample clips (docs/12 §4.6) | Y | S | P2-05 | test green; tolerance documented |
| P4-18 | *(optional)* E11 layer-weighted sum + E14 per-layer probes; plot learned layer weights | Y | M | P4-06 | figure + short interpretation, or cut decision |
| P7-09 | *(optional)* HF Hub model card + upload of the final model (after checking FSC license terms) | Y | S | P7-06 | model page live, card uses numbers from `summary.json` only |
| P10-27 | Draft Bab 2/4 text "Transformer SLU and wav2vec2" from docs/12 §8 (what is HF-based vs custom); Yoga verifies every technical claim | H | M | P10-04 | paragraph merged; consistent with the code |

## 19. Minimum viable path (if time is tight, this is the part that must be excellent)
| Phase | Must (MVP) | Should | Could |
|---|---|---|---|
| P0 | P0-01..P0-04, P0-07, P0-08, P0-10, P0-12 | P0-05, P0-06, P0-09, P0-11, P0-13..P0-15 | — |
| P1 | P1-01..P1-06, P1-08, P1-13 | P1-09..P1-12, P1-14 | — |
| P2 | P2-01, P2-02, P2-04..P2-10, P2-12 | P2-03 (cond.), P2-11, P2-13 | — |
| P3 | P3-01..P3-09, P3-13..P3-16, P3-18 | P3-10..P3-12, P3-17 | — |
| P4 | P4-01, P4-02, P4-04..P4-06 (E1, E2, E3, E5, E6), P4-07 (E8), P4-11, P4-12, P4-16 | P4-08, P4-09, P4-13..P4-15 | P4-10, P4-17, E9 |
| P5 | P5-01..P5-07, P5-09, P5-10 (≥ 10 speakers) | P5-08, P5-11, P5-12 | — |
| P6 | P6-01, P6-03, P6-05, P6-06, P6-08 | P6-04, P6-07, P6-09, P6-10 | — |
| P7 | P7-01..P7-05 | P7-06..P7-08 | int8 if accuracy drops |
| P8 | P8-01..P8-05, P8-07..P8-12, P8-17, P8-19..P8-22 | P8-13..P8-16, P8-18, P8-23..P8-25 | Lab polish |
| P9 | P9-01..P9-03, P9-06 | P9-04, P9-05, P9-07 | — |
| P10 | P10-01..P10-11, P10-13, P10-15..P10-17, P10-20, P10-22, P10-23, P10-25 | P10-12, P10-14, P10-18, P10-19, P10-21, P10-24, P10-26 | — |
Rule: finish all "Must" rows to a verified standard before starting any "Could". Quality of verified results beats quantity of experiments.
