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
