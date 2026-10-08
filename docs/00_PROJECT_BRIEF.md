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
