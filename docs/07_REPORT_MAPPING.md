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
