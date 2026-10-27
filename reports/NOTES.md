# Running Notes — SVARA

Record of technical observations, experiment updates, gate checks, and test-access logs.

## 1. Test-Access Log (docs/11 §1)
| Date | Run ID | Split Evaluated | Purpose / Rationale | Evaluator |
|---|---|---|---|---|

## 2. General Observations and Surprises
- 2026-10-08: Setup started under P0-04.
- 2026-10-27: Phase 1 (Signal Hygiene) verified 100%. Butterworth Bandpass (80-7500 Hz), Adaptive RMS (-20 dBFS), Dual-Threshold VAD (150ms buffer) completed. 13 unit tests in `tests/test_audio.py` passed. Preprocessing latency 1.0 ms.
- 2026-10-27: Phase 2 (Indonesian NLP Corpus Engineering) verified 100%. Expanded colloquial lexicon to 317+ unique vocabulary and evaluated 5.8 million candidate permutations. Generated 4,495 phrases (exactly 145 phrases across all 31 intents, zero class imbalance, Shannon entropy ratio 1.000000). 7-dimension audit passed with 0 duplicates and 0 cross-intent collisions (`reports/corpus_linguistic_audit.json` & `reports/corpus_linguistic_audit.md`). 11 unit tests in `tests/test_corpus.py` passed (total 24 tests passed).
