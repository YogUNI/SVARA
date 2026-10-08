# SVARA — Smart Voice Assistant for Residential Automation

End-to-end spoken language understanding for smart-home commands. A fine-tuned wav2vec 2.0 model maps a short spoken
English command to `(action, object, location)` and drives a simulated home through a web demo.
University project, Deep Learning course (Universitas Mercu Buana). Team: Yoga, Haikal.

> Status: **under construction.** This README is replaced by the final version in task P10-21. Do not copy numbers into it
> unless they come from `reports/metrics/summary.json`.

## Start here
- Agents and contributors: read [`AGENTS.md`](AGENTS.md), then the docs in [`docs/`](docs/) in order (00 → 10).
- Task list with IDs: [`docs/10_DETAILED_TASKS.md`](docs/10_DETAILED_TASKS.md). Collaboration rules: [`docs/09_GITHUB_COLLABORATION.md`](docs/09_GITHUB_COLLABORATION.md).

## Data
Not included in this repository. Fluent Speech Commands is distributed under its own license (PDF in the dataset folder).
Place it under `data/raw/fluent_speech_commands_dataset/` (see `docs/03`).

## Quick start (filled in as the code lands)
```
pip install -r requirements.txt
python scripts/00_check_data.py --root data/raw/fluent_speech_commands_dataset
# training, evaluation, export and demo commands: see AGENTS.md "Commands"
```
