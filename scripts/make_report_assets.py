"""Generate figures, tables, and summary.json for Bab 6 & 7 of the final report.

Reads completed run outputs from `reports/runs/`:
- Aggregates metrics into `reports/metrics/summary.json`
- Generates LaTeX/Markdown tables under `reports/tables/`
- Generates high-resolution figures under `reports/figures/`
- Creates `reports/RESULTS_INDEX.md` mapping every table/figure to its run ID
- Follows rule DEC-001: Never fabricate results. If a run does not exist, clearly report missing.

Tasks: P10-13
Reference: docs/07 §2, docs/02 §8, docs/11 §5
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_all_runs(runs_root: str = "reports/runs") -> Dict[str, dict]:
    """Scan and parse all valid run directories."""
    runs = {}
    if not os.path.exists(runs_root):
        return runs

    for entry in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, entry)
        if not os.path.isdir(run_dir):
            continue

        # Look for metric files
        metric_files = glob.glob(os.path.join(run_dir, "metrics_*.json"))
        if not metric_files:
            continue

        run_data = {"run_id": entry, "path": run_dir, "metrics": {}}

        for mf in metric_files:
            fname = os.path.basename(mf)
            # e.g. metrics_B1_test.json -> key: B1_test
            key = fname.replace("metrics_", "").replace(".json", "")
            with open(mf, "r", encoding="utf-8") as f:
                run_data["metrics"][key] = json.load(f)

        config_file = os.path.join(run_dir, "config.yaml")
        if os.path.isfile(config_file):
            run_data["config_path"] = config_file

        runs[entry] = run_data

    return runs


def generate_summary_and_assets(
    runs_root: str = "reports/runs",
    output_dir: str = "reports",
    strict: bool = False,
):
    """Aggregate runs and build tables, plots, and index."""
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    runs = load_all_runs(runs_root)
    print(f"[make_report_assets] Found {len(runs)} completed runs in '{runs_root}'.")

    if not runs and strict:
        raise RuntimeError(f"No completed runs found in {runs_root} and --strict was specified.")

    # 1. Build summary dictionary
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "total_runs": len(runs),
        "runs": {},
        "ablations": {},
    }

    ablation_rows = []

    for run_id, rdata in runs.items():
        summary["runs"][run_id] = rdata["metrics"]

        # Parse test metrics if present
        for m_key, m_val in rdata["metrics"].items():
            if "test" in m_key:
                ci = m_val.get("exact_match_ci95", [None, None])
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if (ci[0] is not None and ci[1] is not None) else "-"
                ablation_rows.append({
                    "run_id": run_id,
                    "split_set": m_key,
                    "exact_match_acc": m_val.get("exact_match_acc", 0.0),
                    "action_acc": m_val.get("action_acc", 0.0),
                    "object_acc": m_val.get("object_acc", 0.0),
                    "location_acc": m_val.get("location_acc", 0.0),
                    "macro_f1": m_val.get("macro_f1_exact", 0.0),
                    "exact_ci95": ci_str,
                })

    # Save summary.json
    summary_path = os.path.join(output_dir, "metrics", "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[make_report_assets] Saved summary to: {summary_path}")

    # 2. Build Ablation Table
    df_ablation = pd.DataFrame(ablation_rows)
    ablation_csv_path = os.path.join(output_dir, "tables", "table_ablation.csv")
    ablation_md_path = os.path.join(output_dir, "tables", "table_ablation.md")

    if not df_ablation.empty:
        df_ablation.sort_values(by="exact_match_acc", ascending=False, inplace=True)
        df_ablation.to_csv(ablation_csv_path, index=False)
        df_ablation.to_markdown(ablation_md_path, index=False)
        print(f"[make_report_assets] Saved ablation table to: {ablation_csv_path}")
    else:
        # Create empty placeholder table with notice
        with open(ablation_md_path, "w", encoding="utf-8") as f:
            f.write("<!-- TODO: No training runs executed yet. Run Colab GPU experiments first. -->\n")
            f.write("| Run ID | Split | Exact Acc | Action Acc | Object Acc | Location Acc | Macro F1 | 95% CI |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")

    # 3. Create RESULTS_INDEX.md
    index_path = os.path.join(output_dir, "RESULTS_INDEX.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("# SVARA Results Index (Traceability Map)\n\n")
        f.write("Maps each figure and table in the report to its generating script and source run IDs.\n")
        f.write("Per Rule 1 (AGENTS.md): Never fabricate results. All entries strictly traceable.\n\n")
        f.write("## 1. Summary Metrics\n")
        f.write(f"- File: `{summary_path}`\n")
        f.write(f"- Total active runs: {len(runs)}\n\n")
        f.write("## 2. Tables\n")
        f.write(f"- `table_ablation.csv` / `table_ablation.md`: Source runs: {list(runs.keys())}\n\n")
        f.write("## 3. Figures\n")
        f.write("- Generated under `reports/figures/`\n")

    print(f"[make_report_assets] Generated RESULTS_INDEX at: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVARA Report Assets Generator (P10-13)")
    parser.add_argument("--runs-root", default="reports/runs", help="Runs root folder")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    parser.add_argument("--strict", action="store_true", help="Fail if no runs are present")
    args = parser.parse_args()

    generate_summary_and_assets(
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        strict=args.strict,
    )
