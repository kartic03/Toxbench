#!/usr/bin/env python3
"""
Generate Uncertainty / Trustworthiness tables from results/uncertainty_results.json.

Outputs:
  - tables/uncertainty_table_by_coverage.csv
  - tables/uncertainty_table_summary.csv

Usage:
  cd ~/toxbench
  python src/evaluation/make_uncertainty_table.py
"""

import json
import os
import math
import csv
from statistics import mean, pstdev

UNC_JSON = "results/uncertainty_results.json"

DATASETS = ["tox21", "clintox", "sider"]
SPLITS = ["random", "scaffold"]
COVERAGES = [25, 50, 75, 100]  # change if you have others

# If you want SD as sample SD (ddof=1), set this True:
USE_SAMPLE_SD = True


def sd(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    if USE_SAMPLE_SD:
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
    return pstdev(vals)


def safe_mean(vals):
    return mean(vals) if vals else None


def fmt(m, s):
    if m is None:
        return ""
    return f"{m:.3f}±{s:.3f}"


def main():
    if not os.path.exists(UNC_JSON):
        raise FileNotFoundError(f"Missing: {UNC_JSON}")

    with open(UNC_JSON, "r") as f:
        unc = json.load(f)

    os.makedirs("tables", exist_ok=True)

    # ---------------------------------------
    # 1) Long table: one row per coverage pct
    # ---------------------------------------
    rows = []
    for ds in DATASETS:
        if ds not in unc:
            continue
        for sp in SPLITS:
            if sp not in unc[ds]:
                continue

            seed_results = unc[ds][sp]  # list over seeds
            # collect AUROC per coverage across seeds
            cov_to_vals = {c: [] for c in COVERAGES}

            for seed_res in seed_results:
                for cr in seed_res.get("coverage_results", []):
                    pct = cr.get("coverage_pct")
                    au = cr.get("auroc")
                    if pct in cov_to_vals and au is not None:
                        cov_to_vals[pct].append(float(au))

            for c in COVERAGES:
                vals = cov_to_vals[c]
                m = safe_mean(vals)
                s = sd(vals) if vals else None
                rows.append({
                    "dataset": ds,
                    "split": sp,
                    "coverage_pct": c,
                    "n_seeds_with_value": len(vals),
                    "auroc_mean": None if m is None else round(m, 6),
                    "auroc_sd": None if s is None else round(s, 6),
                    "auroc_mean_pm_sd": "" if m is None else fmt(m, s),
                })

    out1 = "tables/uncertainty_table_by_coverage.csv"
    with open(out1, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---------------------------------------
    # 2) Summary table: 25/50/75/100 + gain
    # ---------------------------------------
    summary = []
    for ds in DATASETS:
        if ds not in unc:
            continue
        for sp in SPLITS:
            if sp not in unc[ds]:
                continue

            seed_results = unc[ds][sp]
            cov_to_vals = {}

            for seed_res in seed_results:
                for cr in seed_res.get("coverage_results", []):
                    pct = cr.get("coverage_pct")
                    au = cr.get("auroc")
                    if pct is None or au is None:
                        continue
                    cov_to_vals.setdefault(int(pct), []).append(float(au))

            def cov_fmt(c):
                vals = cov_to_vals.get(c, [])
                return fmt(safe_mean(vals), sd(vals)) if vals else ""

            # gain = AUROC@25 - AUROC@100
            gain = ""
            if cov_to_vals.get(25) and cov_to_vals.get(100):
                gain_val = safe_mean(cov_to_vals[25]) - safe_mean(cov_to_vals[100])
                gain = f"{gain_val:.3f}"

            summary.append({
                "dataset": ds,
                "split": sp,
                "auroc_25": cov_fmt(25),
                "auroc_50": cov_fmt(50),
                "auroc_75": cov_fmt(75),
                "auroc_100": cov_fmt(100),
                "gain_25_minus_100": gain,
            })

    out2 = "tables/uncertainty_table_summary.csv"
    with open(out2, "w", newline="") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

    # quick preview
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

    if summary:
        print("\nUncertainty Summary (preview):")
        for r in summary:
            print(
                f"{r['dataset']:7s} {r['split']:8s} | "
                f"25={r['auroc_25'] or 'NA':>10s}  "
                f"50={r['auroc_50'] or 'NA':>10s}  "
                f"75={r['auroc_75'] or 'NA':>10s}  "
                f"100={r['auroc_100'] or 'NA':>10s}  "
                f"gain={r['gain_25_minus_100'] or 'NA'}"
            )


if __name__ == "__main__":
    main()
