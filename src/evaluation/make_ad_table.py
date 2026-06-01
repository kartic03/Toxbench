#!/usr/bin/env python3
"""
Generate Applicability Domain (AD) analysis tables from results/ad_results.json.

Outputs:
  - tables/ad_table_by_bin.csv
  - tables/ad_table_summary.csv

Usage:
  cd ~/toxbench
  python src/evaluation/make_ad_table.py
"""

import json
import os
import math
import csv
from statistics import mean, pstdev

AD_JSON = "results/ad_results.json"

DATASETS = ["tox21", "clintox", "sider"]
SPLITS = ["random", "scaffold"]
BIN_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

# If you want SD as sample SD (ddof=1), set this True:
USE_SAMPLE_SD = True


def sd(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    if USE_SAMPLE_SD:
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
    # population SD
    return pstdev(vals)


def safe_mean(vals):
    return mean(vals) if vals else None


def fmt(m, s):
    if m is None:
        return ""
    return f"{m:.3f}±{s:.3f}"


def main():
    if not os.path.exists(AD_JSON):
        raise FileNotFoundError(f"Missing: {AD_JSON}")

    with open(AD_JSON, "r") as f:
        ad = json.load(f)

    os.makedirs("tables", exist_ok=True)

    # -----------------------------
    # 1) Bin-level table (long form)
    # -----------------------------
    rows = []
    for ds in DATASETS:
        if ds not in ad:
            continue
        for sp in SPLITS:
            if sp not in ad[ds]:
                continue

            # ad[ds][sp]["bin_results"] = list over seeds;
            # each seed is a list of dicts: {"bin": "...", "auroc": float or None, ...}
            seed_bins = ad[ds][sp].get("bin_results", [])

            # collect values per bin across seeds
            by_bin = {b: [] for b in BIN_LABELS}
            for seed in seed_bins:
                for rec in seed:
                    b = rec.get("bin")
                    v = rec.get("auroc")
                    if b in by_bin and v is not None:
                        by_bin[b].append(float(v))

            for b in BIN_LABELS:
                vals = by_bin[b]
                m = safe_mean(vals)
                s = sd(vals) if vals else None
                rows.append({
                    "dataset": ds,
                    "split": sp,
                    "bin": b,
                    "n_seeds_with_value": len(vals),
                    "auroc_mean": None if m is None else round(m, 6),
                    "auroc_sd": None if s is None else round(s, 6),
                    "auroc_mean_pm_sd": "" if m is None else fmt(m, s),
                })

    out1 = "tables/ad_table_by_bin.csv"
    with open(out1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    # -----------------------------
    # 2) Summary table (wide-ish)
    #    Includes "AD slope" = mean(high sim bin) - mean(low sim bin)
    # -----------------------------
    summary = []
    for ds in DATASETS:
        if ds not in ad:
            continue
        for sp in SPLITS:
            if sp not in ad[ds]:
                continue

            seed_bins = ad[ds][sp].get("bin_results", [])
            by_bin = {b: [] for b in BIN_LABELS}
            for seed in seed_bins:
                for rec in seed:
                    b = rec.get("bin")
                    v = rec.get("auroc")
                    if b in by_bin and v is not None:
                        by_bin[b].append(float(v))

            # low and high similarity bins
            low_vals = by_bin["0.0-0.2"]
            high_vals = by_bin["0.8-1.0"]
            low_m = safe_mean(low_vals)
            high_m = safe_mean(high_vals)

            slope = None
            if low_m is not None and high_m is not None:
                slope = high_m - low_m

            row = {
                "dataset": ds,
                "split": sp,
                "low_bin_0.0-0.2": fmt(safe_mean(low_vals), sd(low_vals)) if low_vals else "",
                "mid_bin_0.4-0.6": fmt(safe_mean(by_bin["0.4-0.6"]), sd(by_bin["0.4-0.6"])) if by_bin["0.4-0.6"] else "",
                "high_bin_0.8-1.0": fmt(safe_mean(high_vals), sd(high_vals)) if high_vals else "",
                "ad_slope_high_minus_low": "" if slope is None else f"{slope:.3f}",
            }
            summary.append(row)

    out2 = "tables/ad_table_summary.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            w.writeheader()
            w.writerows(summary)

    # Print a quick preview
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

    if summary:
        print("\nAD Summary (preview):")
        for r in summary:
            print(
                f"{r['dataset']:7s} {r['split']:8s} | "
                f"low={r['low_bin_0.0-0.2'] or 'NA':>10s}  "
                f"mid={r['mid_bin_0.4-0.6'] or 'NA':>10s}  "
                f"high={r['high_bin_0.8-1.0'] or 'NA':>10s}  "
                f"slope={r['ad_slope_high_minus_low'] or 'NA'}"
            )


if __name__ == "__main__":
    main()
