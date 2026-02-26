#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150
})

COLORS = {"random": "#2196F3", "scaffold": "#F44336"}
DATASETS = ["tox21", "clintox", "sider"]
DATASET_TITLES = {"tox21": "Tox21", "clintox": "ClinTox", "sider": "SIDER"}
BIN_LABELS  = ["0.0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1.0"]
BIN_CENTERS = [0.1, 0.3, 0.5, 0.7, 0.9]

def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r") as f:
        return json.load(f)

def plot_fig4_ad_all3(ad_res: dict, outpath: str):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle(
        "Figure 4: Applicability Domain — Performance vs Similarity to Training Set",
        fontsize=13, fontweight="bold"
    )

    for ax_idx, dataset in enumerate(DATASETS):
        ax = axes[ax_idx]
        ds_title = DATASET_TITLES.get(dataset, dataset)

        ds_block = ad_res.get(dataset)
        if not isinstance(ds_block, dict):
            ax.text(0.5, 0.5, "No AD data", ha="center", va="center", fontsize=12)
            ax.set_title(ds_title)
            ax.set_xlim(0, 1)
            ax.set_ylim(0.2, 1.0)
            ax.grid(alpha=0.3)
            continue

        for split_type in ["random", "scaffold"]:
            split_block = ds_block.get(split_type)
            if not isinstance(split_block, dict):
                continue

            color = COLORS[split_type]
            bin_results = split_block.get("bin_results", [])

            bin_aurocs = {b: [] for b in BIN_LABELS}
            for seed_bins in bin_results:
                if not isinstance(seed_bins, list):
                    continue
                for b in seed_bins:
                    if not isinstance(b, dict):
                        continue
                    auroc = b.get("auroc")
                    bname = b.get("bin")
                    if bname in bin_aurocs and auroc is not None:
                        bin_aurocs[bname].append(auroc)

            xs, ys, yerrs = [], [], []
            for bl, bc in zip(BIN_LABELS, BIN_CENTERS):
                vals = bin_aurocs.get(bl, [])
                if len(vals) >= 2:
                    xs.append(bc); ys.append(float(np.mean(vals))); yerrs.append(float(np.std(vals)))
                elif len(vals) == 1:
                    xs.append(bc); ys.append(float(vals[0])); yerrs.append(0.0)

            if xs:
                ax.errorbar(xs, ys, yerr=yerrs,
                            color=color, marker="o",
                            linewidth=2, markersize=7,
                            capsize=4,
                            label=f"{split_type.capitalize()} split")

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random chance")
        ax.set_xlabel("Nearest-Neighbor Tanimoto Similarity\nto Training Set")
        ax.set_ylabel("AUROC (macro-average ± SD)")
        ax.set_title(ds_title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.2, 1.0)
        ax.grid(alpha=0.3)
        ax.set_xticks(BIN_CENTERS)
        ax.set_xticklabels(BIN_LABELS, rotation=30, ha="right")

        if ax_idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {outpath}")

def plot_fig5_unc_all3(unc_res: dict, outpath: str):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle(
        "Figure 5: Uncertainty — Trustworthiness Curves\n"
        "(Rejecting high-uncertainty predictions improves AUROC)",
        fontsize=13, fontweight="bold"
    )

    for ax_idx, dataset in enumerate(DATASETS):
        ax = axes[ax_idx]
        ds_title = DATASET_TITLES.get(dataset, dataset)

        ds_block = unc_res.get(dataset)
        if not isinstance(ds_block, dict):
            ax.text(0.5, 0.5, "No uncertainty data", ha="center", va="center", fontsize=12)
            ax.set_title(ds_title)
            ax.set_xlim(15, 110)
            ax.grid(alpha=0.3)
            continue

        for split_type in ["random", "scaffold"]:
            split_list = ds_block.get(split_type)
            if not isinstance(split_list, list):
                continue

            color = COLORS[split_type]
            coverage_aurocs = {}

            for seed_res in split_list:
                if not isinstance(seed_res, dict):
                    continue
                for cr in seed_res.get("coverage_results", []):
                    if not isinstance(cr, dict):
                        continue
                    pct = cr.get("coverage_pct")
                    auroc = cr.get("auroc")
                    if pct is None or auroc is None:
                        continue
                    coverage_aurocs.setdefault(pct, []).append(auroc)

            if not coverage_aurocs:
                continue

            pcts = sorted(coverage_aurocs.keys(), reverse=True)
            means = [float(np.mean(coverage_aurocs[p])) for p in pcts]
            stds  = [float(np.std(coverage_aurocs[p]))  for p in pcts]

            ax.errorbar(pcts, means, yerr=stds,
                        color=color, marker="o",
                        linewidth=2, markersize=7,
                        capsize=4,
                        label=f"{split_type.capitalize()} split")

        ax.set_xlabel("Coverage (% of test compounds kept)")
        ax.set_ylabel("AUROC (macro-average ± SD)")
        ax.set_title(ds_title)
        ax.set_xlim(15, 110)
        ax.set_xticks([25, 50, 75, 100])
        ax.set_xticklabels(["25%\n(most confident)", "50%", "75%", "100%\n(all)"])
        ax.grid(alpha=0.3)

        ax.text(0.98, 0.05,
                "Lower coverage = higher confidence\n= better AUROC",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9, alpha=0.6, style="italic")

        if ax_idx == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {outpath}")

def main():
    ad_res = load_json("results/ad_results.json")
    unc_res = load_json("results/uncertainty_results.json")

    plot_fig4_ad_all3(ad_res, "figures/fig4_applicability_domain_all3.png")
    plot_fig5_unc_all3(unc_res, "figures/fig5_uncertainty_all3.png")
    print("\nDONE: Generated Figure 4 and 5 with Tox21, ClinTox, and SIDER.")

if __name__ == "__main__":
    main()
