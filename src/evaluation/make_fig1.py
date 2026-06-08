"""Fig. 1 (revised, R2.3): a single two-panel figure that replaces the eight
random-vs-scaffold Results paragraphs and Table 5.

  Panel A — dumbbell: random vs scaffold macro-AUROC (mean, 0-1 axis), 4 models x 3 datasets.
  Panel B — bars of the AUROC drop (scaffold - random), paired per seed, with SD error bars,
            so the headline overestimation is read directly.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "supplementary_files")
MODELS = {"RandomForest": "random_forest_results.json", "XGBoost": "xgboost_results.json",
          "MLP": "mlp_results.json", "GNN": "gnn_results.json"}
DATASETS = ["tox21", "clintox", "sider"]
DLAB = {"tox21": "Tox21", "clintox": "ClinTox", "sider": "SIDER"}

res = {m: json.load(open(os.path.join(RESULTS, f))) for m, f in MODELS.items()}

def macro_seeds(model, ds, split):
    return np.array([e["test_results"]["MACRO_AVG"]["auroc"] for e in res[model][ds][split]])

# build rows grouped by dataset (top->bottom: Tox21, ClinTox, SIDER), models RF,XGB,MLP,GNN
rows = []  # each: dict
for ds in DATASETS:
    for m in MODELS:
        r = macro_seeds(m, ds, "random"); s = macro_seeds(m, ds, "scaffold")
        drop = s - r  # paired by seed
        rows.append(dict(ds=ds, model=m, r_mean=r.mean(), r_sd=r.std(),
                         s_mean=s.mean(), s_sd=s.std(),
                         d_mean=drop.mean(), d_sd=drop.std()))

# y positions with a gap between datasets
ypos, ylabels, yticks_major = [], [], []
y = 0.0; sep = 0.8
row_y = []
prev_ds = None
for i, rw in enumerate(rows):
    if prev_ds is not None and rw["ds"] != prev_ds:
        y -= sep
    row_y.append(y); ylabels.append(rw["model"]); y -= 1.0; prev_ds = rw["ds"]
row_y = np.array(row_y)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.6),
                               gridspec_kw={"width_ratios": [1.45, 1.0]}, sharey=True)

# ---- Panel A: dumbbell ----
for yy, rw in zip(row_y, rows):
    axA.plot([rw["r_mean"], rw["s_mean"]], [yy, yy], color="0.6", lw=2, zorder=1)
    axA.errorbar(rw["r_mean"], yy, xerr=rw["r_sd"], fmt="o", color="#1f77b4",
                 ms=7, capsize=3, zorder=3)
    axA.errorbar(rw["s_mean"], yy, xerr=rw["s_sd"], fmt="s", color="#d62728",
                 ms=7, capsize=3, zorder=3)
axA.axvline(0.5, color="0.8", ls="--", lw=1, zorder=0)
axA.set_yticks(row_y); axA.set_yticklabels(ylabels, fontsize=9)
axA.set_xlim(0.0, 1.0); axA.set_xlabel("Macro AUROC")
axA.set_title("A  Random vs. scaffold split", loc="left", fontsize=11, fontweight="bold")
axA.grid(axis="x", alpha=0.25)
# legend proxies
from matplotlib.lines import Line2D
axA.legend([Line2D([0],[0],marker="o",color="w",markerfacecolor="#1f77b4",ms=8),
            Line2D([0],[0],marker="s",color="w",markerfacecolor="#d62728",ms=8)],
           ["Random", "Scaffold"], loc="lower left", fontsize=9, frameon=True)

# ---- Panel B: drop bars ----
colors = ["#d62728" if rw["d_mean"] < 0 else "#2ca02c" for rw in rows]
axB.barh(row_y, [rw["d_mean"] for rw in rows], height=0.62,
         xerr=[rw["d_sd"] for rw in rows], color=colors, edgecolor="0.3",
         error_kw=dict(ecolor="0.4", capsize=3), zorder=2)
axB.axvline(0, color="0.4", lw=1, zorder=1)
# x-limits fit the widest whisker plus room for the value labels (nothing clipped)
left_tip = min(rw["d_mean"] - rw["d_sd"] for rw in rows)
right_tip = max(rw["d_mean"] + rw["d_sd"] for rw in rows)
LBL = 0.075          # data-unit room reserved for a label beyond a whisker tip
axB.set_xlim(left_tip - LBL - 0.04, right_tip + LBL + 0.04)
axB.set_xlabel("Δ AUROC  (scaffold − random)")
axB.set_title("B  Performance change", loc="left", fontsize=11, fontweight="bold")
axB.grid(axis="x", alpha=0.25)
# labels sit just beyond each whisker tip, aligned away from the bar -> no overlap
for yy, rw in zip(row_y, rows):
    if rw["d_mean"] >= 0:
        axB.text(rw["d_mean"] + rw["d_sd"] + 0.012, yy, f"{rw['d_mean']:+.3f}",
                 va="center", ha="left", fontsize=8)
    else:
        axB.text(rw["d_mean"] - rw["d_sd"] - 0.012, yy, f"{rw['d_mean']:+.3f}",
                 va="center", ha="right", fontsize=8)

# dataset labels on far left
for ds in DATASETS:
    ys = [yy for yy, rw in zip(row_y, rows) if rw["ds"] == ds]
    ymid = np.mean(ys)
    axA.text(-0.16, ymid, DLAB[ds], transform=axA.get_yaxis_transform(),
             rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")

fig.suptitle("Random vs. scaffold split: macro-AUROC across 4 models × 3 datasets "
             "(mean ± SD, 5 seeds)", y=0.99, fontsize=12)
fig.tight_layout(rect=[0.04, 0, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig1_random_vs_scaffold.{ext}"), dpi=300, bbox_inches="tight")
print("saved Fig1_random_vs_scaffold.png/pdf")
print("rows:", [(r['ds'], r['model'], round(r['d_mean'],3)) for r in rows])
