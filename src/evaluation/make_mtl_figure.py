"""Two-architecture MTL pilot figure + supplementary table.
Combines the MLP and GIN cross-dataset MTL pilots into one figure (CT_TOX
baseline vs MTL, mean +/- SD over 5 seeds) and one tidy CSV."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "supplementary_files")

mlp = json.load(open(os.path.join(OUT, "mtl_pilot_summary.json")))
gin = json.load(open(os.path.join(OUT, "mtl_pilot_gnn_summary.json")))

# ---- tidy supplementary table (CT_TOX, macro, FDA for both architectures) ----
def grab(d, arch, base_key, mtl_key):
    rows = []
    fields = [("CT_TOX", "CT_TOX_scaffold_AUROC"),
              ("ClinTox macro (2 tasks)", "ClinTox_scaffold_macro_AUROC"),
              ("FDA_APPROVED", "FDA_APPROVED_scaffold_AUROC")]
    for label, key in fields:
        b = d[key][base_key]; m = d[key][mtl_key]
        rows.append(dict(architecture=arch, endpoint=label,
                         baseline_mean=b[0], baseline_sd=b[1],
                         MTL_mean=m[0], MTL_sd=m[1],
                         delta=round(m[0] - b[0], 4)))
    return rows

tbl = grab(mlp, "MLP", "baseline_ClinTox_only", "MTL_3dataset") + \
      grab(gin, "GIN (GNN)", "baseline", "MTL")
df = pd.DataFrame(tbl)
df.to_csv(os.path.join(OUT, "S_MTL_two_architectures.csv"), index=False)
print(df.to_string(index=False))

# ---- figure: CT_TOX baseline -> MTL for both architectures ----
archs = ["MLP", "GIN (GNN)"]
base = [mlp["CT_TOX_scaffold_AUROC"]["baseline_ClinTox_only"],
        gin["CT_TOX_scaffold_AUROC"]["baseline"]]
mtlv = [mlp["CT_TOX_scaffold_AUROC"]["MTL_3dataset"],
        gin["CT_TOX_scaffold_AUROC"]["MTL"]]

x = np.arange(len(archs)); w = 0.34
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.bar(x - w/2, [b[0] for b in base], w, yerr=[b[1] for b in base],
       capsize=5, color="#9ecae1", edgecolor="#3182bd",
       label="Single-dataset baseline (ClinTox only)")
ax.bar(x + w/2, [m[0] for m in mtlv], w, yerr=[m[1] for m in mtlv],
       capsize=5, color="#fc9272", edgecolor="#de2d26",
       label="Cross-dataset MTL (Tox21+SIDER+ClinTox)")
ax.axhline(0.5, color="0.6", ls="--", lw=1, label="Random (0.5)")
# value labels placed above each error-bar cap (mean + sd), so whiskers never cross text
for xi, (mean, sd) in zip(x - w/2, base):
    ax.text(xi, mean + sd + 0.015, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
for xi, (mean, sd) in zip(x + w/2, mtlv):
    ax.text(xi, mean + sd + 0.015, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(archs)
ax.set_ylabel("CT_TOX AUROC (ClinTox scaffold test)")
ax.set_ylim(0.4, 1.15)
ax.set_title("Cross-dataset MTL improves ClinTox scaffold generalization\n"
             "on both architectures (mean ± SD, 5 seeds)", fontsize=11)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"FigS_MTL_two_architectures.{ext}"),
                dpi=300, bbox_inches="tight")
print("\nSaved FigS_MTL_two_architectures.png/pdf and S_MTL_two_architectures.csv")
