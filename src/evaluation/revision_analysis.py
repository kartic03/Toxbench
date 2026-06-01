"""
Revision re-analysis for ToxBench (BMC Bioinformatics major revision).
Produces, with NO model retraining, every reviewer-requested artifact that is
computable from data already in the repo:

  S5  -> per-task dataset statistics (pos/neg/missing/%pos)            [R1.3, R1.7]
  S6  -> per-task AUROC random vs scaffold (mean+/-SD, drop)           [R1.4, R2.4]
  S7  -> per-task AUPRC random vs scaffold (mean+/-SD, drop)           [R1.4]
  endpoint_difficulty -> ranking of endpoints by scaffold AUROC        [R2.4]
  leakage_audit -> train->test NN Tanimoto dist, residual >0.4/0.6/0.8 [E2, R1.1]
  fig_dumbbell  -> consolidated random->scaffold macro AUROC figure    [R2.3]

Outputs land in  supplementary_files/.
"""
import os, json, glob
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")
SPLITS = os.path.join(ROOT, "splits")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "supplementary_files")
os.makedirs(OUT, exist_ok=True)

DATASETS = ["tox21", "clintox", "sider"]
SEEDS = [42, 123, 456, 789, 1337]
MODELS = {
    "RandomForest": "random_forest_results.json",
    "XGBoost": "xgboost_results.json",
    "MLP": "mlp_results.json",
    "GNN": "gnn_results.json",
}

def label_cols(df):
    return [c for c in df.columns if c.lower() != "smiles"]

# ---------------------------------------------------------------- S5: task stats
def s5_task_stats():
    rows = []
    for ds in DATASETS:
        df = pd.read_csv(os.path.join(PROC, f"{ds}_clean.csv"))
        for task in label_cols(df):
            col = pd.to_numeric(df[task], errors="coerce")
            n_pos = int((col == 1).sum())
            n_neg = int((col == 0).sum())
            n_miss = int(col.isna().sum())
            n_lab = n_pos + n_neg
            pct = 100.0 * n_pos / n_lab if n_lab else np.nan
            rows.append(dict(dataset=ds, task=task, n_positive=n_pos,
                             n_negative=n_neg, n_missing=n_miss,
                             n_labeled=n_lab, pct_positive=round(pct, 2)))
    s5 = pd.DataFrame(rows)
    s5.to_csv(os.path.join(OUT, "S5_per_task_statistics.csv"), index=False)
    # imbalance summary per dataset
    summ = (s5.groupby("dataset")["pct_positive"]
              .agg(["min", "median", "max", "mean"]).round(2))
    summ.to_csv(os.path.join(OUT, "S5_imbalance_summary.csv"))
    print("[S5] per-task stats:", s5.shape, "-> S5_per_task_statistics.csv")
    print(summ.to_string())
    return s5

# ------------------------------------------------- S6/S7: per-task performance
def load_all_results():
    data = {}
    for model, fname in MODELS.items():
        data[model] = json.load(open(os.path.join(RESULTS, fname)))
    return data

def s6_s7_per_task(metric):
    """metric in {'auroc','auprc'}. Aggregate per task across seeds & models."""
    allres = load_all_results()
    rows = []
    for ds in DATASETS:
        # discover tasks from one model's first seed
        sample = allres["RandomForest"][ds]["random"][0]["test_results"]
        tasks = [t for t in sample.keys() if t != "MACRO_AVG"]
        for task in tasks:
            rec = dict(dataset=ds, task=task)
            # per-model values + a cross-model pool
            rnd_pool, scf_pool = [], []
            for model in MODELS:
                for split, pool in (("random", rnd_pool), ("scaffold", scf_pool)):
                    vals = []
                    for seed_entry in allres[model][ds][split]:
                        tr = seed_entry["test_results"].get(task)
                        if tr is not None and tr.get(metric) is not None:
                            vals.append(tr[metric])
                    if vals:
                        rec[f"{model}_{split}_mean"] = round(np.mean(vals), 4)
                        rec[f"{model}_{split}_sd"] = round(np.std(vals), 4)
                        pool.extend(vals)
            if rnd_pool and scf_pool:
                rec["all_random_mean"] = round(np.mean(rnd_pool), 4)
                rec["all_scaffold_mean"] = round(np.mean(scf_pool), 4)
                rec["drop"] = round(np.mean(scf_pool) - np.mean(rnd_pool), 4)
            rows.append(rec)
    out = pd.DataFrame(rows)
    name = "S6_per_task_AUROC" if metric == "auroc" else "S7_per_task_AUPRC"
    out.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"[{name}] {out.shape} -> {name}.csv")
    return out

def endpoint_difficulty(s6):
    """Rank endpoints by cross-model scaffold AUROC; flag easy/hard."""
    df = s6.dropna(subset=["all_scaffold_mean"]).copy()
    df = df.sort_values(["dataset", "all_scaffold_mean"])
    df_rank = df[["dataset", "task", "all_random_mean",
                  "all_scaffold_mean", "drop"]].copy()
    df_rank.to_csv(os.path.join(OUT, "endpoint_difficulty_ranking.csv"), index=False)
    print("[endpoint_difficulty] -> endpoint_difficulty_ranking.csv")
    for ds in DATASETS:
        sub = df_rank[df_rank.dataset == ds].sort_values("all_scaffold_mean")
        if sub.empty:
            continue
        easiest = sub.iloc[-1]
        hardest = sub.iloc[0]
        print(f"  {ds:8s}: hardest={hardest.task} ({hardest.all_scaffold_mean:.3f})"
              f"  easiest={easiest.task} ({easiest.all_scaffold_mean:.3f})")
    return df_rank

# ------------------------------------------------------- leakage audit
def tanimoto_nn_max(test_fp, train_fp):
    """Max Tanimoto of each test row to any train row. Bit fingerprints."""
    test_fp = test_fp.astype(np.float32)
    train_fp = train_fp.astype(np.float32)
    a = test_fp.sum(1)             # popcount test
    b = train_fp.sum(1)            # popcount train
    inter = test_fp @ train_fp.T   # (Nte, Ntr)
    union = a[:, None] + b[None, :] - inter
    union[union == 0] = 1.0
    tani = inter / union
    return tani.max(1)

def leakage_audit():
    rows = []
    dist_rows = []
    for ds in DATASETS:
        fp = np.load(os.path.join(PROC, f"{ds}_ecfp4.npy"))
        for split in ["random", "scaffold"]:
            nn_all = []
            for seed in SEEDS:
                sp = pd.read_csv(os.path.join(SPLITS, f"{ds}_{split}_seed{seed}.csv"))
                tr_idx = sp.loc[sp.split == "train", "index"].values
                te_idx = sp.loc[sp.split == "test", "index"].values
                nn = tanimoto_nn_max(fp[te_idx], fp[tr_idx])
                nn_all.append(nn)
                rows.append(dict(
                    dataset=ds, split=split, seed=seed,
                    n_test=len(nn),
                    nn_mean=round(float(nn.mean()), 4),
                    nn_median=round(float(np.median(nn)), 4),
                    pct_gt_0p4=round(100*float((nn > 0.4).mean()), 1),
                    pct_gt_0p6=round(100*float((nn > 0.6).mean()), 1),
                    pct_gt_0p8=round(100*float((nn > 0.8).mean()), 1),
                ))
            nn_cat = np.concatenate(nn_all)
            dist_rows.append(dict(
                dataset=ds, split=split,
                nn_mean=round(float(nn_cat.mean()), 4),
                nn_median=round(float(np.median(nn_cat)), 4),
                pct_gt_0p4=round(100*float((nn_cat > 0.4).mean()), 1),
                pct_gt_0p6=round(100*float((nn_cat > 0.6).mean()), 1),
                pct_gt_0p8=round(100*float((nn_cat > 0.8).mean()), 1),
            ))
    per_seed = pd.DataFrame(rows)
    summary = pd.DataFrame(dist_rows)
    per_seed.to_csv(os.path.join(OUT, "leakage_audit_per_seed.csv"), index=False)
    summary.to_csv(os.path.join(OUT, "leakage_audit_summary.csv"), index=False)
    print("[leakage_audit] -> leakage_audit_summary.csv")
    print(summary.to_string(index=False))
    return summary

# ------------------------------------------------------- consolidated figure
def macro_table():
    allres = load_all_results()
    rows = []
    for model in MODELS:
        for ds in DATASETS:
            for split in ["random", "scaffold"]:
                vals = [e["test_results"]["MACRO_AVG"]["auroc"]
                        for e in allres[model][ds][split]]
                rows.append(dict(model=model, dataset=ds, split=split,
                                 mean=np.mean(vals), sd=np.std(vals)))
    return pd.DataFrame(rows)

def fig_dumbbell(macro):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds_titles = {"tox21": "Tox21 (12 tasks)",
                 "clintox": "ClinTox (2 tasks)",
                 "sider": "SIDER (27 tasks)"}
    models = list(MODELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=False)
    for ax, ds in zip(axes, DATASETS):
        sub = macro[macro.dataset == ds]
        ypos = np.arange(len(models))
        for i, m in enumerate(models):
            r = sub[(sub.model == m) & (sub.split == "random")].iloc[0]
            s = sub[(sub.model == m) & (sub.split == "scaffold")].iloc[0]
            ax.plot([r["mean"], s["mean"]], [i, i], color="0.6", zorder=1, lw=2)
            ax.errorbar(r["mean"], i, xerr=r["sd"], fmt="o", color="#1f77b4",
                        capsize=3, ms=7, zorder=3,
                        label="Random" if i == 0 else None)
            ax.errorbar(s["mean"], i, xerr=s["sd"], fmt="s", color="#d62728",
                        capsize=3, ms=7, zorder=3,
                        label="Scaffold" if i == 0 else None)
        ax.axvline(0.5, color="0.8", ls="--", lw=1, zorder=0)
        ax.set_yticks(ypos); ax.set_yticklabels(models)
        ax.set_title(ds_titles[ds]); ax.set_xlabel("Macro AUROC")
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=9)
    fig.suptitle("Random vs. scaffold split: macro-AUROC (mean ± SD over 5 seeds)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    p_png = os.path.join(OUT, "Fig1_random_vs_scaffold_dumbbell.png")
    p_pdf = os.path.join(OUT, "Fig1_random_vs_scaffold_dumbbell.pdf")
    fig.savefig(p_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    print("[fig_dumbbell] ->", os.path.basename(p_png))

if __name__ == "__main__":
    print("=== ToxBench revision re-analysis ===")
    s5 = s5_task_stats()
    s6 = s6_s7_per_task("auroc")
    s7 = s6_s7_per_task("auprc")
    endpoint_difficulty(s6)
    leakage_audit()
    fig_dumbbell(macro_table())
    print("\nAll outputs in:", OUT)
