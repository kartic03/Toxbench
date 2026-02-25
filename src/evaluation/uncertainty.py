import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
import json
import warnings
warnings.filterwarnings('ignore')

SEEDS        = [42, 123, 456, 789, 1337]
ENSEMBLE_SIZE = 5


def load_data(dataset_name, split_type, seed):
    fps      = np.load(f"data/processed/{dataset_name}_ecfp4.npy")
    df       = pd.read_csv(f"data/processed/{dataset_name}_clean.csv")
    split_df = pd.read_csv(
        f"splits/{dataset_name}_{split_type}_seed{seed}.csv"
    )
    train_idx = split_df[split_df['split']=='train']['index'].values
    val_idx   = split_df[split_df['split']=='val']['index'].values
    test_idx  = split_df[split_df['split']=='test']['index'].values

    X_train = fps[train_idx]
    X_val   = fps[val_idx]
    X_test  = fps[test_idx]

    task_cols = [c for c in df.columns if c != 'smiles']
    y_train = df[task_cols].values[train_idx]
    y_val   = df[task_cols].values[val_idx]
    y_test  = df[task_cols].values[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test, task_cols


def run_uncertainty_analysis(dataset_name, split_type):
    """
    Ensemble uncertainty analysis:
    1. For each split seed, train 5 RF models
       (each with a different internal random seed)
    2. Get predictions from all 5 models
    3. Mean prediction = ensemble prediction
    4. Std of predictions = uncertainty estimate
    5. Bin compounds by uncertainty level
    6. Show: high uncertainty = lower AUROC
       (trustworthiness curve)
    """
    print(f"\n{'='*60}")
    print(f"Uncertainty | {dataset_name} | {split_type} split")
    print(f"{'='*60}")

    all_results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        X_train, y_train, X_val, y_val, \
        X_test, y_test, task_cols = \
            load_data(dataset_name, split_type, seed)

        n_tasks = len(task_cols)

        # Train ensemble of 5 RF models
        ensemble_probs = []
        print(f"    Training ensemble of {ENSEMBLE_SIZE} models...")

        for e_seed in range(ENSEMBLE_SIZE):
            rf = RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=seed * 100 + e_seed,
                n_jobs=-1
            )
            rf.fit(X_train,
                   np.nan_to_num(y_train, nan=0.0))

            probs = rf.predict_proba(X_test)
            if isinstance(probs, list):
                probs = np.column_stack(
                    [p[:, 1] for p in probs]
                )
            else:
                probs = probs[:, 1:2]

            ensemble_probs.append(probs)

        # Stack: shape = (ENSEMBLE_SIZE, n_test, n_tasks)
        ensemble_stack = np.stack(ensemble_probs, axis=0)

        # Mean and std across ensemble members
        mean_probs = ensemble_stack.mean(axis=0)
        std_probs  = ensemble_stack.std(axis=0)

        # Average uncertainty per compound
        # (mean std across tasks)
        compound_uncertainty = std_probs.mean(axis=1)

        print(f"    Uncertainty stats: "
              f"min={compound_uncertainty.min():.4f} "
              f"mean={compound_uncertainty.mean():.4f} "
              f"max={compound_uncertainty.max():.4f}")

        # Bin compounds by uncertainty level
        # Low uncertainty = model is confident
        # High uncertainty = model is unsure
        percentiles = [0, 25, 50, 75, 100]
        thresholds  = np.percentile(
            compound_uncertainty, percentiles
        )

        bin_labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
        print(f"\n    {'Uncertainty Bin':<18} "
              f"{'N':>5} {'AUROC':>8} {'Meaning'}")
        print(f"    {'-'*55}")

        bin_results = []

        for i in range(4):
            lo   = thresholds[i]
            hi   = thresholds[i+1]
            mask = ((compound_uncertainty >= lo) &
                    (compound_uncertainty <= hi))
            n_bin = mask.sum()

            bin_aurocs = []
            for j, task in enumerate(task_cols):
                yt   = y_test[mask, j]
                yp   = mean_probs[mask, j]
                vmask = ~np.isnan(yt)
                yt, yp = yt[vmask], yp[vmask]
                if len(np.unique(yt)) < 2:
                    continue
                try:
                    bin_aurocs.append(roc_auc_score(yt, yp))
                except:
                    pass

            if bin_aurocs:
                auroc   = np.mean(bin_aurocs)
                meaning = (
                    "Trust predictions" if i == 0
                    else "Use with caution" if i == 1
                    else "Be skeptical" if i == 2
                    else "Do not trust"
                )
                print(f"    {bin_labels[i]:<18} "
                      f"{n_bin:>5} {auroc:>8.4f}  {meaning}")
                bin_results.append({
                    "bin":   bin_labels[i],
                    "n":     int(n_bin),
                    "auroc": round(float(auroc), 4),
                    "unc_lo": round(float(lo), 4),
                    "unc_hi": round(float(hi), 4)
                })
            else:
                bin_results.append({
                    "bin":   bin_labels[i],
                    "n":     int(n_bin),
                    "auroc": None
                })

        # Also compute: what happens if we REJECT
        # high-uncertainty predictions?
        # (the "trustworthiness curve")
        coverage_results = []
        thresholds_pct = [100, 75, 50, 25]

        print(f"\n    Trustworthiness curve "
              f"(keeping only low-uncertainty predictions):")
        print(f"    {'Coverage':>10} {'N kept':>8} {'AUROC':>8}")
        print(f"    {'-'*30}")

        for pct in thresholds_pct:
            threshold = np.percentile(
                compound_uncertainty, pct
            )
            mask = compound_uncertainty <= threshold
            n_kept = mask.sum()

            kept_aurocs = []
            for j, task in enumerate(task_cols):
                yt    = y_test[mask, j]
                yp    = mean_probs[mask, j]
                vmask = ~np.isnan(yt)
                yt, yp = yt[vmask], yp[vmask]
                if len(np.unique(yt)) < 2:
                    continue
                try:
                    kept_aurocs.append(roc_auc_score(yt, yp))
                except:
                    pass

            if kept_aurocs:
                auroc = np.mean(kept_aurocs)
                print(f"    {pct:>9}%  {n_kept:>8}  {auroc:>8.4f}")
                coverage_results.append({
                    "coverage_pct": pct,
                    "n_kept":       int(n_kept),
                    "auroc":        round(float(auroc), 4)
                })

        all_results.append({
            "seed":             seed,
            "bin_results":      bin_results,
            "coverage_results": coverage_results,
            "mean_uncertainty": float(
                compound_uncertainty.mean()
            )
        })

    return all_results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_unc_results = {}

    for dataset in ["tox21", "clintox"]:
        all_unc_results[dataset] = {}
        for split_type in ["random", "scaffold"]:
            res = run_uncertainty_analysis(
                dataset, split_type
            )
            all_unc_results[dataset][split_type] = res

    out_path = "results/uncertainty_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_unc_results, f, indent=2)

    print(f"\nUncertainty results saved to: {out_path}")
    print("UNCERTAINTY ANALYSIS COMPLETE")
