import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import os
import json
import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 123, 456, 789, 1337]

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


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error.
    Measures how well predicted probabilities match
    actual frequencies.
    Perfect calibration = ECE of 0.0
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def get_calibration_metrics(y_true, y_prob, task_names):
    """Compute Brier + ECE for each task, return averages."""
    briers, eces = [], []

    for i, task in enumerate(task_names):
        yt   = y_true[:, i]
        yp   = y_prob[:, i]
        mask = ~np.isnan(yt)
        yt, yp = yt[mask], yp[mask]

        if len(np.unique(yt)) < 2 or len(yt) < 10:
            continue

        briers.append(brier_score_loss(yt, yp))
        eces.append(compute_ece(yt, yp))

    return {
        "brier_mean": round(float(np.mean(briers)), 4),
        "brier_std":  round(float(np.std(briers)),  4),
        "ece_mean":   round(float(np.mean(eces)),    4),
        "ece_std":    round(float(np.std(eces)),     4),
    }


def run_calibration_analysis(dataset_name, split_type):
    """
    For each seed:
    1. Train RF on training set
    2. Get raw probabilities on val + test
    3. Fit Platt scaling on val set
    4. Fit Isotonic regression on val set
    5. Apply both to test set
    6. Compare Brier + ECE before and after
    """
    print(f"\n{'='*60}")
    print(f"Calibration | {dataset_name} | {split_type} split")
    print(f"{'='*60}")

    results_by_seed = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        X_train, y_train, X_val, y_val, \
        X_test, y_test, task_cols = \
            load_data(dataset_name, split_type, seed)

        n_tasks = len(task_cols)

        # Train RF (same config as before)
        rf = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
        y_train_filled = np.nan_to_num(y_train, nan=0.0)
        rf.fit(X_train, y_train_filled)

        # Raw probabilities on val and test
        val_probs_raw  = rf.predict_proba(X_val)
        test_probs_raw = rf.predict_proba(X_test)

        if isinstance(val_probs_raw, list):
            val_probs_raw  = np.column_stack(
                [p[:, 1] for p in val_probs_raw]
            )
            test_probs_raw = np.column_stack(
                [p[:, 1] for p in test_probs_raw]
            )
        else:
            val_probs_raw  = val_probs_raw[:, 1:2]
            test_probs_raw = test_probs_raw[:, 1:2]

        # Calibrate per task using val set
        platt_probs    = np.zeros_like(test_probs_raw)
        isotonic_probs = np.zeros_like(test_probs_raw)

        for i, task in enumerate(task_cols):
            # Val labels for this task
            yv   = y_val[:, i]
            mask = ~np.isnan(yv)

            if mask.sum() < 10 or len(np.unique(yv[mask])) < 2:
                platt_probs[:, i]    = test_probs_raw[:, i]
                isotonic_probs[:, i] = test_probs_raw[:, i]
                continue

            yv_clean  = yv[mask]
            xv_scores = val_probs_raw[:, i][mask].reshape(-1, 1)

            # Platt scaling (logistic regression on scores)
            from sklearn.linear_model import LogisticRegression
            platt = LogisticRegression(C=1.0)
            platt.fit(xv_scores, yv_clean)
            platt_probs[:, i] = platt.predict_proba(
                test_probs_raw[:, i].reshape(-1, 1)
            )[:, 1]

            # Isotonic regression
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(xv_scores.ravel(), yv_clean)
            isotonic_probs[:, i] = iso.predict(
                test_probs_raw[:, i]
            )

        # Compute metrics for all three versions
        raw_metrics      = get_calibration_metrics(
            y_test, test_probs_raw, task_cols
        )
        platt_metrics    = get_calibration_metrics(
            y_test, platt_probs, task_cols
        )
        isotonic_metrics = get_calibration_metrics(
            y_test, isotonic_probs, task_cols
        )

        print(f"    {'Method':<12} {'Brier':>8} {'ECE':>8}")
        print(f"    {'Raw RF':<12} "
              f"{raw_metrics['brier_mean']:>8.4f} "
              f"{raw_metrics['ece_mean']:>8.4f}")
        print(f"    {'Platt':<12} "
              f"{platt_metrics['brier_mean']:>8.4f} "
              f"{platt_metrics['ece_mean']:>8.4f}")
        print(f"    {'Isotonic':<12} "
              f"{isotonic_metrics['brier_mean']:>8.4f} "
              f"{isotonic_metrics['ece_mean']:>8.4f}")

        # Save reliability curve data for plotting later
        reliability_data = []
        for i, task in enumerate(task_cols):
            yt   = y_test[:, i]
            mask = ~np.isnan(yt)
            if mask.sum() < 10 or len(np.unique(yt[mask])) < 2:
                continue
            yt_clean = yt[mask]

            for method, probs in [
                ('raw',      test_probs_raw),
                ('platt',    platt_probs),
                ('isotonic', isotonic_probs)
            ]:
                yp = probs[:, i][mask]
                try:
                    frac_pos, mean_pred = calibration_curve(
                        yt_clean, yp, n_bins=10,
                        strategy='uniform'
                    )
                    reliability_data.append({
                        'task':      task,
                        'method':    method,
                        'mean_pred': mean_pred.tolist(),
                        'frac_pos':  frac_pos.tolist()
                    })
                except Exception:
                    pass

        results_by_seed.append({
            "seed":             seed,
            "raw":              raw_metrics,
            "platt":            platt_metrics,
            "isotonic":         isotonic_metrics,
            "reliability_data": reliability_data
        })

    # Summary across seeds
    for method in ['raw', 'platt', 'isotonic']:
        briers = [r[method]['brier_mean'] for r in results_by_seed]
        eces   = [r[method]['ece_mean']   for r in results_by_seed]
        print(f"\n  {method.upper():<10} "
              f"Brier: {np.mean(briers):.4f}+-{np.std(briers):.4f} | "
              f"ECE: {np.mean(eces):.4f}+-{np.std(eces):.4f}")

    return results_by_seed


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_results = {}

    for dataset in ["tox21", "clintox"]:
        all_results[dataset] = {}
        for split_type in ["random", "scaffold"]:
            res = run_calibration_analysis(dataset, split_type)
            all_results[dataset][split_type] = res

    out_path = "results/calibration_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nCalibration results saved to: {out_path}")
    print("CALIBRATION ANALYSIS COMPLETE")
