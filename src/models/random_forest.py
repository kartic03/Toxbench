import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import brier_score_loss
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ── Hyperparameters to try (small grid) ───────────────────
RF_CONFIGS = [
    {"n_estimators": 100, "max_depth": None, "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": None, "class_weight": "balanced"},
    {"n_estimators": 100, "max_depth": 20,   "class_weight": "balanced"},
    {"n_estimators": 500, "max_depth": 20,   "class_weight": "balanced"},
]

SEEDS = [42, 123, 456, 789, 1337]


def load_data(dataset_name, split_type, seed):
    """
    Load fingerprints and labels for a specific
    dataset, split type, and seed.
    Returns X_train, y_train, X_val, y_val, X_test, y_test
    for each task.
    """
    # Load fingerprints (the 2048-number vectors)
    fps = np.load(f"data/processed/{dataset_name}_ecfp4.npy")

    # Load labels
    df = pd.read_csv(f"data/processed/{dataset_name}_clean.csv")

    # Load split indices
    split_df = pd.read_csv(
        f"splits/{dataset_name}_{split_type}_seed{seed}.csv"
    )

    # Get indices for each split
    train_idx = split_df[split_df['split']=='train']['index'].values
    val_idx   = split_df[split_df['split']=='val']['index'].values
    test_idx  = split_df[split_df['split']=='test']['index'].values

    # Split fingerprints
    X_train = fps[train_idx]
    X_val   = fps[val_idx]
    X_test  = fps[test_idx]

    # Get task column names
    task_cols = [c for c in df.columns if c != 'smiles']

    # Split labels
    y_train = df[task_cols].values[train_idx]
    y_val   = df[task_cols].values[val_idx]
    y_test  = df[task_cols].values[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test, task_cols


def evaluate_predictions(y_true, y_prob, task_names):
    """
    Calculate AUROC, AUPRC, and Brier score for each task.
    Skips tasks with only one class (can't compute AUROC).
    """
    results = {}
    aurocs, auprcs, briers = [], [], []

    for i, task in enumerate(task_names):
        yt = y_true[:, i]
        yp = y_prob[:, i]

        # Remove NaN labels
        mask = ~np.isnan(yt)
        yt = yt[mask]
        yp = yp[mask]

        # Need both classes present
        if len(np.unique(yt)) < 2:
            continue

        auroc = roc_auc_score(yt, yp)
        auprc = average_precision_score(yt, yp)
        brier = brier_score_loss(yt, yp)

        results[task] = {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "brier": round(brier, 4)
        }
        aurocs.append(auroc)
        auprcs.append(auprc)
        briers.append(brier)

    # Macro averages across tasks
    results['MACRO_AVG'] = {
        "auroc": round(np.mean(aurocs), 4),
        "auprc": round(np.mean(auprcs), 4),
        "brier": round(np.mean(briers), 4)
    }

    return results


def train_random_forest(dataset_name, split_type):
    """
    Train Random Forest on one dataset with one split type.
    Tries all hyperparameter configs, picks best on validation AUROC.
    Repeats across 5 seeds for confidence intervals.
    """
    print(f"\n{'='*60}")
    print(f"Random Forest | {dataset_name} | {split_type} split")
    print(f"{'='*60}")

    all_seed_results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test, task_cols = \
            load_data(dataset_name, split_type, seed)

        print(f"    Train: {X_train.shape[0]} | "
              f"Val: {X_val.shape[0]} | "
              f"Test: {X_test.shape[0]}")

        # Try each hyperparameter config
        best_val_auroc = -1
        best_config    = None
        best_model     = None

        for config in RF_CONFIGS:
            # Train one RF per task (multi-task via loop)
            # For simplicity, use first task's val AUROC to pick config
            rf = RandomForestClassifier(
                random_state=seed,
                n_jobs=-1,  # use all CPU cores
                **config
            )

            # For multi-task: train on all tasks jointly
            # Handle NaN labels by using mean imputation for training
            y_train_filled = np.nan_to_num(y_train, nan=0.0)

            rf.fit(X_train, y_train_filled)
            val_probs = rf.predict_proba(X_val)

            # predict_proba returns list of arrays for multi-output
            # Stack into matrix
            if isinstance(val_probs, list):
                val_probs_matrix = np.column_stack(
                    [p[:, 1] for p in val_probs]
                )
            else:
                val_probs_matrix = val_probs[:, 1:2]

            val_results = evaluate_predictions(
                y_val, val_probs_matrix, task_cols
            )
            val_auroc = val_results['MACRO_AVG']['auroc']

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_config    = config
                best_model     = rf

        print(f"    Best config: n_est={best_config['n_estimators']} "
              f"max_depth={best_config['max_depth']} "
              f"val_AUROC={best_val_auroc:.4f}")

        # Evaluate best model on test set
        test_probs = best_model.predict_proba(X_test)
        if isinstance(test_probs, list):
            test_probs_matrix = np.column_stack(
                [p[:, 1] for p in test_probs]
            )
        else:
            test_probs_matrix = test_probs[:, 1:2]

        test_results = evaluate_predictions(
            y_test, test_probs_matrix, task_cols
        )

        macro = test_results['MACRO_AVG']
        print(f"    Test  -> AUROC:{macro['auroc']:.4f} | "
              f"AUPRC:{macro['auprc']:.4f} | "
              f"Brier:{macro['brier']:.4f}")

        all_seed_results.append({
            "seed": seed,
            "best_config": best_config,
            "val_auroc": best_val_auroc,
            "test_results": test_results
        })

    # Summary across seeds
    test_aurocs = [r['test_results']['MACRO_AVG']['auroc']
                   for r in all_seed_results]
    test_auprcs = [r['test_results']['MACRO_AVG']['auprc']
                   for r in all_seed_results]

    print(f"\n  {'='*40}")
    print(f"  FINAL RESULTS ({split_type} split):")
    print(f"  AUROC: {np.mean(test_aurocs):.4f} ± {np.std(test_aurocs):.4f}")
    print(f"  AUPRC: {np.mean(test_auprcs):.4f} ± {np.std(test_auprcs):.4f}")
    print(f"  {'='*40}")

    return all_seed_results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_results = {}

    for dataset in ["tox21", "clintox"]:
        all_results[dataset] = {}
        for split_type in ["random", "scaffold"]:
            results = train_random_forest(dataset, split_type)
            all_results[dataset][split_type] = results

    # Save all results
    out_path = "results/random_forest_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {out_path}")
    print("\nRANDOM FOREST COMPLETE")
