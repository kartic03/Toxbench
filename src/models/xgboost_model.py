import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import brier_score_loss
import os
import json
import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 123, 456, 789, 1337]

# Small hyperparameter grid
XGB_CONFIGS = [
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.1,  "max_depth": 6},
    {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 500, "learning_rate": 0.1,  "max_depth": 6},
]

def load_data(dataset_name, split_type, seed):
    fps = np.load(f"data/processed/{dataset_name}_ecfp4.npy")
    df  = pd.read_csv(f"data/processed/{dataset_name}_clean.csv")
    split_df = pd.read_csv(
        f"splits/{dataset_name}_{split_type}_seed{seed}.csv"
    )
    train_idx = split_df[split_df['split']=='train']['index'].values
    val_idx   = split_df[split_df['split']=='val']['index'].values
    test_idx  = split_df[split_df['split']=='test']['index'].values

    X_train, X_val, X_test = fps[train_idx], fps[val_idx], fps[test_idx]
    task_cols = [c for c in df.columns if c != 'smiles']
    y_train = df[task_cols].values[train_idx]
    y_val   = df[task_cols].values[val_idx]
    y_test  = df[task_cols].values[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test, task_cols


def evaluate_predictions(y_true, y_prob, task_names):
    results = {}
    aurocs, auprcs, briers = [], [], []

    for i, task in enumerate(task_names):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        mask = ~np.isnan(yt)
        yt, yp = yt[mask], yp[mask]
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

    results['MACRO_AVG'] = {
        "auroc": round(np.mean(aurocs), 4),
        "auprc": round(np.mean(auprcs), 4),
        "brier": round(np.mean(briers), 4)
    }
    return results


def train_per_task_xgb(X_train, y_train, X_val, y_val,
                        X_test, y_test, task_cols, seed):
    """
    XGBoost trains one model per task (unlike RF which
    handles multi-output natively).
    We pick best config using average val AUROC across tasks.
    """
    best_val_auroc = -1
    best_config    = None
    best_models    = None

    for config in XGB_CONFIGS:
        models = []
        val_aurocs = []

        for i, task in enumerate(task_cols):
            yt_train = y_train[:, i]
            yt_val   = y_val[:, i]

            # Remove NaN
            train_mask = ~np.isnan(yt_train)
            val_mask   = ~np.isnan(yt_val)

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                models.append(None)
                continue

            xt = X_train[train_mask]
            yt = yt_train[train_mask]
            xv = X_val[val_mask]
            yv = yt_val[val_mask]

            # Handle class imbalance
            pos = yt.sum()
            neg = len(yt) - pos
            scale = neg / pos if pos > 0 else 1.0

            model = XGBClassifier(
                random_state=seed,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=scale,
                verbosity=0,
                **config
            )
            model.fit(xt, yt)
            models.append(model)

            if len(np.unique(yv)) >= 2:
                vp = model.predict_proba(xv)[:, 1]
                val_aurocs.append(roc_auc_score(yv, vp))

        avg_val_auroc = np.mean(val_aurocs) if val_aurocs else 0

        if avg_val_auroc > best_val_auroc:
            best_val_auroc = avg_val_auroc
            best_config    = config
            best_models    = models

    # Get test predictions using best models
    test_probs = np.zeros((len(X_test), len(task_cols)))
    for i, model in enumerate(best_models):
        if model is not None:
            test_probs[:, i] = model.predict_proba(X_test)[:, 1]

    return test_probs, best_val_auroc, best_config


def train_xgboost(dataset_name, split_type):
    print(f"\n{'='*60}")
    print(f"XGBoost | {dataset_name} | {split_type} split")
    print(f"{'='*60}")

    all_seed_results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        X_train, y_train, X_val, y_val, X_test, y_test, task_cols = \
            load_data(dataset_name, split_type, seed)

        print(f"    Train:{X_train.shape[0]} | "
              f"Val:{X_val.shape[0]} | "
              f"Test:{X_test.shape[0]}")

        test_probs, best_val_auroc, best_config = train_per_task_xgb(
            X_train, y_train, X_val, y_val,
            X_test, y_test, task_cols, seed
        )

        test_results = evaluate_predictions(y_test, test_probs, task_cols)
        macro = test_results['MACRO_AVG']

        print(f"    Best config: lr={best_config['learning_rate']} "
              f"depth={best_config['max_depth']} "
              f"val_AUROC={best_val_auroc:.4f}")
        print(f"    Test -> AUROC:{macro['auroc']:.4f} | "
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
            results = train_xgboost(dataset, split_type)
            all_results[dataset][split_type] = results

    out_path = "results/xgboost_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {out_path}")
    print("XGBOOST COMPLETE")
