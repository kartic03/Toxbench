import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import brier_score_loss
import os
import json
import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 123, 456, 789, 1337]

# Hyperparameter grid
MLP_CONFIGS = [
    {"hidden_dims": [512, 256],       "dropout": 0.2, "lr": 1e-3},
    {"hidden_dims": [512, 256],       "dropout": 0.4, "lr": 1e-3},
    {"hidden_dims": [1024, 512, 256], "dropout": 0.2, "lr": 1e-3},
    {"hidden_dims": [1024, 512, 256], "dropout": 0.4, "lr": 3e-4},
]

EPOCHS    = 50
BATCH     = 64
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """
    Simple feedforward neural network.
    Input: 2048 fingerprint bits
    Hidden: configurable layers with ReLU + Dropout
    Output: one probability per task
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.network(x))


def load_data(dataset_name, split_type, seed):
    fps      = np.load(f"data/processed/{dataset_name}_ecfp4.npy")
    df       = pd.read_csv(f"data/processed/{dataset_name}_clean.csv")
    split_df = pd.read_csv(
        f"splits/{dataset_name}_{split_type}_seed{seed}.csv"
    )
    train_idx = split_df[split_df['split']=='train']['index'].values
    val_idx   = split_df[split_df['split']=='val']['index'].values
    test_idx  = split_df[split_df['split']=='test']['index'].values

    X_train = fps[train_idx].astype(np.float32)
    X_val   = fps[val_idx].astype(np.float32)
    X_test  = fps[test_idx].astype(np.float32)

    task_cols = [c for c in df.columns if c != 'smiles']
    y_train = df[task_cols].values[train_idx].astype(np.float32)
    y_val   = df[task_cols].values[val_idx].astype(np.float32)
    y_test  = df[task_cols].values[test_idx].astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test, task_cols


def evaluate_predictions(y_true, y_prob, task_names):
    results  = {}
    aurocs, auprcs, briers = [], [], []

    for i, task in enumerate(task_names):
        yt   = y_true[:, i]
        yp   = y_prob[:, i]
        mask = ~np.isnan(yt)
        yt, yp = yt[mask], yp[mask]
        if len(np.unique(yt)) < 2:
            continue
        auroc = roc_auc_score(yt, yp)
        auprc = average_precision_score(yt, yp)
        brier = brier_score_loss(yt, yp)
        results[task] = {
            "auroc": round(float(auroc), 4),
            "auprc": round(float(auprc), 4),
            "brier": round(float(brier), 4)
        }
        aurocs.append(auroc)
        auprcs.append(auprc)
        briers.append(brier)

    results['MACRO_AVG'] = {
        "auroc": round(float(np.mean(aurocs)), 4),
        "auprc": round(float(np.mean(auprcs)), 4),
        "brier": round(float(np.mean(briers)), 4)
    }
    return results


def train_one_mlp(X_train, y_train, X_val, y_val, config, seed, n_tasks):
    """Train one MLP with given config. Returns model + val AUROC."""
    torch.manual_seed(seed)

    model = MLP(
        input_dim   = X_train.shape[1],
        hidden_dims = config['hidden_dims'],
        output_dim  = n_tasks,
        dropout     = config['dropout']
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCELoss()

    # Convert to tensors
    Xt = torch.FloatTensor(X_train).to(DEVICE)
    yt = torch.FloatTensor(np.nan_to_num(y_train, nan=0.0)).to(DEVICE)
    Xv = torch.FloatTensor(X_val).to(DEVICE)

    dataset  = TensorDataset(Xt, yt)
    loader   = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    best_val_auroc = -1
    best_state     = None
    patience       = 10
    no_improve     = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            # Mask NaN labels
            mask  = ~torch.isnan(yb)
            loss  = criterion(preds[mask], yb[mask])
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = model(Xv).cpu().numpy()

        val_results  = evaluate_predictions(y_val, val_probs,
                                             [f"t{i}" for i in range(n_tasks)])
        val_auroc    = val_results['MACRO_AVG']['auroc']

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state     = {k: v.clone()
                              for k, v in model.state_dict().items()}
            no_improve     = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best weights
    model.load_state_dict(best_state)
    return model, best_val_auroc


def train_mlp(dataset_name, split_type):
    print(f"\n{'='*60}")
    print(f"MLP | {dataset_name} | {split_type} split | device: {DEVICE}")
    print(f"{'='*60}")

    all_seed_results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        X_train, y_train, X_val, y_val, X_test, y_test, task_cols = \
            load_data(dataset_name, split_type, seed)

        print(f"    Train:{X_train.shape[0]} | "
              f"Val:{X_val.shape[0]} | "
              f"Test:{X_test.shape[0]}")

        n_tasks        = len(task_cols)
        best_val_auroc = -1
        best_config    = None
        best_model     = None

        for config in MLP_CONFIGS:
            model, val_auroc = train_one_mlp(
                X_train, y_train, X_val, y_val,
                config, seed, n_tasks
            )
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_config    = config
                best_model     = model

        # Test evaluation
        best_model.eval()
        Xtest_t = torch.FloatTensor(X_test).to(DEVICE)
        with torch.no_grad():
            test_probs = best_model(Xtest_t).cpu().numpy()

        test_results = evaluate_predictions(y_test, test_probs, task_cols)
        macro        = test_results['MACRO_AVG']

        print(f"    Best: hidden={best_config['hidden_dims']} "
              f"dropout={best_config['dropout']} "
              f"val_AUROC={best_val_auroc:.4f}")
        print(f"    Test -> AUROC:{macro['auroc']:.4f} | "
              f"AUPRC:{macro['auprc']:.4f} | "
              f"Brier:{macro['brier']:.4f}")

        all_seed_results.append({
            "seed":        seed,
            "best_config": {
                "hidden_dims": best_config['hidden_dims'],
                "dropout":     best_config['dropout'],
                "lr":          best_config['lr']
            },
            "val_auroc":    best_val_auroc,
            "test_results": test_results
        })

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
    print(f"Using device: {DEVICE}")
    all_results = {}

    for dataset in ["tox21", "clintox"]:
        all_results[dataset] = {}
        for split_type in ["random", "scaffold"]:
            results = train_mlp(dataset, split_type)
            all_results[dataset][split_type] = results

    out_path = "results/mlp_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {out_path}")
    print("MLP COMPLETE")
