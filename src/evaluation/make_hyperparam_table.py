#!/usr/bin/env python3
"""
Generate Table S6: Model Hyperparameter Grid for ToxBench paper.
Reads hyperparameter grids directly from the four model source files.
Outputs: tables/model_hyperparameters.csv and tables/model_hyperparameters.md

Usage (run from ~/toxbench):
    python src/evaluation/make_hyperparam_table.py
"""

import os
import pandas as pd

OUT_DIR = "tables"

# ── Hyperparameter data extracted from model source files ────────────────────
# Source: src/models/{random_forest,xgboost_model,mlp_model,gnn_model}.py

HYPERPARAMS = {
    "Random Forest": {
        "source_file": "src/models/random_forest.py",
        "input_features": "ECFP4 fingerprints (2048 bits)",
        "selection": "Grid search over 4 configs; best val AUROC per seed",
        "grid": [
            {"n_estimators": 100, "max_depth": "None (unlimited)", "class_weight": "balanced"},
            {"n_estimators": 500, "max_depth": "None (unlimited)", "class_weight": "balanced"},
            {"n_estimators": 100, "max_depth": 20,                 "class_weight": "balanced"},
            {"n_estimators": 500, "max_depth": 20,                 "class_weight": "balanced"},
        ],
        "fixed": {
            "max_features":        "sqrt (sklearn default)",
            "bootstrap":           "True (sklearn default)",
            "min_samples_split":   "2 (sklearn default)",
            "min_samples_leaf":    "1 (sklearn default)",
            "n_jobs":              "-1 (all cores)",
            "NaN handling":        "nan_to_num fill with 0.0",
            "Multi-task strategy": "Single RF, multi-output",
        }
    },

    "XGBoost": {
        "source_file": "src/models/xgboost_model.py",
        "input_features": "ECFP4 fingerprints (2048 bits)",
        "selection": "Grid search over 4 configs; best mean val AUROC across tasks",
        "grid": [
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
            {"n_estimators": 200, "learning_rate": 0.10, "max_depth": 6},
            {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3},
            {"n_estimators": 500, "learning_rate": 0.10, "max_depth": 6},
        ],
        "fixed": {
            "objective":           "binary:logistic",
            "eval_metric":         "logloss",
            "scale_pos_weight":    "neg/pos ratio per task",
            "subsample":           "1.0 (default)",
            "colsample_bytree":    "1.0 (default)",
            "n_jobs":              "-1 (all cores)",
            "Multi-task strategy": "One model per task",
        }
    },

    "MLP": {
        "source_file": "src/models/mlp_model.py",
        "input_features": "ECFP4 fingerprints (2048 bits)",
        "selection": "Grid search over 4 configs; best val AUROC with early stopping",
        "grid": [
            {"hidden_dims": "[512, 256]",       "dropout": 0.2, "learning_rate": 0.001},
            {"hidden_dims": "[512, 256]",       "dropout": 0.4, "learning_rate": 0.001},
            {"hidden_dims": "[1024, 512, 256]", "dropout": 0.2, "learning_rate": 0.001},
            {"hidden_dims": "[1024, 512, 256]", "dropout": 0.4, "learning_rate": 3e-4},
        ],
        "fixed": {
            "activation":          "ReLU",
            "output_activation":   "Sigmoid",
            "optimizer":           "Adam",
            "loss":                "BCELoss (masked NaN)",
            "batch_size":          64,
            "max_epochs":          50,
            "early_stopping":      "patience=10 (val AUROC)",
            "device":              "CUDA if available, else CPU",
            "Multi-task strategy": "Single model, all tasks jointly",
        }
    },

    "GIN (GNN)": {
        "source_file": "src/models/gnn_model.py",
        "input_features": "Molecular graph (atom features: atomic num, degree, formal charge, hybridization, aromaticity, ring membership)",
        "selection": "Grid search over 4 configs; best val AUROC with early stopping",
        "grid": [
            {"hidden_dim": 128, "dropout": 0.1, "learning_rate": 0.001,  "num_layers": 3},
            {"hidden_dim": 256, "dropout": 0.1, "learning_rate": 0.001,  "num_layers": 3},
            {"hidden_dim": 128, "dropout": 0.3, "learning_rate": 3e-4,   "num_layers": 3},
            {"hidden_dim": 256, "dropout": 0.3, "learning_rate": 3e-4,   "num_layers": 3},
        ],
        "fixed": {
            "GNN architecture":    "Graph Isomorphism Network (GIN)",
            "Pooling":             "Global add pooling",
            "Batch normalisation": "Per-layer BatchNorm1d",
            "MLP head":            "hidden_dim → hidden_dim/2 → n_tasks",
            "output_activation":   "Sigmoid",
            "optimizer":           "Adam",
            "loss":                "BCELoss (masked NaN)",
            "batch_size":          64,
            "max_epochs":          100,
            "early_stopping":      "patience=15 (val AUROC)",
            "device":              "CUDA if available, else CPU",
            "Multi-task strategy": "Single model, all tasks jointly",
        }
    },
}

SHARED_SETTINGS = {
    "Seeds":            "5 (42, 123, 456, 789, 1337)",
    "Evaluation":       "Macro-averaged AUROC across tasks",
    "NaN labels":       "Masked during loss / evaluation",
    "Val split use":    "Config selection only; test set held out",
    "Fingerprints":     "RDKit ECFP4, radius=2, 2048 bits",
    "Scaffold split":   "Bemis-Murcko (RDKit), deterministic per seed",
    "Random split":     "Stratified by first task, fixed per seed",
}


# ── Flat table for CSV / docx ─────────────────────────────────────────────────

def build_flat_table():
    """One row per (model, hyperparameter, value) — suitable for a 3-col table."""
    rows = []
    for model_name, info in HYPERPARAMS.items():
        # Grid configs
        for i, cfg in enumerate(info["grid"], 1):
            param_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
            rows.append({
                "Model": model_name,
                "Category": f"Config {i}",
                "Value": param_str,
            })
        # Fixed settings
        for k, v in info["fixed"].items():
            rows.append({
                "Model": model_name,
                "Category": k,
                "Value": str(v),
            })
        # Source
        rows.append({
            "Model": model_name,
            "Category": "Source file",
            "Value": info["source_file"],
        })
        rows.append({
            "Model": model_name,
            "Category": "Input features",
            "Value": info["input_features"],
        })
        rows.append({
            "Model": model_name,
            "Category": "Config selection",
            "Value": info["selection"],
        })
    return pd.DataFrame(rows)


def build_grid_summary():
    """Wide table: one row per config, columns = key hyperparameters."""
    rows = []
    for model_name, info in HYPERPARAMS.items():
        for i, cfg in enumerate(info["grid"], 1):
            row = {"Model": model_name, "Config": i}
            row.update(cfg)
            rows.append(row)
    return pd.DataFrame(rows).fillna("—")


def df_to_markdown(df):
    """Write a DataFrame as a markdown table without requiring tabulate."""
    cols = list(df.columns)
    col_widths = [max(len(str(c)), max(len(str(v)) for v in df[c])) for c in cols]
    def row_str(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines = [row_str(cols), sep] + [row_str(df.iloc[i]) for i in range(len(df))]
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # CSV: flat 3-column version
    df_flat = build_flat_table()
    csv_path = f"{OUT_DIR}/model_hyperparameters.csv"
    df_flat.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Markdown: grid summary + fixed settings
    md_path = f"{OUT_DIR}/model_hyperparameters.md"
    with open(md_path, "w") as f:
        f.write("## Table S6: Model Hyperparameters\n\n")

        f.write("### Hyperparameter Grids\n\n")
        f.write("*Best configuration selected per seed by validation AUROC.*\n\n")
        df_grid = build_grid_summary()
        f.write(df_to_markdown(df_grid))
        f.write("\n\n")

        f.write("### Fixed Settings per Model\n\n")
        for model_name, info in HYPERPARAMS.items():
            f.write(f"**{model_name}** (`{info['source_file']}`)\n\n")
            f.write(f"Input: {info['input_features']}\n\n")
            for k, v in info["fixed"].items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

        f.write("### Shared Settings (All Models)\n\n")
        for k, v in SHARED_SETTINGS.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n")

    print(f"Saved: {md_path}")
    print("\nGrid summary preview:")
    print(build_grid_summary().to_string(index=False))


if __name__ == "__main__":
    main()
