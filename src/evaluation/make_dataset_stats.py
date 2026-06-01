#!/usr/bin/env python3
"""
Generate Table S5: Dataset Statistics for ToxBench paper.
Reads from data/processed/ and splits/ directories.
Outputs: tables/dataset_statistics.csv and tables/dataset_statistics.md

Usage (run from ~/toxbench):
    python src/evaluation/make_dataset_stats.py
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR   = "data/processed"
SPLITS_DIR = "splits"
OUT_DIR    = "tables"

DATASETS = ["tox21", "clintox", "sider"]
SEEDS    = [42, 123, 456, 789, 1337]

def get_split_sizes(dataset, split_type, seeds=SEEDS):
    """Return mean ± sd train/val/test sizes across seeds."""
    trains, vals, tests = [], [], []
    for seed in seeds:
        path = f"{SPLITS_DIR}/{dataset}_{split_type}_seed{seed}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        trains.append((df['split'] == 'train').sum())
        vals.append((df['split'] == 'val').sum())
        tests.append((df['split'] == 'test').sum())
    if not trains:
        return None
    def fmt(vals):
        m, s = int(np.mean(vals)), int(np.std(vals))
        return f"{m}" if s == 0 else f"{m}±{s}"
    return fmt(trains), fmt(vals), fmt(tests)


def count_scaffolds(dataset, split_type, seed=42):
    """Count unique Bemis-Murcko scaffolds in train/test splits."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return "N/A", "N/A"

    df_clean = pd.read_csv(f"{DATA_DIR}/{dataset}_clean.csv")
    split_df = pd.read_csv(f"{SPLITS_DIR}/{dataset}_{split_type}_seed{seed}.csv")

    def get_scaffold(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

    scaffolds = df_clean['smiles'].apply(get_scaffold)

    train_idx = split_df[split_df['split'] == 'train']['index'].values
    test_idx  = split_df[split_df['split'] == 'test']['index'].values

    n_train_scaffolds = scaffolds.iloc[train_idx].nunique()
    n_test_scaffolds  = scaffolds.iloc[test_idx].nunique()
    return n_train_scaffolds, n_test_scaffolds


def dataset_stats(dataset):
    """Collect all statistics for one dataset."""
    clean_path = f"{DATA_DIR}/{dataset}_clean.csv"
    if not os.path.exists(clean_path):
        print(f"  [WARN] Missing {clean_path}")
        return None

    df = pd.read_csv(clean_path)
    task_cols = [c for c in df.columns if c != 'smiles']
    n_compounds = len(df)
    n_tasks = len(task_cols)

    # Positive rate per task (ignoring NaN)
    pos_rates = []
    for col in task_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            pos_rates.append(vals.mean())
    mean_pos_rate = np.mean(pos_rates) if pos_rates else float('nan')

    # Missing label rate
    total_labels = n_compounds * n_tasks
    missing = df[task_cols].isna().sum().sum()
    missing_pct = 100.0 * missing / total_labels if total_labels > 0 else 0.0

    # Split sizes
    rand_sizes = get_split_sizes(dataset, "random")
    scaf_sizes = get_split_sizes(dataset, "scaffold")

    # Scaffold counts (seed 42, scaffold split)
    n_train_scaf, n_test_scaf = count_scaffolds(dataset, "scaffold", seed=42)

    return {
        "Dataset":              dataset.capitalize() if dataset != "sider" else "SIDER",
        "Compounds":            n_compounds,
        "Tasks":                n_tasks,
        "Mean pos. rate":       f"{mean_pos_rate:.1%}",
        "Missing labels (%)":   f"{missing_pct:.1f}%",
        "Random: train/val/test": f"{rand_sizes[0]} / {rand_sizes[1]} / {rand_sizes[2]}" if rand_sizes else "N/A",
        "Scaffold: train/val/test": f"{scaf_sizes[0]} / {scaf_sizes[1]} / {scaf_sizes[2]}" if scaf_sizes else "N/A",
        "Unique scaffolds (train/test)": f"{n_train_scaf} / {n_test_scaf}",
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for ds in DATASETS:
        print(f"Processing {ds}...")
        stats = dataset_stats(ds)
        if stats:
            rows.append(stats)
            print(f"  Compounds: {stats['Compounds']}, Tasks: {stats['Tasks']}, "
                  f"Pos rate: {stats['Mean pos. rate']}")

    if not rows:
        print("No data found. Check DATA_DIR and SPLITS_DIR paths.")
        return

    df_out = pd.DataFrame(rows)
    csv_path = f"{OUT_DIR}/dataset_statistics.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Markdown table (no tabulate required)
    def df_to_markdown(df):
        cols = list(df.columns)
        col_widths = [max(len(str(c)), max(len(str(v)) for v in df[c])) for c in cols]
        def row_str(vals):
            return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths)) + " |"
        sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
        lines = [row_str(cols), sep] + [row_str(df.iloc[i]) for i in range(len(df))]
        return "\n".join(lines)

    md_path = f"{OUT_DIR}/dataset_statistics.md"
    with open(md_path, "w") as f:
        f.write("## Table S5: Dataset Statistics\n\n")
        f.write(df_to_markdown(df_out))
        f.write("\n\n*Split sizes are mean across 5 seeds (42, 123, 456, 789, 1337). "
                "Scaffold counts based on Bemis-Murcko scaffolds, seed 42.*\n")
    print(f"Saved: {md_path}")
    print("\nPreview:")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
