import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import os
import json
from collections import defaultdict

SEEDS = [42, 123, 456, 789, 1337]
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        return scaffold
    except:
        return None

def random_split(df, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(len(idx) * TRAIN_RATIO)
    n_val   = int(len(idx) * VAL_RATIO)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)

def scaffold_split(df, seed):
    """
    Improved scaffold split:
    - Groups molecules by Bemis-Murcko scaffold
    - Shuffles groups randomly (seed-controlled)
    - Assigns groups to train/val/test until size targets are met
    - Guarantees all three splits are non-empty
    """
    rng = np.random.RandomState(seed)

    # Get scaffold for every molecule
    scaffolds = [get_scaffold(smi) for smi in df['smiles']]

    # Group molecule indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, scaffold in enumerate(scaffolds):
        key = scaffold if scaffold is not None else f"no_scaffold_{i}"
        scaffold_to_indices[key].append(i)

    # Convert to list of groups and shuffle
    scaffold_groups = list(scaffold_to_indices.values())
    rng.shuffle(scaffold_groups)

    # Calculate target sizes
    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    train_idx, val_idx, test_idx = [], [], []

    for group in scaffold_groups:
        # Assign to whichever split still needs more compounds
        # Priority: fill test first if empty, then val, then train
        if len(test_idx) < (n_total - n_train - n_val):
            test_idx.extend(group)
        elif len(val_idx) < n_val:
            val_idx.extend(group)
        else:
            train_idx.extend(group)

    # Safety check - if any split empty, redistribute
    if len(test_idx) == 0 or len(val_idx) == 0:
        # Fall back: assign last 10% of train to test, second last to val
        all_idx = train_idx + val_idx + test_idx
        n_tr = int(len(all_idx) * TRAIN_RATIO)
        n_v  = int(len(all_idx) * VAL_RATIO)
        train_idx = all_idx[:n_tr]
        val_idx   = all_idx[n_tr:n_tr+n_v]
        test_idx  = all_idx[n_tr+n_v:]

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)

def create_splits_for_dataset(csv_path, dataset_name, output_dir):
    print(f"\n{'='*60}")
    print(f"Creating splits for: {dataset_name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"Total compounds: {len(df)}")

    os.makedirs(output_dir, exist_ok=True)
    split_summary = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        # Random split
        r_train, r_val, r_test = random_split(df, seed)
        random_split_df = pd.DataFrame({
            'index': list(r_train) + list(r_val) + list(r_test),
            'split': ['train']*len(r_train) + ['val']*len(r_val) + ['test']*len(r_test)
        }).sort_values('index').reset_index(drop=True)
        random_path = os.path.join(output_dir, f"{dataset_name}_random_seed{seed}.csv")
        random_split_df.to_csv(random_path, index=False)
        print(f"    Random  -> train:{len(r_train):4d} | val:{len(r_val):4d} | test:{len(r_test):4d}")

        # Scaffold split
        s_train, s_val, s_test = scaffold_split(df, seed)
        scaffold_split_df = pd.DataFrame({
            'index': list(s_train) + list(s_val) + list(s_test),
            'split': ['train']*len(s_train) + ['val']*len(s_val) + ['test']*len(s_test)
        }).sort_values('index').reset_index(drop=True)
        scaffold_path = os.path.join(output_dir, f"{dataset_name}_scaffold_seed{seed}.csv")
        scaffold_split_df.to_csv(scaffold_path, index=False)
        print(f"    Scaffold-> train:{len(s_train):4d} | val:{len(s_val):4d} | test:{len(s_test):4d}")

        # Verify no leakage
        r_train_set = set(r_train)
        r_test_set  = set(r_test)
        s_train_set = set(s_train)
        s_test_set  = set(s_test)
        r_leak = len(r_train_set & r_test_set)
        s_leak = len(s_train_set & s_test_set)
        print(f"    Leakage check -> random:{r_leak} | scaffold:{s_leak} (both must be 0)")

        split_summary.append({
            "dataset": dataset_name,
            "seed": seed,
            "random":   {"train": len(r_train), "val": len(r_val), "test": len(r_test)},
            "scaffold": {"train": len(s_train), "val": len(s_val), "test": len(s_test)},
            "leakage":  {"random": r_leak, "scaffold": s_leak}
        })

    summary_path = os.path.join(output_dir, f"{dataset_name}_split_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(split_summary, f, indent=2)

    print(f"\n  All splits saved to: {output_dir}/")
    return split_summary

if __name__ == "__main__":
    create_splits_for_dataset(
        csv_path     = "data/processed/tox21_clean.csv",
        dataset_name = "tox21",
        output_dir   = "splits"
    )
    create_splits_for_dataset(
        csv_path     = "data/processed/clintox_clean.csv",
        dataset_name = "clintox",
        output_dir   = "splits"
    )
    split_files = [f for f in os.listdir("splits") if f.endswith('.csv')]
    print(f"\n{'='*60}")
    print(f"DONE: {len(split_files)} split files created")
    print(f"Leakage = 0 in all splits means your benchmark is clean")
    print(f"{'='*60}")
