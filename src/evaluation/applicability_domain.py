import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
import json
import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 123, 456, 789, 1337]


def compute_ecfp4(smiles_list, radius=2, nbits=2048):
    """Compute ECFP4 fingerprints."""
    fps = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(np.zeros(nbits))
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=radius, nBits=nbits
                )
                fps.append(np.array(fp))
        except:
            fps.append(np.zeros(nbits))
    return np.array(fps)


def tanimoto_similarity(fp1, fp2):
    """
    Tanimoto similarity between two binary fingerprints.
    Range: 0.0 (completely different) to 1.0 (identical)
    This is the standard similarity metric in cheminformatics.
    """
    intersection = np.sum(fp1 & fp2.astype(bool), axis=1)
    union = np.sum(fp1 | fp2.astype(bool), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = np.where(union > 0, intersection / union, 0.0)
    return sim


def nearest_neighbor_similarity(train_fps, test_fps, batch_size=500):
    """
    For each test compound, find its most similar
    training compound (nearest neighbor Tanimoto).
    Returns array of max similarities for each test compound.

    Uses batching to avoid memory issues.
    """
    print(f"    Computing Tanimoto similarities "
          f"({len(test_fps)} test vs {len(train_fps)} train)...")

    train_bool = train_fps.astype(bool)
    max_sims   = np.zeros(len(test_fps))

    for start in range(0, len(test_fps), batch_size):
        end      = min(start + batch_size, len(test_fps))
        batch    = test_fps[start:end].astype(bool)

        # Compute similarity of batch against all training fps
        batch_sims = np.zeros((end - start, len(train_fps)))
        for j, test_fp in enumerate(batch):
            intersection = np.sum(train_bool & test_fp, axis=1)
            union        = np.sum(train_bool | test_fp, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                sims = np.where(union > 0,
                                intersection / union, 0.0)
            batch_sims[j] = sims

        max_sims[start:end] = batch_sims.max(axis=1)

        if start % 1000 == 0:
            print(f"      Processed {end}/{len(test_fps)}")

    return max_sims


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


def analyze_ad(dataset_name, split_type):
    """
    Applicability Domain Analysis:
    1. Train RF model
    2. Compute nearest-neighbor Tanimoto for each test compound
    3. Bin test compounds by similarity (0-0.2, 0.2-0.4, etc.)
    4. Compute AUROC in each bin
    5. Show: low similarity = low AUROC = model not trustworthy
    """
    print(f"\n{'='*60}")
    print(f"AD Analysis | {dataset_name} | {split_type} split")
    print(f"{'='*60}")

    # Use only seed 42 for AD analysis (for speed)
    # In paper: note this is representative
    all_sims    = []
    all_errors  = []
    bin_results = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        X_train, y_train, X_val, y_val, \
        X_test, y_test, task_cols = \
            load_data(dataset_name, split_type, seed)

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X_train, np.nan_to_num(y_train, nan=0.0))

        # Get predictions
        test_probs = rf.predict_proba(X_test)
        if isinstance(test_probs, list):
            test_probs = np.column_stack(
                [p[:, 1] for p in test_probs]
            )
        else:
            test_probs = test_probs[:, 1:2]

        # Compute nearest-neighbor Tanimoto similarity
        nn_sims = nearest_neighbor_similarity(X_train, X_test)
        print(f"    NN similarity stats: "
              f"min={nn_sims.min():.3f} "
              f"mean={nn_sims.mean():.3f} "
              f"max={nn_sims.max():.3f}")

        # Bin by similarity and compute AUROC per bin
        bins       = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        bin_labels = ['0.0-0.2','0.2-0.4',
                      '0.4-0.6','0.6-0.8','0.8-1.0']
        seed_bin_results = []

        print(f"    {'Bin':<12} {'N':>5} {'AUROC':>8}")
        print(f"    {'-'*28}")

        for i in range(len(bins)-1):
            lo, hi = bins[i], bins[i+1]
            mask   = (nn_sims >= lo) & (nn_sims < hi)
            n_bin  = mask.sum()

            if n_bin < 5:
                print(f"    {bin_labels[i]:<12} "
                      f"{n_bin:>5}  (too few)")
                seed_bin_results.append({
                    "bin":   bin_labels[i],
                    "n":     int(n_bin),
                    "auroc": None
                })
                continue

            # Compute AUROC for this bin across all tasks
            bin_aurocs = []
            for j, task in enumerate(task_cols):
                yt   = y_test[mask, j]
                yp   = test_probs[mask, j]
                vmask = ~np.isnan(yt)
                yt, yp = yt[vmask], yp[vmask]
                if len(np.unique(yt)) < 2:
                    continue
                try:
                    bin_aurocs.append(roc_auc_score(yt, yp))
                except:
                    pass

            if bin_aurocs:
                bin_auroc = np.mean(bin_aurocs)
                print(f"    {bin_labels[i]:<12} "
                      f"{n_bin:>5}  {bin_auroc:.4f}")
                seed_bin_results.append({
                    "bin":   bin_labels[i],
                    "n":     int(n_bin),
                    "auroc": round(float(bin_auroc), 4)
                })
            else:
                seed_bin_results.append({
                    "bin":   bin_labels[i],
                    "n":     int(n_bin),
                    "auroc": None
                })

        # Store per-compound data for plotting
        all_sims.extend(nn_sims.tolist())
        bin_results.append(seed_bin_results)

    return {
        "bin_results":    bin_results,
        "all_sims_seed0": all_sims[:len(all_sims)//len(SEEDS)]
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_ad_results = {}

    for dataset in ["tox21", "clintox"]:
        all_ad_results[dataset] = {}
        for split_type in ["random", "scaffold"]:
            res = analyze_ad(dataset, split_type)
            all_ad_results[dataset][split_type] = res

    out_path = "results/ad_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_ad_results, f, indent=2)

    print(f"\nAD results saved to: {out_path}")
    print("APPLICABILITY DOMAIN ANALYSIS COMPLETE")
