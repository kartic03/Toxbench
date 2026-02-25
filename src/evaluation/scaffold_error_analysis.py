import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "invalid"
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
    except:
        return "invalid"

def load_data(dataset_name, split_type, seed):
    fps      = np.load(f"data/processed/{dataset_name}_ecfp4.npy")
    df       = pd.read_csv(f"data/processed/{dataset_name}_clean.csv")
    split_df = pd.read_csv(f"splits/{dataset_name}_{split_type}_seed{seed}.csv")
    train_idx = split_df[split_df['split']=='train']['index'].values
    test_idx  = split_df[split_df['split']=='test']['index'].values
    task_cols = [c for c in df.columns if c != 'smiles']
    return (fps[train_idx], df[task_cols].values[train_idx],
            fps[test_idx],  df[task_cols].values[test_idx],
            df['smiles'].values[test_idx], task_cols)

def analyze_scaffold_errors(dataset_name):
    print(f"\n{'='*60}")
    print(f"Scaffold Error Analysis: {dataset_name}")
    print(f"{'='*60}")

    seed       = 42
    split_type = 'scaffold'

    (X_train, y_train, X_test, y_test,
     test_smiles, task_cols) = load_data(
        dataset_name, split_type, seed)

    print("  Computing scaffolds for test set...")
    scaffolds = [get_scaffold(smi) for smi in test_smiles]

    print("  Training RF model...")
    rf = RandomForestClassifier(
        n_estimators=500, class_weight='balanced',
        random_state=seed, n_jobs=-1)
    rf.fit(X_train, np.nan_to_num(y_train, nan=0.0))
    test_probs = rf.predict_proba(X_test)
    if isinstance(test_probs, list):
        test_probs = np.column_stack([p[:, 1] for p in test_probs])
    else:
        test_probs = test_probs[:, 1:2]

    scaffold_groups = {}
    for i, scaffold in enumerate(scaffolds):
        scaffold_groups.setdefault(scaffold, []).append(i)

    print(f"  Unique scaffolds in test set: {len(scaffold_groups)}")

    scaffold_results = []
    for scaffold, indices in scaffold_groups.items():
        if len(indices) < 3:
            continue
        indices    = np.array(indices)
        yt_group   = y_test[indices]
        yp_group   = test_probs[indices]
        task_aurocs = []
        for j in range(len(task_cols)):
            yt   = yt_group[:, j]
            yp   = yp_group[:, j]
            mask = ~np.isnan(yt)
            yt, yp = yt[mask], yp[mask]
            if len(np.unique(yt)) < 2:
                continue
            try:
                task_aurocs.append(roc_auc_score(yt, yp))
            except:
                pass
        if not task_aurocs:
            continue
        scaffold_results.append({
            'scaffold':          scaffold,
            'n_compounds':       len(indices),
            'mean_auroc':        round(float(np.mean(task_aurocs)), 4),
            'n_tasks_evaluated': len(task_aurocs)
        })

    scaffold_results.sort(key=lambda x: x['mean_auroc'])

    print(f"\n  TOP 10 WORST SCAFFOLDS:")
    print(f"  {'Scaffold':<45} {'N':>4} {'AUROC':>7}")
    print(f"  {'-'*58}")
    for r in scaffold_results[:10]:
        sd = r['scaffold'][:42]+'...' if len(r['scaffold'])>45 else r['scaffold']
        print(f"  {sd:<45} {r['n_compounds']:>4} {r['mean_auroc']:>7.4f}")

    print(f"\n  TOP 10 BEST SCAFFOLDS:")
    print(f"  {'Scaffold':<45} {'N':>4} {'AUROC':>7}")
    print(f"  {'-'*58}")
    for r in sorted(scaffold_results, key=lambda x: x['mean_auroc'], reverse=True)[:10]:
        sd = r['scaffold'][:42]+'...' if len(r['scaffold'])>45 else r['scaffold']
        print(f"  {sd:<45} {r['n_compounds']:>4} {r['mean_auroc']:>7.4f}")

    all_aurocs = [r['mean_auroc'] for r in scaffold_results]
    print(f"\n  Summary:")
    print(f"  Total scaffolds analyzed: {len(scaffold_results)}")
    print(f"  Mean AUROC:               {np.mean(all_aurocs):.4f}")
    print(f"  Failing (AUROC < 0.6):    {sum(1 for a in all_aurocs if a < 0.6)}")
    print(f"  Good    (AUROC > 0.8):    {sum(1 for a in all_aurocs if a > 0.8)}")

    return {
        'worst_scaffolds': scaffold_results[:10],
        'best_scaffolds':  sorted(scaffold_results, key=lambda x: x['mean_auroc'], reverse=True)[:10],
        'summary': {
            'total_scaffolds': len(scaffold_results),
            'mean_auroc':      round(float(np.mean(all_aurocs)), 4),
            'failing_count':   int(sum(1 for a in all_aurocs if a < 0.6)),
            'good_count':      int(sum(1 for a in all_aurocs if a > 0.8))
        }
    }

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    all_results = {}
    for dataset in ["tox21", "clintox", "sider"]:
        res = analyze_scaffold_errors(dataset)
        all_results[dataset] = res
    with open("results/scaffold_error_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nScaffold error results saved.")
    print("SCAFFOLD ERROR ANALYSIS COMPLETE")
