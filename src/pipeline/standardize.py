import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import os
import json

def standardize_smiles(smiles):
    """
    Clean a single SMILES string through 5 steps:
    1. Parse SMILES
    2. Remove salts (keep largest fragment)
    3. Neutralize charges
    4. Generate canonical SMILES
    5. Return None if anything fails
    """
    try:
        # Step 1: Parse
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "unparseable"

        # Step 2: Remove salts - keep largest fragment
        remover = rdMolStandardize.LargestFragmentChooser()
        mol = remover.choose(mol)

        # Step 3: Neutralize charges
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Step 4: Canonical SMILES
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical is None or canonical == "":
            return None, "canonical_failed"

        return canonical, "ok"

    except Exception as e:
        return None, f"error: {str(e)}"


def process_dataset(input_path, output_path, dataset_name):
    """
    Process a full dataset CSV file.
    Reports exactly what happened to every molecule.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    df = pd.read_csv(input_path)
    print(f"Input:  {len(df)} compounds")

    # Track what happens to each molecule
    report = {
        "dataset": dataset_name,
        "input_count": len(df),
        "unparseable": 0,
        "failed_standardization": 0,
        "duplicates_removed": 0,
        "conflicting_labels_removed": 0,
        "final_count": 0
    }

    # --- STEP 1: Standardize each molecule ---
    print("\nStep 1: Standardizing molecules...")
    canonical_smiles = []
    reasons = []

    for smi in df['smiles']:
        canon, reason = standardize_smiles(str(smi))
        canonical_smiles.append(canon)
        reasons.append(reason)

    df['canonical_smiles'] = canonical_smiles
    df['standardization_status'] = reasons

    # Count failures
    failed = df['canonical_smiles'].isna()
    report['unparseable'] = int(failed.sum())
    print(f"  Failed to standardize: {report['unparseable']} compounds")

    # Keep only successful ones
    df_clean = df[~failed].copy()
    print(f"  Remaining: {len(df_clean)} compounds")

    # --- STEP 2: Deduplicate ---
    print("\nStep 2: Deduplicating...")

    # Get task columns (everything except smiles columns and status)
    task_cols = [c for c in df_clean.columns
                 if c not in ['smiles', 'canonical_smiles',
                               'standardization_status']]

    # Find duplicate canonical SMILES
    duplicated_mask = df_clean['canonical_smiles'].duplicated(keep=False)
    n_duplicated_groups = df_clean[duplicated_mask]['canonical_smiles'].nunique()
    print(f"  Canonical SMILES appearing more than once: {n_duplicated_groups} unique structures")

    # For each duplicate group check if labels conflict
    conflicts = []
    keep_indices = []

    for canon_smi, group in df_clean.groupby('canonical_smiles'):
        if len(group) == 1:
            # No duplicate - keep it
            keep_indices.append(group.index[0])
        else:
            # Check if all rows have same labels
            task_values = group[task_cols].values
            # Check for conflicts (ignoring NaN)
            has_conflict = False
            for col in task_cols:
                col_vals = group[col].dropna().unique()
                if len(col_vals) > 1:
                    has_conflict = True
                    break

            if has_conflict:
                # Conflicting labels - remove entire group
                conflicts.append(canon_smi)
                report['conflicting_labels_removed'] += len(group)
            else:
                # Same labels - keep first occurrence
                keep_indices.append(group.index[0])
                report['duplicates_removed'] += len(group) - 1

    print(f"  Exact duplicates removed: {report['duplicates_removed']}")
    print(f"  Conflicting label groups removed: {len(conflicts)} groups "
          f"({report['conflicting_labels_removed']} compounds)")

    # Final clean dataset
    df_final = df_clean.loc[keep_indices].copy()
    df_final = df_final.drop(columns=['smiles', 'standardization_status'])
    df_final = df_final.rename(columns={'canonical_smiles': 'smiles'})

    # Reorder columns: smiles first
    cols = ['smiles'] + task_cols
    df_final = df_final[cols].reset_index(drop=True)

    report['final_count'] = len(df_final)

    # --- STEP 3: Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY for {dataset_name}:")
    print(f"  Input compounds:              {report['input_count']:>6}")
    print(f"  Failed standardization:       {report['unparseable']:>6}")
    print(f"  Exact duplicates removed:     {report['duplicates_removed']:>6}")
    print(f"  Conflicting labels removed:   {report['conflicting_labels_removed']:>6}")
    print(f"  FINAL clean compounds:        {report['final_count']:>6}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}")

    return df_final, report


# ── Run both datasets ──────────────────────────────────────
if __name__ == "__main__":

    all_reports = {}

    # Process Tox21
    tox21_df, tox21_report = process_dataset(
        input_path  = "data/raw/tox21_raw.csv",
        output_path = "data/processed/tox21_clean.csv",
        dataset_name = "Tox21"
    )
    all_reports['tox21'] = tox21_report

    # Process ClinTox
    clintox_df, clintox_report = process_dataset(
        input_path  = "data/raw/clintox_raw.csv",
        output_path = "data/processed/clintox_clean.csv",
        dataset_name = "ClinTox"
    )
    all_reports['clintox'] = clintox_report

    # Save report as JSON (goes in your paper as Table 1)
    report_path = "results/standardization_report.json"
    os.makedirs("results", exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(all_reports, f, indent=2)

    print(f"\nStandardization report saved to: {report_path}")
    print("\nALL DATASETS PROCESSED SUCCESSFULLY")
