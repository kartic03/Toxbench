import deepchem as dc
import pandas as pd
import os

# Create save directory
save_dir = os.path.expanduser("~/toxbench/data/raw")
os.makedirs(save_dir, exist_ok=True)

print("="*50)
print("Downloading Tox21 dataset...")
print("="*50)

# Download Tox21
tox21_tasks, tox21_datasets, _ = dc.molnet.load_tox21(
    splitter=None,
    featurizer='Raw'
)

# Get all data as one dataset
tox21_data = tox21_datasets[0]

# Save to CSV
tox21_df = pd.DataFrame(
    tox21_data.y,
    columns=tox21_tasks
)
tox21_df.insert(0, 'smiles', tox21_data.ids)

tox21_path = os.path.join(save_dir, "tox21_raw.csv")
tox21_df.to_csv(tox21_path, index=False)

print(f"Tox21 saved: {tox21_df.shape[0]} compounds, {len(tox21_tasks)} tasks")
print(f"Tasks: {tox21_tasks}")
print(f"Saved to: {tox21_path}")

print()
print("="*50)
print("Downloading ClinTox dataset...")
print("="*50)

# Download ClinTox
clintox_tasks, clintox_datasets, _ = dc.molnet.load_clintox(
    splitter=None,
    featurizer='Raw'
)

clintox_data = clintox_datasets[0]

clintox_df = pd.DataFrame(
    clintox_data.y,
    columns=clintox_tasks
)
clintox_df.insert(0, 'smiles', clintox_data.ids)

clintox_path = os.path.join(save_dir, "clintox_raw.csv")
clintox_df.to_csv(clintox_path, index=False)

print(f"ClinTox saved: {clintox_df.shape[0]} compounds, {len(clintox_tasks)} tasks")
print(f"Tasks: {clintox_tasks}")
print(f"Saved to: {clintox_path}")

print()
print("="*50)
print("ALL DATASETS DOWNLOADED SUCCESSFULLY")
print("="*50)
