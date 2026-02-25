import deepchem as dc
import pandas as pd
import os

save_dir = os.path.expanduser("~/toxbench/data/raw")
os.makedirs(save_dir, exist_ok=True)

print("="*50)
print("Downloading SIDER dataset...")
print("="*50)

sider_tasks, sider_datasets, _ = dc.molnet.load_sider(
    splitter=None,
    featurizer='Raw'
)

sider_data = sider_datasets[0]

sider_df = pd.DataFrame(
    sider_data.y,
    columns=sider_tasks
)
sider_df.insert(0, 'smiles', sider_data.ids)

sider_path = os.path.join(save_dir, "sider_raw.csv")
sider_df.to_csv(sider_path, index=False)

print(f"SIDER saved: {sider_df.shape[0]} compounds, "
      f"{len(sider_tasks)} tasks")
print(f"Saved to: {sider_path}")
print("DONE")
