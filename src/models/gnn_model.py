import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from rdkit import Chem
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import brier_score_loss
import os
import json
import warnings
warnings.filterwarnings('ignore')

SEEDS   = [42, 123, 456, 789, 1337]
EPOCHS  = 100
BATCH   = 64
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GNN_CONFIGS = [
    {"hidden_dim": 128, "dropout": 0.1, "lr": 1e-3, "layers": 3},
    {"hidden_dim": 256, "dropout": 0.1, "lr": 1e-3, "layers": 3},
    {"hidden_dim": 128, "dropout": 0.3, "lr": 3e-4, "layers": 3},
    {"hidden_dim": 256, "dropout": 0.3, "lr": 3e-4, "layers": 3},
]

# ── Atom & Bond Featurization ──────────────────────────────

ATOM_FEATURES = {
    'atomic_num':     list(range(1, 119)),
    'degree':         [0,1,2,3,4,5],
    'formal_charge':  [-2,-1,0,1,2],
    'hybridization':  [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
    ],
}

def one_hot(value, choices):
    """Convert a value to a one-hot vector."""
    vec = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)  # unknown → last slot
    vec[idx] = 1
    return vec

def atom_features(atom):
    """Convert RDKit atom to feature vector."""
    feats = []
    feats += one_hot(atom.GetAtomicNum(),
                     ATOM_FEATURES['atomic_num'])
    feats += one_hot(atom.GetDegree(),
                     ATOM_FEATURES['degree'])
    feats += one_hot(atom.GetFormalCharge(),
                     ATOM_FEATURES['formal_charge'])
    feats += one_hot(atom.GetHybridization(),
                     ATOM_FEATURES['hybridization'])
    feats += [
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]
    return feats

def smiles_to_graph(smiles):
    """Convert SMILES string to PyG graph Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (one per atom)
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # Edge indices (bonds, bidirectional)
    edges_src, edges_dst = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges_src += [i, j]
        edges_dst += [j, i]

    if len(edges_src) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(
            [edges_src, edges_dst], dtype=torch.long
        )

    return Data(x=x, edge_index=edge_index)


# ── GIN Model ─────────────────────────────────────────────

class GINModel(nn.Module):
    """
    Graph Isomorphism Network (GIN).
    Reads a molecular graph and predicts toxicity.

    How it works:
    1. Each atom starts with its feature vector
    2. Each GIN layer updates atom features by
       aggregating info from neighboring atoms
    3. After 3 layers, we sum all atom features
       to get one vector for the whole molecule
    4. MLP head converts that to toxicity probabilities
    """
    def __init__(self, input_dim, hidden_dim,
                 output_dim, num_layers, dropout):
        super().__init__()
        self.convs    = nn.ModuleList()
        self.bns      = nn.ModuleList()
        self.dropout  = dropout

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = \
            data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout,
                          training=self.training)

        # Global pooling: sum all atom vectors
        x = global_add_pool(x, batch)
        return torch.sigmoid(self.head(x))


# ── Data Loading ───────────────────────────────────────────

def load_graphs(dataset_name):
    """Pre-convert all molecules to graphs once."""
    df = pd.read_csv(
        f"data/processed/{dataset_name}_clean.csv"
    )
    graphs = []
    failed = 0
    for smi in df['smiles']:
        g = smiles_to_graph(smi)
        if g is None:
            failed += 1
        graphs.append(g)
    if failed:
        print(f"  Warning: {failed} molecules failed graph conversion")
    return graphs, df


def get_split_loader(graphs, df, split_df,
                     split_name, task_cols, batch_size):
    idx = split_df[split_df['split']==split_name]['index'].values
    data_list = []
    for i in idx:
        g = graphs[i]
        if g is None:
            continue
        labels = df[task_cols].values[i]
        g2 = Data(x=g.x, edge_index=g.edge_index,
                  y=torch.tensor(labels, dtype=torch.float).unsqueeze(0))
        data_list.append(g2)
    return DataLoader(data_list, batch_size=batch_size, shuffle=True)


# ── Training ───────────────────────────────────────────────

def evaluate_loader(model, loader, task_cols):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            probs = model(batch).cpu().numpy()
            labels = batch.y.cpu().numpy().reshape(-1, len(task_cols))
            all_probs.append(probs)
            all_labels.append(labels)

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    return evaluate_metrics(y_true, y_prob, task_cols)


def evaluate_metrics(y_true, y_prob, task_names):
    results = {}
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


def train_gnn(dataset_name, split_type):
    print(f"\n{'='*60}")
    print(f"GNN | {dataset_name} | {split_type} | device:{DEVICE}")
    print(f"{'='*60}")

    # Load graphs once (reuse across seeds)
    print("  Converting molecules to graphs...")
    graphs, df = load_graphs(dataset_name)
    task_cols   = [c for c in df.columns if c != 'smiles']
    input_dim   = graphs[0].x.shape[1]
    print(f"  Atom feature dim: {input_dim}")

    all_seed_results = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        print(f"\n  Seed {seed}:")

        split_df = pd.read_csv(
            f"splits/{dataset_name}_{split_type}_seed{seed}.csv"
        )

        train_loader = get_split_loader(
            graphs, df, split_df, 'train', task_cols, BATCH
        )
        val_loader = get_split_loader(
            graphs, df, split_df, 'val', task_cols, BATCH
        )
        test_loader = get_split_loader(
            graphs, df, split_df, 'test', task_cols, BATCH
        )

        n_train = sum(1 for _ in train_loader.dataset)
        n_val   = sum(1 for _ in val_loader.dataset)
        n_test  = sum(1 for _ in test_loader.dataset)
        print(f"    Train:{n_train} | Val:{n_val} | Test:{n_test}")

        best_val_auroc = -1
        best_config    = None
        best_state     = None

        for config in GNN_CONFIGS:
            model = GINModel(
                input_dim  = input_dim,
                hidden_dim = config['hidden_dim'],
                output_dim = len(task_cols),
                num_layers = config['layers'],
                dropout    = config['dropout']
            ).to(DEVICE)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=config['lr']
            )
            criterion = nn.BCELoss()

            patience   = 15
            no_improve = 0
            best_v     = -1
            best_s     = None

            for epoch in range(EPOCHS):
                model.train()
                for batch in train_loader:
                    batch = batch.to(DEVICE)
                    optimizer.zero_grad()
                    preds  = model(batch)
                    labels = batch.y.reshape(-1, len(task_cols))
                    mask   = ~torch.isnan(labels)
                    loss   = criterion(preds[mask], labels[mask])
                    loss.backward()
                    optimizer.step()

                val_res   = evaluate_loader(model, val_loader, task_cols)
                val_auroc = val_res['MACRO_AVG']['auroc']

                if val_auroc > best_v:
                    best_v = val_auroc
                    best_s = {k: v.clone()
                              for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            if best_v > best_val_auroc:
                best_val_auroc = best_v
                best_config    = config
                best_state     = best_s

        # Load best model and evaluate on test
        best_model = GINModel(
            input_dim  = input_dim,
            hidden_dim = best_config['hidden_dim'],
            output_dim = len(task_cols),
            num_layers = best_config['layers'],
            dropout    = best_config['dropout']
        ).to(DEVICE)
        best_model.load_state_dict(best_state)

        test_results = evaluate_loader(
            best_model, test_loader, task_cols
        )
        macro = test_results['MACRO_AVG']

        print(f"    Best: hidden={best_config['hidden_dim']} "
              f"dropout={best_config['dropout']} "
              f"val_AUROC={best_val_auroc:.4f}")
        print(f"    Test -> AUROC:{macro['auroc']:.4f} | "
              f"AUPRC:{macro['auprc']:.4f} | "
              f"Brier:{macro['brier']:.4f}")

        all_seed_results.append({
            "seed":        seed,
            "best_config": best_config,
            "val_auroc":   best_val_auroc,
            "test_results": test_results
        })

    test_aurocs = [r['test_results']['MACRO_AVG']['auroc']
                   for r in all_seed_results]
    test_auprcs = [r['test_results']['MACRO_AVG']['auprc']
                   for r in all_seed_results]

    print(f"\n  {'='*40}")
    print(f"  FINAL RESULTS ({split_type} split):")
    print(f"  AUROC: {np.mean(test_aurocs):.4f} "
          f"+/- {np.std(test_aurocs):.4f}")
    print(f"  AUPRC: {np.mean(test_auprcs):.4f} "
          f"+/- {np.std(test_auprcs):.4f}")
    print(f"  {'='*40}")

    return all_seed_results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print(f"Device: {DEVICE}")
    all_results = {}

    for dataset in ["tox21", "clintox"]:
        all_results[dataset] = {}
        for split_type in ["random", "scaffold"]:
            results = train_gnn(dataset, split_type)
            all_results[dataset][split_type] = results

    out_path = "results/gnn_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {out_path}")
    print("GNN COMPLETE")
