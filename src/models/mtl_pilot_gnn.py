"""
Cross-dataset MTL pilot -- GNN (GIN) variant (revision R1.5, second architecture).

Same design as mtl_pilot.py but with a Graph Isomorphism Network instead of an
MLP, to show the cross-dataset transfer effect is not architecture-specific.

  Arm A (baseline): GIN trained on ClinTox only (2 tasks).
  Arm B (MTL)     : GIN trained jointly on Tox21+SIDER+ClinTox (41 tasks).

Evaluated on identical ClinTox scaffold test compounds across 5 seeds; ClinTox
test+val compounds removed from all training (SMILES-matched) to block leakage.
CPU-tuned: 2 configs, 40 epochs, patience 8.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from rdkit import Chem
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")
SPLITS = os.path.join(ROOT, "splits")
OUT = os.path.join(ROOT, "supplementary_files")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1337]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BATCH, PATIENCE = 40, 64, 8
CONFIGS = [
    {"hidden_dim": 128, "dropout": 0.1, "lr": 1e-3, "layers": 3},
    {"hidden_dim": 256, "dropout": 0.1, "lr": 1e-3, "layers": 3},
]
CLINTOX_TASKS = ["FDA_APPROVED", "CT_TOX"]

ATOM = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [Chem.rdchem.HybridizationType.SP,
                      Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3],
}

def one_hot(v, choices):
    vec = [0] * (len(choices) + 1)
    vec[choices.index(v) if v in choices else len(choices)] = 1
    return vec

def atom_features(a):
    f = []
    f += one_hot(a.GetAtomicNum(), ATOM["atomic_num"])
    f += one_hot(a.GetDegree(), ATOM["degree"])
    f += one_hot(a.GetFormalCharge(), ATOM["formal_charge"])
    f += one_hot(a.GetHybridization(), ATOM["hybridization"])
    f += [int(a.GetIsAromatic()), int(a.IsInRing())]
    return f

def smiles_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    ei = (torch.tensor([src, dst], dtype=torch.long) if src
          else torch.zeros((2, 0), dtype=torch.long))
    return Data(x=x, edge_index=ei)


class GIN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers, dropout):
        super().__init__()
        self.convs, self.bns, self.dropout = nn.ModuleList(), nn.ModuleList(), dropout
        for i in range(layers):
            d = in_dim if i == 0 else hidden
            mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, out_dim))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(F.relu(bn(conv(x, ei))), p=self.dropout, training=self.training)
        return torch.sigmoid(self.head(global_add_pool(x, batch)))


def build_combined():
    frames = {ds: pd.read_csv(os.path.join(PROC, f"{ds}_clean.csv"))
              for ds in ["tox21", "sider", "clintox"]}
    task_order = []
    for ds in ["tox21", "sider", "clintox"]:
        task_order += [c for c in frames[ds].columns if c != "smiles"]
    all_smiles = sorted(set().union(*[set(f["smiles"]) for f in frames.values()]))
    comb = pd.DataFrame({"smiles": all_smiles}).set_index("smiles")
    for t in task_order:
        comb[t] = np.nan
    for ds, f in frames.items():
        f2 = f.set_index("smiles")
        for t in [c for c in f.columns if c != "smiles"]:
            comb.loc[f2.index, t] = f2[t].values
    return comb.reset_index(), task_order


def make_loader(graphs, smiles, Ymat, idx, n_tasks, shuffle):
    dl = []
    for i in idx:
        g = graphs[smiles[i]]
        if g is None:
            continue
        y = torch.tensor(Ymat[i], dtype=torch.float).unsqueeze(0)
        dl.append(Data(x=g.x, edge_index=g.edge_index, y=y))
    return DataLoader(dl, batch_size=BATCH, shuffle=shuffle)


def eval_loader(model, loader, n_tasks, task_pos):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            P.append(model(b).cpu().numpy())
            Y.append(b.y.cpu().numpy().reshape(-1, n_tasks))
    P, Y = np.vstack(P), np.vstack(Y)
    out, aurocs = {}, []
    for name, j in task_pos.items():
        yt, yp = Y[:, j], P[:, j]
        m = ~np.isnan(yt); yt, yp = yt[m], yp[m]
        if len(np.unique(yt)) < 2:
            out[name] = None; continue
        a = roc_auc_score(yt, yp); out[name] = round(float(a), 4); aurocs.append(a)
    macro = round(float(np.mean(aurocs)), 4) if aurocs else None
    return macro, out


def train(graphs, smiles, Ymat, tr_idx, va_idx, cfg, seed, n_tasks, val_pos):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GIN(graphs_dim, cfg["hidden_dim"], n_tasks, cfg["layers"], cfg["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.BCELoss()
    tl = make_loader(graphs, smiles, Ymat, tr_idx, n_tasks, True)
    vl = make_loader(graphs, smiles, Ymat, va_idx, n_tasks, False)
    best, best_state, noimp = -1, None, 0
    for _ in range(EPOCHS):
        model.train()
        for b in tl:
            b = b.to(DEVICE); opt.zero_grad()
            p = model(b); lab = b.y.reshape(-1, n_tasks)
            mask = ~torch.isnan(lab)
            if mask.sum() == 0:
                continue
            loss = crit(p[mask], lab[mask]); loss.backward(); opt.step()
        macro, _ = eval_loader(model, vl, n_tasks, val_pos)
        macro = macro if macro is not None else -1
        if macro > best:
            best, noimp = macro, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            noimp += 1
            if noimp >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best


def run_arm(graphs, smiles, Ymat, tr_idx, va_idx, te_idx, seed, n_tasks, pos):
    best = (-1, None)
    for cfg in CONFIGS:
        model, va = train(graphs, smiles, Ymat, tr_idx, va_idx, cfg, seed, n_tasks, pos)
        if va > best[0]:
            best = (va, model)
    tl = make_loader(graphs, smiles, Ymat, te_idx, n_tasks, False)
    return eval_loader(best[1], tl, n_tasks, pos)


graphs_dim = None

def main():
    global graphs_dim
    print(f"device={DEVICE}")
    comb, task_order = build_combined()
    print(f"combined compounds={len(comb)} tasks={len(task_order)}")
    smiles = comb["smiles"].tolist()
    print("featurizing graphs ...")
    gmap = {s: smiles_to_graph(s) for s in smiles}
    graphs_dim = next(g for g in gmap.values() if g is not None).x.shape[1]
    print("atom feature dim:", graphs_dim)
    Ycomb = comb[task_order].values.astype(np.float32)
    smi2row = {s: i for i, s in enumerate(smiles)}
    comb_ct_pos = {t: task_order.index(t) for t in CLINTOX_TASKS}

    # ClinTox-only graphs
    ct = pd.read_csv(os.path.join(PROC, "clintox_clean.csv"))
    ct_smiles = ct["smiles"].tolist()
    ct_gmap = {s: gmap.get(s) or smiles_to_graph(s) for s in ct_smiles}
    ct_Y = ct[CLINTOX_TASKS].values.astype(np.float32)
    ctonly_pos = {t: CLINTOX_TASKS.index(t) for t in CLINTOX_TASKS}

    records = []
    for seed in SEEDS:
        sp = pd.read_csv(os.path.join(SPLITS, f"clintox_scaffold_seed{seed}.csv"))
        tr_i = sp.loc[sp.split == "train", "index"].values
        va_i = sp.loc[sp.split == "val", "index"].values
        te_i = sp.loc[sp.split == "test", "index"].values
        hold = set(ct["smiles"].values[te_i]) | set(ct["smiles"].values[va_i])

        # Arm A: ClinTox-only GIN
        a_macro, a_per = run_arm(ct_gmap, ct_smiles, ct_Y, tr_i, va_i, te_i,
                                 seed, len(CLINTOX_TASKS), ctonly_pos)

        # Arm B: MTL GIN (41 tasks)
        tr_rows = [i for i, s in enumerate(smiles) if s not in hold]
        va_rows = [smi2row[s] for s in ct["smiles"].values[va_i]]
        te_rows = [smi2row[s] for s in ct["smiles"].values[te_i]]
        b_macro, b_per = run_arm(gmap, smiles, Ycomb, tr_rows, va_rows, te_rows,
                                 seed, len(task_order), comb_ct_pos)

        print(f"seed {seed}: A_CT_TOX={a_per['CT_TOX']} -> B_CT_TOX={b_per['CT_TOX']} "
              f"| A_macro={a_macro} B_macro={b_macro} (train_B n={len(tr_rows)})",
              flush=True)
        records.append(dict(seed=seed, A_macro=a_macro, A_CT_TOX=a_per["CT_TOX"],
                            A_FDA=a_per["FDA_APPROVED"], B_macro=b_macro,
                            B_CT_TOX=b_per["CT_TOX"], B_FDA=b_per["FDA_APPROVED"],
                            n_train_B=len(tr_rows)))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT, "mtl_pilot_gnn_per_seed.csv"), index=False)

    def ms(c):
        v = df[c].dropna().astype(float).values
        return (round(float(v.mean()), 4), round(float(v.std()), 4))
    summary = {
        "architecture": "GIN",
        "ClinTox_scaffold_macro_AUROC": {"baseline": ms("A_macro"), "MTL": ms("B_macro")},
        "CT_TOX_scaffold_AUROC": {"baseline": ms("A_CT_TOX"), "MTL": ms("B_CT_TOX")},
        "FDA_APPROVED_scaffold_AUROC": {"baseline": ms("A_FDA"), "MTL": ms("B_FDA")},
    }
    json.dump(summary, open(os.path.join(OUT, "mtl_pilot_gnn_summary.json"), "w"), indent=2)
    print("\n=== GNN MTL PILOT SUMMARY (mean, SD over 5 seeds) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
