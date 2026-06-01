"""
Cross-dataset multi-task learning pilot (revision item R1.5).

Question from Reviewer 1: does cross-dataset MTL / transfer learning improve
scaffold generalization and reduce instability on ClinTox?

Design (two arms, identical test sets, identical code path, same hyperparameter
grid, same 5 seeds, evaluated on the ClinTox *scaffold* test split):

  Arm A  (baseline)  : MLP trained on ClinTox only (2 tasks).
  Arm B  (MTL)       : MLP trained jointly on Tox21+SIDER+ClinTox (41 tasks),
                       evaluated on the same ClinTox tasks / test compounds.

Leakage control: the ClinTox scaffold *test* (and *val*) compounds for a given
seed are removed from ALL training data in both arms, matched by canonical
SMILES, so a ClinTox test molecule can never enter training via Tox21/SIDER.

We report ClinTox macro-AUROC (2 tasks) and CT_TOX AUROC (the toxicity label),
mean +/- SD over 5 seeds, for both arms.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")
SPLITS = os.path.join(ROOT, "splits")
OUT = os.path.join(ROOT, "supplementary_files")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1337]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BATCH, PATIENCE = 60, 64, 10
CONFIGS = [
    {"hidden_dims": [512, 256], "dropout": 0.2, "lr": 1e-3},
    {"hidden_dims": [512, 256], "dropout": 0.4, "lr": 1e-3},
    {"hidden_dims": [1024, 512, 256], "dropout": 0.2, "lr": 1e-3},
    {"hidden_dims": [1024, 512, 256], "dropout": 0.4, "lr": 3e-4},
]
CLINTOX_TASKS = ["FDA_APPROVED", "CT_TOX"]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def smiles_to_fp():
    """Map each canonical SMILES -> its precomputed ECFP4 (from any dataset)."""
    mapping = {}
    for ds in ["tox21", "sider", "clintox"]:
        df = pd.read_csv(os.path.join(PROC, f"{ds}_clean.csv"))
        fp = np.load(os.path.join(PROC, f"{ds}_ecfp4.npy")).astype(np.float32)
        for i, s in enumerate(df["smiles"].values):
            if s not in mapping:
                mapping[s] = fp[i]
    return mapping


def build_combined():
    """Union of the three datasets keyed by canonical SMILES, 41-task labels."""
    frames = {ds: pd.read_csv(os.path.join(PROC, f"{ds}_clean.csv"))
              for ds in ["tox21", "sider", "clintox"]}
    task_order = []
    for ds in ["tox21", "sider", "clintox"]:
        task_order += [c for c in frames[ds].columns if c != "smiles"]
    # union of smiles
    all_smiles = sorted(set().union(*[set(f["smiles"]) for f in frames.values()]))
    comb = pd.DataFrame({"smiles": all_smiles})
    comb = comb.set_index("smiles")
    for t in task_order:
        comb[t] = np.nan
    for ds, f in frames.items():
        f2 = f.set_index("smiles")
        for t in [c for c in f.columns if c != "smiles"]:
            comb.loc[f2.index, t] = f2[t].values
    comb = comb.reset_index()
    return comb, task_order


def eval_tasks(y_true, y_prob, task_idx):
    aurocs, per = [], {}
    for name, j in task_idx.items():
        yt, yp = y_true[:, j], y_prob[:, j]
        mask = ~np.isnan(yt)
        yt, yp = yt[mask], yp[mask]
        if len(np.unique(yt)) < 2:
            per[name] = None
            continue
        a = roc_auc_score(yt, yp)
        per[name] = round(float(a), 4)
        aurocs.append(a)
    macro = round(float(np.mean(aurocs)), 4) if aurocs else None
    return macro, per


def train(X_tr, Y_tr, X_va, Y_va, cfg, seed, n_out, val_task_idx):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(X_tr.shape[1], cfg["hidden_dims"], n_out, cfg["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.BCELoss()
    Xt = torch.FloatTensor(X_tr).to(DEVICE)
    Yt = torch.FloatTensor(np.nan_to_num(Y_tr, nan=0.0)).to(DEVICE)
    Mt = torch.FloatTensor((~np.isnan(Y_tr)).astype(np.float32)).to(DEVICE)
    loader = DataLoader(TensorDataset(Xt, Yt, Mt), batch_size=BATCH, shuffle=True)
    Xv = torch.FloatTensor(X_va).to(DEVICE)
    best_auroc, best_state, noimp = -1, None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb, mb in loader:
            opt.zero_grad()
            p = model(xb)
            mask = mb.bool()
            if mask.sum() == 0:
                continue
            loss = crit(p[mask], yb[mask])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv).cpu().numpy()
        macro, _ = eval_tasks(Y_va, vp, val_task_idx)
        macro = macro if macro is not None else -1
        if macro > best_auroc:
            best_auroc, noimp = macro, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            noimp += 1
            if noimp >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_auroc


def run_arm(name, X_tr, Y_tr, X_va, Y_va, X_te, Y_te, seed, n_out,
            val_task_idx, test_task_idx):
    best = (-1, None, None)
    for cfg in CONFIGS:
        model, va = train(X_tr, Y_tr, X_va, Y_va, cfg, seed, n_out, val_task_idx)
        if va > best[0]:
            best = (va, model, cfg)
    model = best[1]
    model.eval()
    with torch.no_grad():
        tp = model(torch.FloatTensor(X_te).to(DEVICE)).cpu().numpy()
    macro, per = eval_tasks(Y_te, tp, test_task_idx)
    return macro, per, best[2]


def main():
    print(f"device={DEVICE}")
    comb, task_order = build_combined()
    print(f"combined compounds={len(comb)}  tasks={len(task_order)}")
    fpmap = smiles_to_fp()
    Xcomb = np.vstack([fpmap[s] for s in comb["smiles"]]).astype(np.float32)
    Ycomb = comb[task_order].values.astype(np.float32)
    smi2row = {s: i for i, s in enumerate(comb["smiles"])}

    # ClinTox-only arrays
    ct = pd.read_csv(os.path.join(PROC, "clintox_clean.csv"))
    ct_fp = np.load(os.path.join(PROC, "clintox_ecfp4.npy")).astype(np.float32)
    ct_Y = ct[CLINTOX_TASKS].values.astype(np.float32)

    comb_ct_idx = {t: task_order.index(t) for t in CLINTOX_TASKS}
    ctonly_idx = {t: CLINTOX_TASKS.index(t) for t in CLINTOX_TASKS}

    records = []
    for seed in SEEDS:
        sp = pd.read_csv(os.path.join(SPLITS, f"clintox_scaffold_seed{seed}.csv"))
        tr_i = sp.loc[sp.split == "train", "index"].values
        va_i = sp.loc[sp.split == "val", "index"].values
        te_i = sp.loc[sp.split == "test", "index"].values
        test_smiles = set(ct["smiles"].values[te_i])
        val_smiles = set(ct["smiles"].values[va_i])
        hold = test_smiles | val_smiles

        # ---- Arm A: ClinTox-only ----
        a_macro, a_per, a_cfg = run_arm(
            "A", ct_fp[tr_i], ct_Y[tr_i], ct_fp[va_i], ct_Y[va_i],
            ct_fp[te_i], ct_Y[te_i], seed, len(CLINTOX_TASKS),
            ctonly_idx, ctonly_idx)

        # ---- Arm B: MTL on 41 tasks ----
        train_mask = np.array([s not in hold for s in comb["smiles"]])
        Xtr_b, Ytr_b = Xcomb[train_mask], Ycomb[train_mask]
        val_rows = [smi2row[s] for s in ct["smiles"].values[va_i]]
        te_rows = [smi2row[s] for s in ct["smiles"].values[te_i]]
        b_macro, b_per, b_cfg = run_arm(
            "B", Xtr_b, Ytr_b, Xcomb[val_rows], Ycomb[val_rows],
            Xcomb[te_rows], Ycomb[te_rows], seed, len(task_order),
            comb_ct_idx, comb_ct_idx)

        print(f"seed {seed}: A_macro={a_macro} CT_TOX_A={a_per['CT_TOX']} | "
              f"B_macro={b_macro} CT_TOX_B={b_per['CT_TOX']} "
              f"(train_B n={train_mask.sum()})")
        records.append(dict(seed=seed,
                            A_macro=a_macro, A_CT_TOX=a_per["CT_TOX"],
                            A_FDA=a_per["FDA_APPROVED"],
                            B_macro=b_macro, B_CT_TOX=b_per["CT_TOX"],
                            B_FDA=b_per["FDA_APPROVED"],
                            n_train_B=int(train_mask.sum())))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT, "mtl_pilot_per_seed.csv"), index=False)

    def ms(col):
        v = df[col].dropna().astype(float).values
        return (round(float(v.mean()), 4), round(float(v.std()), 4))

    summary = {
        "ClinTox_scaffold_macro_AUROC": {
            "baseline_ClinTox_only": ms("A_macro"),
            "MTL_3dataset": ms("B_macro")},
        "CT_TOX_scaffold_AUROC": {
            "baseline_ClinTox_only": ms("A_CT_TOX"),
            "MTL_3dataset": ms("B_CT_TOX")},
        "FDA_APPROVED_scaffold_AUROC": {
            "baseline_ClinTox_only": ms("A_FDA"),
            "MTL_3dataset": ms("B_FDA")},
    }
    json.dump(summary, open(os.path.join(OUT, "mtl_pilot_summary.json"), "w"),
              indent=2)
    print("\n=== MTL PILOT SUMMARY (mean, SD over 5 seeds) ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved -> supplementary_files/mtl_pilot_summary.json")


if __name__ == "__main__":
    main()
