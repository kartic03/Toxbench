import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150
})

COLORS = {
    'RF':      '#2196F3',
    'XGBoost': '#4CAF50',
    'MLP':     '#FF9800',
    'GNN':     '#9C27B0',
    'random':  '#2196F3',
    'scaffold':'#F44336',
    'raw':     '#F44336',
    'platt':   '#4CAF50',
    'isotonic':'#FF9800',
}

with open('results/random_forest_results.json') as f:
    rf_res = json.load(f)
with open('results/xgboost_results.json') as f:
    xgb_res = json.load(f)
with open('results/mlp_results.json') as f:
    mlp_res = json.load(f)
with open('results/gnn_results.json') as f:
    gnn_res = json.load(f)
with open('results/calibration_results.json') as f:
    cal_res = json.load(f)
with open('results/ad_results.json') as f:
    ad_res = json.load(f)
with open('results/uncertainty_results.json') as f:
    unc_res = json.load(f)

def get_auroc(results, dataset, split_type):
    seed_results = results[dataset][split_type]
    aurocs = [r['test_results']['MACRO_AVG']['auroc']
              for r in seed_results]
    return np.mean(aurocs), np.std(aurocs)

model_data = [
    ('RF',      rf_res,  COLORS['RF']),
    ('XGBoost', xgb_res, COLORS['XGBoost']),
    ('MLP',     mlp_res, COLORS['MLP']),
    ('GNN',     gnn_res, COLORS['GNN']),
]

# ════════════════════════════════════════════════════
# FIGURE 1: AUROC comparison all models all datasets
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'Figure 1: Model Performance — Random vs Scaffold Split',
    fontsize=14, fontweight='bold'
)

dataset_titles = {
    'tox21':   'Tox21 (12 tasks)',
    'clintox': 'ClinTox (2 tasks)',
    'sider':   'SIDER (27 tasks)'
}

for ax_idx, dataset in enumerate(['tox21', 'clintox', 'sider']):
    ax = axes[ax_idx]
    x  = np.arange(len(model_data))
    w  = 0.35

    for i, (name, res, color) in enumerate(model_data):
        r_mean, r_std = get_auroc(res, dataset, 'random')
        s_mean, s_std = get_auroc(res, dataset, 'scaffold')
        ax.bar(i - w/2, r_mean, w, yerr=r_std,
               color=color, alpha=0.9, capsize=4)
        ax.bar(i + w/2, s_mean, w, yerr=s_std,
               color=color, alpha=0.4, capsize=4,
               hatch='//')
        if abs(r_mean - s_mean) > 0.02:
            ax.annotate('',
                xy=(i + w/2, s_mean + 0.01),
                xytext=(i - w/2, r_mean - 0.01),
                arrowprops=dict(arrowstyle='->',
                                color='red', lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in model_data])
    ax.set_ylabel('AUROC (macro-average ± SD)')
    ax.set_title(dataset_titles[dataset])
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    if ax_idx == 0:
        solid = mpatches.Patch(color='gray', alpha=0.9,
                               label='Random split')
        hatch = mpatches.Patch(color='gray', alpha=0.4,
                               hatch='//',
                               label='Scaffold split')
        ax.legend(handles=[solid, hatch], loc='lower right')

plt.tight_layout()
plt.savefig('figures/fig1_auroc_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved")

# ════════════════════════════════════════════════════
# FIGURE 2: Calibration reliability curves
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    'Figure 2: Reliability Curves — Before and After Calibration\n'
    '(Perfect calibration = diagonal line)',
    fontsize=13, fontweight='bold'
)

method_styles = {
    'raw':      ('#F44336', '-',  'Raw RF'),
    'platt':    ('#4CAF50', '--', 'Platt Scaling'),
    'isotonic': ('#FF9800', ':',  'Isotonic Regression'),
}

for row, dataset in enumerate(['tox21', 'clintox']):
    for col, split_type in enumerate(['random', 'scaffold', 'sider']):
        ax = axes[row][col]
        if col == 2 and dataset == 'tox21':
            # Use sider random for this slot
            d, s = 'sider', 'random'
        elif col == 2 and dataset == 'clintox':
            d, s = 'sider', 'scaffold'
        else:
            d, s = dataset, split_type

        if d not in cal_res:
            ax.text(0.5, 0.5, 'No data',
                    ha='center', va='center')
            continue

        seed_results = cal_res[d][s]
        method_curves = {m: {'x': [], 'y': []}
                         for m in ['raw', 'platt', 'isotonic']}

        for seed_res in seed_results:
            for rel in seed_res.get('reliability_data', []):
                m = rel['method']
                if m in method_curves:
                    method_curves[m]['x'].extend(
                        rel['mean_pred'])
                    method_curves[m]['y'].extend(
                        rel['frac_pos'])

        ax.plot([0,1],[0,1],'k--',alpha=0.5,
                label='Perfect',lw=1.5)

        for method, (color, ls, label) in method_styles.items():
            xs = method_curves[method]['x']
            ys = method_curves[method]['y']
            if len(xs) < 5:
                continue
            bins  = np.linspace(0, 1, 11)
            x_avg, y_avg = [], []
            for i in range(len(bins)-1):
                mask = [(bins[i] <= x < bins[i+1]) for x in xs]
                if sum(mask) > 0:
                    x_avg.append(np.mean(
                        [xs[j] for j,m2 in enumerate(mask) if m2]))
                    y_avg.append(np.mean(
                        [ys[j] for j,m2 in enumerate(mask) if m2]))
            ax.plot(x_avg, y_avg, color=color,
                    linestyle=ls, linewidth=2,
                    marker='o', markersize=4, label=label)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction Positive')
        ax.set_title(
            f'{d.capitalize()} — '
            f'{"Random" if s=="random" else "Scaffold"} Split'
        )
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig2_calibration_curves.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved")

# ════════════════════════════════════════════════════
# FIGURE 3: Calibration metrics bar chart
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Figure 3: Calibration Metrics Across Datasets and Split Types',
    fontsize=13, fontweight='bold'
)

method_colors_bar = [COLORS['raw'],
                     COLORS['platt'],
                     COLORS['isotonic']]
methods = ['raw', 'platt', 'isotonic']

for ax_idx, metric in enumerate(['brier_mean', 'ece_mean']):
    ax     = axes[ax_idx]
    groups = []
    labels = []

    for dataset in ['tox21', 'clintox', 'sider']:
        if dataset not in cal_res:
            continue
        for split_type in ['random', 'scaffold']:
            seed_results = cal_res[dataset][split_type]
            group_vals, group_errs = [], []
            for method in methods:
                vals = [r[method][metric]
                        for r in seed_results]
                group_vals.append(np.mean(vals))
                group_errs.append(np.std(vals))
            groups.append((group_vals, group_errs))
            ds = ('T21' if dataset=='tox21'
                  else 'CT' if dataset=='clintox'
                  else 'SD')
            sp = 'Rnd' if split_type=='random' else 'Scf'
            labels.append(f'{ds}\n{sp}')

    x = np.arange(len(groups))
    w = 0.25

    for j, method in enumerate(methods):
        vals = [g[0][j] for g in groups]
        errs = [g[1][j] for g in groups]
        ax.bar(x + (j-1)*w, vals, w,
               yerr=errs, capsize=3,
               color=method_colors_bar[j],
               label=method.capitalize(),
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(
        'Brier Score' if 'brier' in metric else 'ECE'
    )
    ax.set_title(
        'Brier Score (lower = better)'
        if 'brier' in metric
        else 'Expected Calibration Error (lower = better)'
    )
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig3_calibration_metrics.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved")

# ════════════════════════════════════════════════════
# FIGURE 4: Applicability Domain
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Figure 4: Applicability Domain — '
    'Performance vs Similarity to Training Set',
    fontsize=13, fontweight='bold'
)

bin_labels  = ['0.0-0.2','0.2-0.4','0.4-0.6',
                '0.6-0.8','0.8-1.0']
bin_centers = [0.1, 0.3, 0.5, 0.7, 0.9]

for ax_idx, dataset in enumerate(['tox21', 'clintox']):
    ax = axes[ax_idx]

    for split_type in ['random', 'scaffold']:
        color = COLORS[split_type]
        res   = ad_res[dataset][split_type]
        bin_results = res['bin_results']

        bin_aurocs = {b: [] for b in bin_labels}
        for seed_bins in bin_results:
            for b in seed_bins:
                if b['auroc'] is not None:
                    bin_aurocs[b['bin']].append(b['auroc'])

        xs, ys, yerrs = [], [], []
        for bl, bc in zip(bin_labels, bin_centers):
            vals = bin_aurocs[bl]
            if len(vals) >= 2:
                xs.append(bc)
                ys.append(np.mean(vals))
                yerrs.append(np.std(vals))

        if xs:
            ax.errorbar(xs, ys, yerr=yerrs,
                        color=color, marker='o',
                        linewidth=2, markersize=7,
                        capsize=4,
                        label=split_type.capitalize()+' split')

    ax.axhline(0.5, color='gray', linestyle='--',
               alpha=0.5, label='Random chance')
    ax.set_xlabel(
        'Nearest-Neighbor Tanimoto Similarity\nto Training Set'
    )
    ax.set_ylabel('AUROC (macro-average ± SD)')
    ax.set_title(
        'Tox21' if dataset=='tox21' else 'ClinTox'
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1.0)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, rotation=30, ha='right')

plt.tight_layout()
plt.savefig('figures/fig4_applicability_domain.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved")

# ════════════════════════════════════════════════════
# FIGURE 5: Uncertainty trustworthiness curve
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Figure 5: Uncertainty — Trustworthiness Curves\n'
    '(Keeping only low-uncertainty predictions improves AUROC)',
    fontsize=13, fontweight='bold'
)

for ax_idx, dataset in enumerate(['tox21', 'clintox']):
    ax = axes[ax_idx]

    for split_type in ['random', 'scaffold']:
        color        = COLORS[split_type]
        seed_results = unc_res[dataset][split_type]

        coverage_aurocs = {}
        for seed_res in seed_results:
            for cr in seed_res.get('coverage_results', []):
                pct = cr['coverage_pct']
                if pct not in coverage_aurocs:
                    coverage_aurocs[pct] = []
                coverage_aurocs[pct].append(cr['auroc'])

        pcts  = sorted(coverage_aurocs.keys(), reverse=True)
        means = [np.mean(coverage_aurocs[p]) for p in pcts]
        stds  = [np.std(coverage_aurocs[p])  for p in pcts]

        if pcts:
            ax.errorbar(pcts, means, yerr=stds,
                        color=color, marker='o',
                        linewidth=2, markersize=7,
                        capsize=4,
                        label=split_type.capitalize()+' split')

    ax.set_xlabel('Coverage (% of test compounds kept)')
    ax.set_ylabel('AUROC (macro-average ± SD)')
    ax.set_title(
        'Tox21' if dataset=='tox21' else 'ClinTox'
    )
    ax.set_xlim(15, 110)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xticklabels(['25%\n(most confident)',
                        '50%', '75%', '100%\n(all)'])
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig5_uncertainty.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved")

# ════════════════════════════════════════════════════
# FIGURE 6: Heatmap all models all datasets
# ════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle(
    'Figure 6: AUROC Heatmap — All Models, Datasets, and Splits',
    fontsize=13, fontweight='bold'
)

for ax_idx, dataset in enumerate(['tox21', 'clintox', 'sider']):
    ax   = axes[ax_idx]
    rows = ['Random', 'Scaffold']
    cols = [m[0] for m in model_data]
    data = np.zeros((2, 4))
    annot = np.empty((2, 4), dtype=object)

    for col_i, (name, res, _) in enumerate(model_data):
        for row_i, split_type in enumerate(
                ['random', 'scaffold']):
            mean, std = get_auroc(res, dataset, split_type)
            data[row_i, col_i]  = mean
            annot[row_i, col_i] = f'{mean:.3f}\n±{std:.3f}'

    im = ax.imshow(data, cmap='RdYlGn',
                   vmin=0.4, vmax=0.9, aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(cols, fontsize=11)
    ax.set_yticks(range(2))
    ax.set_yticklabels(rows, fontsize=11)
    ax.set_title(dataset_titles[dataset])

    for r in range(2):
        for c in range(4):
            ax.text(c, r, annot[r, c],
                    ha='center', va='center',
                    fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax, label='AUROC')

plt.tight_layout()
plt.savefig('figures/fig6_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved")

print("\nALL FIGURES GENERATED SUCCESSFULLY")
print("Check the figures/ folder")
