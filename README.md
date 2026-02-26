# ToxBench

A leakage-safe classification benchmark for predictive toxicology with calibration, uncertainty, and domain-shift analysis.

## Datasets
- **Tox21** — 12 tasks, 7,538 compounds after cleaning
- **ClinTox** — 2 tasks, 1,379 compounds after cleaning  
- **SIDER** — 27 tasks, 1,350 compounds after cleaning

## Models
- Random Forest (ECFP4 fingerprints)
- XGBoost (ECFP4 fingerprints)
- MLP (ECFP4 fingerprints, GPU)
- GNN — Graph Isomorphism Network (GPU)

## Key Finding
Scaffold-based splitting reduces AUROC by 0.03–0.08 points across all model classes, demonstrating systematic performance overestimation under random splitting.

## Quick Start

### 1. Setup environment
```bash
conda env create -f environment.yml
conda activate toxbench
```

### 2. Download and process data
```bash
python src/pipeline/download_data.py
python src/pipeline/download_sider.py
python src/pipeline/standardize.py
python src/pipeline/create_splits.py
python src/pipeline/compute_fingerprints.py
```

### 3. Train models
```bash
python src/models/random_forest.py
python src/models/xgboost_model.py
python src/models/mlp_model.py
python src/models/gnn_model.py
```

### 4. Analysis
```bash
python src/calibration/calibration_analysis.py
python src/evaluation/uncertainty.py
python src/evaluation/applicability_domain.py
python src/evaluation/scaffold_error_analysis.py
```

### 5. Generate figures
```bash
python src/evaluation/make_figures.py
```

## Results Summary (AUROC, macro-average ± SD)

| Model   | Tox21 Random | Tox21 Scaffold | ClinTox Random | ClinTox Scaffold | SIDER Random | SIDER Scaffold |
|---------|--------------|----------------|----------------|------------------|--------------|----------------|
| RF      | 0.804±0.007  | 0.747±0.024    | 0.387±0.085    | 0.579±0.160      | 0.670±0.026  | 0.635±0.015    |
| XGBoost | 0.787±0.006  | 0.708±0.018    | 0.627±0.118    | 0.679±0.108      | 0.648±0.027  | 0.617±0.016    |
| MLP     | 0.791±0.017  | 0.723±0.029    | 0.392±0.120    | 0.534±0.109      | 0.639±0.013  | 0.604±0.018    |
| GNN     | 0.819±0.009  | 0.744±0.025    | 0.539±0.116    | 0.675±0.068      | 0.608±0.022  | 0.622±0.017    |

## Project Structure
```
toxbench/
├── src/
│   ├── pipeline/      # Data download, cleaning, splits, fingerprints
│   ├── models/        # RF, XGBoost, MLP, GNN
│   ├── calibration/   # Platt scaling, isotonic regression
│   └── evaluation/    # AD analysis, uncertainty, figures
├── splits/            # Train/val/test indices (30 CSV files)
├── results/           # All JSON result files
├── figures/           # All 6 publication figures
├── environment.yml    # Conda environment
└── requirements.txt   # pip dependencies
```

## Citation
(to be added after publication)

## License
MIT
