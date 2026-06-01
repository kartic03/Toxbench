## Table S6: Model Hyperparameters

### Hyperparameter Grids

*Best configuration selected per seed by validation AUROC.*

| Model         | Config | n_estimators | max_depth        | class_weight | learning_rate | hidden_dims      | dropout | hidden_dim | num_layers |
| ------------- | ------ | ------------ | ---------------- | ------------ | ------------- | ---------------- | ------- | ---------- | ---------- |
| Random Forest | 1      | 100.0        | None (unlimited) | balanced     | —             | —                | —       | —          | —          |
| Random Forest | 2      | 500.0        | None (unlimited) | balanced     | —             | —                | —       | —          | —          |
| Random Forest | 3      | 100.0        | 20               | balanced     | —             | —                | —       | —          | —          |
| Random Forest | 4      | 500.0        | 20               | balanced     | —             | —                | —       | —          | —          |
| XGBoost       | 1      | 200.0        | 3                | —            | 0.05          | —                | —       | —          | —          |
| XGBoost       | 2      | 200.0        | 6                | —            | 0.1           | —                | —       | —          | —          |
| XGBoost       | 3      | 500.0        | 3                | —            | 0.05          | —                | —       | —          | —          |
| XGBoost       | 4      | 500.0        | 6                | —            | 0.1           | —                | —       | —          | —          |
| MLP           | 1      | —            | —                | —            | 0.001         | [512, 256]       | 0.2     | —          | —          |
| MLP           | 2      | —            | —                | —            | 0.001         | [512, 256]       | 0.4     | —          | —          |
| MLP           | 3      | —            | —                | —            | 0.001         | [1024, 512, 256] | 0.2     | —          | —          |
| MLP           | 4      | —            | —                | —            | 0.0003        | [1024, 512, 256] | 0.4     | —          | —          |
| GIN (GNN)     | 1      | —            | —                | —            | 0.001         | —                | 0.1     | 128.0      | 3.0        |
| GIN (GNN)     | 2      | —            | —                | —            | 0.001         | —                | 0.1     | 256.0      | 3.0        |
| GIN (GNN)     | 3      | —            | —                | —            | 0.0003        | —                | 0.3     | 128.0      | 3.0        |
| GIN (GNN)     | 4      | —            | —                | —            | 0.0003        | —                | 0.3     | 256.0      | 3.0        |

### Fixed Settings per Model

**Random Forest** (`src/models/random_forest.py`)

Input: ECFP4 fingerprints (2048 bits)

- max_features: sqrt (sklearn default)
- bootstrap: True (sklearn default)
- min_samples_split: 2 (sklearn default)
- min_samples_leaf: 1 (sklearn default)
- n_jobs: -1 (all cores)
- NaN handling: nan_to_num fill with 0.0
- Multi-task strategy: Single RF, multi-output

**XGBoost** (`src/models/xgboost_model.py`)

Input: ECFP4 fingerprints (2048 bits)

- objective: binary:logistic
- eval_metric: logloss
- scale_pos_weight: neg/pos ratio per task
- subsample: 1.0 (default)
- colsample_bytree: 1.0 (default)
- n_jobs: -1 (all cores)
- Multi-task strategy: One model per task

**MLP** (`src/models/mlp_model.py`)

Input: ECFP4 fingerprints (2048 bits)

- activation: ReLU
- output_activation: Sigmoid
- optimizer: Adam
- loss: BCELoss (masked NaN)
- batch_size: 64
- max_epochs: 50
- early_stopping: patience=10 (val AUROC)
- device: CUDA if available, else CPU
- Multi-task strategy: Single model, all tasks jointly

**GIN (GNN)** (`src/models/gnn_model.py`)

Input: Molecular graph (atom features: atomic num, degree, formal charge, hybridization, aromaticity, ring membership)

- GNN architecture: Graph Isomorphism Network (GIN)
- Pooling: Global add pooling
- Batch normalisation: Per-layer BatchNorm1d
- MLP head: hidden_dim → hidden_dim/2 → n_tasks
- output_activation: Sigmoid
- optimizer: Adam
- loss: BCELoss (masked NaN)
- batch_size: 64
- max_epochs: 100
- early_stopping: patience=15 (val AUROC)
- device: CUDA if available, else CPU
- Multi-task strategy: Single model, all tasks jointly

### Shared Settings (All Models)

- **Seeds**: 5 (42, 123, 456, 789, 1337)
- **Evaluation**: Macro-averaged AUROC across tasks
- **NaN labels**: Masked during loss / evaluation
- **Val split use**: Config selection only; test set held out
- **Fingerprints**: RDKit ECFP4, radius=2, 2048 bits
- **Scaffold split**: Bemis-Murcko (RDKit), deterministic per seed
- **Random split**: Stratified by first task, fixed per seed

