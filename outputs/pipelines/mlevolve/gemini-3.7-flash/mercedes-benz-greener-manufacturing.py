import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, FactorAnalysis
from sklearn.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Tuple

# =========================================================================
# 0. Global Setup & Seed Configuration
# =========================================================================
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Evaluation Metric Definitions
# =========================================================================
def calculate_r2_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the exact official evaluation metric (R2 score)."""
    return float(r2_score(y_true, y_pred))


# =========================================================================
# 2. PyTorch Tabular Deep Learning Architecture
# =========================================================================
class TabularResidualBlock(nn.Module):
    """Residual Dense Block with LayerNorm, GELU, and Dropout for Tabular Data."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.norm1(self.linear1(x)))
        out = self.dropout(out)
        out = self.norm2(self.linear2(out))
        out = self.act(out + residual)
        return out


class TabularResNet(nn.Module):
    """Deep Tabular Residual Network for Continuous Target Regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
        )

        blocks = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            blocks.append(
                TabularResidualBlock(
                    in_features=in_dim,
                    out_features=out_dim,
                    dropout_rate=dropout_rate,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        out = self.head(x)
        return out.squeeze(-1)


# =========================================================================
# 3. Model Factories and Hyperparameter Configurations
# =========================================================================
def get_lgb_model_params(seed: int = 42) -> Dict[str, Any]:
    """Returns regularized LightGBM hyperparameters."""
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.015,
        "max_depth": 4,
        "num_leaves": 12,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.55,
        "reg_alpha": 1.5,
        "reg_lambda": 3.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_model_params(seed: int = 42) -> Dict[str, Any]:
    """Returns regularized XGBoost hyperparameters."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.015,
        "max_depth": 3,
        "min_child_weight": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "reg_alpha": 2.0,
        "reg_lambda": 4.0,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_linear_models(seed: int = 42) -> Dict[str, Any]:
    """Returns regularized linear and kernel regression models."""
    return {
        "ridge": Ridge(alpha=100.0, random_state=seed),
        "elasticnet": ElasticNet(
            alpha=0.1, l1_ratio=0.3, random_state=seed, max_iter=2000
        ),
        "huber": HuberRegressor(alpha=100.0, max_iter=2000),
        "svr_rbf": SVR(C=2.0, epsilon=0.1, kernel="rbf"),
    }


def build_neural_network_setup(
    input_dim: int,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    device: Any = device,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    """Builds the TabularResNet model, Huber loss, optimizer, and scheduler."""
    model = TabularResNet(
        input_dim=input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.25
    )
    model.to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    return model, criterion, optimizer, scheduler


# =========================================================================
# 4. Data Loading and Feature Engineering
# =========================================================================
# Load raw datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

id_col_train = "ID" if "ID" in train_df.columns else "id"
id_col_test = "ID" if "ID" in test_df.columns else "id"

train_ids = train_df[id_col_train].values
test_ids = test_df[id_col_test].values
y_train = train_df["y"].values

# Identify column types
cat_cols = [
    c
    for c in train_df.columns
    if train_df[c].dtype == "object" and c not in [id_col_train, "y"]
]
num_cols = [
    c for c in train_df.columns if c not in cat_cols and c not in [id_col_train, "y"]
]

# Filter invariant and duplicate binary columns
constant_cols = [c for c in num_cols if train_df[c].nunique() <= 1]
valid_num_cols = [c for c in num_cols if c not in constant_cols]

dup_cols = []
seen_hashes = {}
for col in valid_num_cols:
    col_hash = hash(tuple(train_df[col].values))
    if col_hash in seen_hashes:
        dup_cols.append(col)
    else:
        seen_hashes[col_hash] = col

clean_num_cols = [c for c in valid_num_cols if c not in dup_cols]

# Extract binary matrices
X_bin_train = train_df[clean_num_cols].values.astype(np.float32)
X_bin_test = test_df[clean_num_cols].values.astype(np.float32)


# Row-wise summary statistics
def compute_row_stats(arr: np.ndarray) -> np.ndarray:
    row_sum = np.sum(arr, axis=1, keepdims=True)
    row_mean = np.mean(arr, axis=1, keepdims=True)
    row_std = np.std(arr, axis=1, keepdims=True)
    row_nonzero = np.count_nonzero(arr, axis=1).reshape(-1, 1)
    return np.hstack([row_sum, row_mean, row_std, row_nonzero])


train_row_stats = compute_row_stats(X_bin_train)
test_row_stats = compute_row_stats(X_bin_test)
row_stat_names = ["bin_row_sum", "bin_row_mean", "bin_row_std", "bin_row_nonzero_count"]

# Multi-decomposition latent representations
n_comp = 10
decompositions = {
    "pca": PCA(n_components=n_comp, random_state=42),
    "ica": FastICA(n_components=n_comp, random_state=42, max_iter=1000, tol=1e-3),
    "tsvd": TruncatedSVD(n_components=n_comp, algorithm="randomized", random_state=42),
    "grp": GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=42),
    "srp": SparseRandomProjection(
        n_components=n_comp, dense_output=True, random_state=42
    ),
    "fa": FactorAnalysis(n_components=n_comp, random_state=42),
}

train_decomp_feats = []
test_decomp_feats = []
decomp_col_names = []

for name, decomp_model in decompositions.items():
    tr_comp = decomp_model.fit_transform(X_bin_train)
    te_comp = decomp_model.transform(X_bin_test)
    train_decomp_feats.append(tr_comp)
    test_decomp_feats.append(te_comp)
    decomp_col_names.extend([f"{name}_{i}" for i in range(n_comp)])

train_decomp_arr = np.hstack(train_decomp_feats)
test_decomp_arr = np.hstack(test_decomp_feats)

# Categorical Feature Engineering & Interactions
train_cat_df = pd.DataFrame(index=train_df.index)
test_cat_df = pd.DataFrame(index=test_df.index)

interaction_pairs = [
    ("X0", "X1"),
    ("X0", "X2"),
    ("X0", "X5"),
    ("X1", "X2"),
    ("X3", "X6"),
]
for col1, col2 in interaction_pairs:
    if col1 in cat_cols and col2 in cat_cols:
        inter_name = f"{col1}_{col2}"
        train_cat_df[inter_name] = (
            train_df[col1].astype(str) + "_" + train_df[col2].astype(str)
        )
        test_cat_df[inter_name] = (
            test_df[col1].astype(str) + "_" + test_df[col2].astype(str)
        )

all_cat_cols = cat_cols + list(train_cat_df.columns)

for c in all_cat_cols:
    src_train = train_df[c] if c in cat_cols else train_cat_df[c]
    src_test = test_df[c] if c in cat_cols else test_cat_df[c]

    freq_map = src_train.value_counts(normalize=True).to_dict()
    train_cat_df[f"{c}_freq"] = src_train.map(freq_map).fillna(0.0).values
    test_cat_df[f"{c}_freq"] = src_test.map(freq_map).fillna(0.0).values

    le = LabelEncoder()
    unique_vals = list(
        set(src_train.dropna().unique()).union(set(src_test.dropna().unique()))
    )
    le.fit(unique_vals)

    train_cat_df[f"{c}_le"] = le.transform(src_train)
    test_cat_df[f"{c}_le"] = le.transform(src_test)

# Out-Of-Fold Target Encoding (Mean and Standard Deviation)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
target_enc_cols = ["X0", "X1", "X2", "X3", "X5", "X6", "X8"]

oof_te_train = pd.DataFrame(index=train_df.index)
oof_te_test = pd.DataFrame(index=test_df.index)

global_mean = float(np.mean(y_train))
global_std = float(np.std(y_train))
smoothing_weight = 10.0

for c in target_enc_cols:
    if c in cat_cols:
        col_te_mean_train = np.zeros(len(train_df), dtype=np.float32)
        col_te_mean_test_folds = np.zeros(len(test_df), dtype=np.float32)
        col_te_std_train = np.zeros(len(train_df), dtype=np.float32)
        col_te_std_test_folds = np.zeros(len(test_df), dtype=np.float32)

        for tr_idx, val_idx in kf.split(train_df):
            tr_subset = train_df.iloc[tr_idx]
            val_subset = train_df.iloc[val_idx]

            stats_mean = tr_subset.groupby(c)["y"].agg(["count", "mean"])
            smooth_mean_map = (
                (stats_mean["count"] * stats_mean["mean"] + smoothing_weight * global_mean)
                / (stats_mean["count"] + smoothing_weight)
            ).to_dict()

            col_te_mean_train[val_idx] = (
                val_subset[c].map(smooth_mean_map).fillna(global_mean).values
            )
            col_te_mean_test_folds += (
                test_df[c].map(smooth_mean_map).fillna(global_mean).values / 10.0
            )

            stats_std = tr_subset.groupby(c)["y"].agg(["count", "std"])
            smooth_std_map = (
                (stats_std["count"] * stats_std["std"].fillna(global_std) + smoothing_weight * global_std)
                / (stats_std["count"] + smoothing_weight)
            ).to_dict()

            col_te_std_train[val_idx] = (
                val_subset[c].map(smooth_std_map).fillna(global_std).values
            )
            col_te_std_test_folds += (
                test_df[c].map(smooth_std_map).fillna(global_std).values / 10.0
            )

        oof_te_train[f"{c}_te_mean"] = col_te_mean_train
        oof_te_test[f"{c}_te_mean"] = col_te_mean_test_folds
        oof_te_train[f"{c}_te_std"] = col_te_std_train
        oof_te_test[f"{c}_te_std"] = col_te_std_test_folds

# Assemble Final Feature Matrices
train_features = pd.concat(
    [
        pd.DataFrame({id_col_train: train_ids, "y": y_train}),
        pd.DataFrame(X_bin_train, columns=clean_num_cols),
        pd.DataFrame(train_row_stats, columns=row_stat_names),
        pd.DataFrame(train_decomp_arr, columns=decomp_col_names),
        train_cat_df.reset_index(drop=True),
        oof_te_train.reset_index(drop=True),
    ],
    axis=1,
)

test_features = pd.concat(
    [
        pd.DataFrame({id_col_test: test_ids}),
        pd.DataFrame(X_bin_test, columns=clean_num_cols),
        pd.DataFrame(test_row_stats, columns=row_stat_names),
        pd.DataFrame(test_decomp_arr, columns=decomp_col_names),
        test_cat_df.reset_index(drop=True),
        oof_te_test.reset_index(drop=True),
    ],
    axis=1,
)

train_features["fold"] = -1
for fold, (tr_idx, val_idx) in enumerate(kf.split(train_features)):
    train_features.loc[val_idx, "fold"] = fold

# =========================================================================
# 5. Model Training & Cross-Validation
# =========================================================================
feature_cols = [
    c
    for c in train_features.columns
    if c not in [id_col_train, "y", "fold"]
    and train_features[c].dtype
    in [np.float32, np.float64, np.int32, np.int64, float, int]
]

X_train_all = train_features[feature_cols].values.astype(np.float32)
y_train_all = train_features["y"].values.astype(np.float32)
folds = train_features["fold"].values
X_test_all = test_features[feature_cols].values.astype(np.float32)

X_train_all = np.nan_to_num(X_train_all, nan=0.0, posinf=0.0, neginf=0.0)
X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)

n_samples = len(X_train_all)
n_test = len(X_test_all)
n_folds = int(np.max(folds) + 1)

model_names = [
    "lgb",
    "xgb",
    "cat",
    "extratrees",
    "ridge",
    "elasticnet",
    "huber",
    "svr_rbf",
    "tabular_resnet",
]
oof_predictions = {name: np.zeros(n_samples, dtype=np.float32) for name in model_names}
test_predictions = {name: np.zeros(n_test, dtype=np.float32) for name in model_names}

for fold in range(n_folds):
    tr_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]

    X_tr, y_tr = X_train_all[tr_idx], y_train_all[tr_idx]
    X_val, y_val = X_train_all[val_idx], y_train_all[val_idx]
    X_te = X_test_all.copy()

    # Feature scaling strictly fit on fold training set
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_te_scaled = scaler.transform(X_te)

    # 1. LightGBM
    lgb_params = get_lgb_model_params(seed=42 + fold)
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1200,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=lgb_callbacks,
    )
    oof_predictions["lgb"][val_idx] = lgb_model.predict(
        X_val, num_iteration=lgb_model.best_iteration
    )
    test_predictions["lgb"] += (
        lgb_model.predict(X_te, num_iteration=lgb_model.best_iteration) / n_folds
    )

    # 2. XGBoost
    xgb_params = get_xgb_model_params(seed=42 + fold)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_te)
    evals = [(dtrain, "train"), (dval, "val")]
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1200,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    oof_predictions["xgb"][val_idx] = xgb_model.predict(
        dval, iteration_range=(0, xgb_model.best_iteration + 1)
    )
    test_predictions["xgb"] += (
        xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1))
        / n_folds
    )

    # 3. CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=5.0,
        rsm=0.6,
        random_seed=42 + fold,
        verbose=0,
    )
    cat_model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_predictions["cat"][val_idx] = cat_model.predict(X_val)
    test_predictions["cat"] += cat_model.predict(X_te) / n_folds

    # 4. ExtraTrees
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=6,
        max_features=0.6,
        min_samples_leaf=3,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    oof_predictions["extratrees"][val_idx] = et_model.predict(X_val)
    test_predictions["extratrees"] += et_model.predict(X_te) / n_folds

    # 5. Linear & Kernel Models
    linear_suite = get_linear_models(seed=42 + fold)
    for model_key in ["ridge", "elasticnet", "huber", "svr_rbf"]:
        clf = linear_suite[model_key]
        clf.fit(X_tr_scaled, y_tr)
        oof_predictions[model_key][val_idx] = clf.predict(X_val_scaled)
        test_predictions[model_key] += clf.predict(X_te_scaled) / n_folds

    # 6. TabularResNet (PyTorch)
    nn_model, criterion, optimizer, scheduler = build_neural_network_setup(
        input_dim=X_tr_scaled.shape[1],
        lr=1.5e-3,
        weight_decay=1e-4,
        epochs=60,
        device=device,
    )

    tr_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    tr_loader = DataLoader(tr_dataset, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=False)

    best_val_loss = float("inf")
    best_nn_state = None

    for epoch in range(60):
        nn_model.train()
        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = nn_model(batch_x)
                val_loss += criterion(preds, batch_y).item() * len(batch_y)
        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_state = {k: v.cpu() for k, v in nn_model.state_dict().items()}

    nn_model.load_state_dict(best_nn_state)
    nn_model.to(device)
    nn_model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        oof_predictions["tabular_resnet"][val_idx] = nn_model(val_tensor).cpu().numpy()

        test_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        test_predictions["tabular_resnet"] += (
            nn_model(test_tensor).cpu().numpy() / n_folds
        )

# =========================================================================
# 6. Out-Of-Fold Ensembling & Final Scoring
# =========================================================================
OOF_matrix = np.column_stack([oof_predictions[name] for name in model_names])
TEST_matrix = np.column_stack([test_predictions[name] for name in model_names])

# Constrained convex optimization ensemble (SLSQP: non-negative weights summing to 1.0)
n_models = len(model_names)
init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def loss_func(w):
    pred = OOF_matrix @ w
    return np.mean((y_train_all - pred) ** 2)


opt_res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 1000, "ftol": 1e-9},
)

optimal_weights = opt_res.x
optimal_weights = np.clip(optimal_weights, 0, 1)
optimal_weights = optimal_weights / np.sum(optimal_weights)

for name, weight in zip(model_names, optimal_weights):
    print(f"Model: {name:15s} | Weight: {weight:.4f}")

final_oof_preds = OOF_matrix @ optimal_weights
final_test_preds = TEST_matrix @ optimal_weights

final_r2_score = calculate_r2_metric(y_train_all, final_oof_preds)

# =========================================================================
# 7. Generate Submission
# =========================================================================
submission_df = pd.DataFrame(
    {id_col_test: test_ids, "y": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_r2_score}")
