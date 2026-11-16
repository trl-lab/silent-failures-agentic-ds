import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    Ridge,
    RidgeCV,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Directories & Data Loading
# ---------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_path = Path("./input/train.parquet")
test_path = Path("./input/test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Standardize column naming
train_df.columns = [col.strip() for col in train_df.columns]
test_df.columns = [col.strip() for col in test_df.columns]

if "Id" in train_df.columns:
    train_df.rename(columns={"Id": "id"}, inplace=True)
elif "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df))

if "Id" in test_df.columns:
    test_df.rename(columns={"Id": "id"}, inplace=True)
elif "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))

# ---------------------------------------------------------
# 2. Date & Temporal Feature Engineering
# ---------------------------------------------------------
ref_date = pd.to_datetime("2015-05-01")


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = pd.to_datetime(df["Open Date"], format="mixed")

    df["open_year"] = date_col.dt.year
    df["open_month"] = date_col.dt.month
    df["open_day"] = date_col.dt.day
    df["open_dayofweek"] = date_col.dt.dayofweek
    df["open_quarter"] = date_col.dt.quarter
    df["open_is_weekend"] = (date_col.dt.dayofweek >= 5).astype(int)

    # Restaurant age / tenure in days, months, and years
    df["days_open"] = (ref_date - date_col).dt.days
    df["months_open"] = df["days_open"] / 30.4375
    df["years_open"] = df["days_open"] / 365.25

    # Non-linear tenure representations & saturation curves
    df["log_days_open"] = np.log1p(np.maximum(0, df["days_open"]))
    df["sqrt_days_open"] = np.sqrt(np.maximum(0, df["days_open"]))
    df["exp_sat_age"] = 1.0 - np.exp(-df["days_open"] / (365.25 * 5.0))
    df["years_open_sq"] = df["years_open"] ** 2

    # Cyclical encoding of launch month
    df["month_sin"] = np.sin(2 * np.pi * df["open_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["open_month"] / 12.0)

    return df


train_df = extract_date_features(train_df)
test_df = extract_date_features(test_df)

# ---------------------------------------------------------
# 3. Categorical Encodings (Leak-Free)
# ---------------------------------------------------------
city_group_map = {"Big Cities": 1, "Other": 0}
train_df["city_group_enc"] = train_df["City Group"].map(city_group_map).fillna(0)
test_df["city_group_enc"] = test_df["City Group"].map(city_group_map).fillna(0)

type_cols = ["IL", "FC", "DT", "MB"]
for t in type_cols:
    train_df[f"type_{t}"] = (train_df["Type"] == t).astype(int)
    test_df[f"type_{t}"] = (test_df["Type"] == t).astype(int)

city_freq = train_df["City"].value_counts(normalize=True).to_dict()
train_df["city_freq"] = train_df["City"].map(city_freq).fillna(0.0)
test_df["city_freq"] = test_df["City"].map(city_freq).fillna(0.0)

big_cities = {"İstanbul", "Ankara", "İzmir"}
train_df["is_major_metro"] = train_df["City"].isin(big_cities).astype(int)
test_df["is_major_metro"] = test_df["City"].isin(big_cities).astype(int)

# Domain tenure interactions
train_df["city_group_days_open"] = train_df["city_group_enc"] * train_df["days_open"]
test_df["city_group_days_open"] = test_df["city_group_enc"] * test_df["days_open"]
train_df["city_group_log_days"] = train_df["city_group_enc"] * train_df["log_days_open"]
test_df["city_group_log_days"] = test_df["city_group_enc"] * test_df["log_days_open"]
train_df["is_major_metro_log_days"] = train_df["is_major_metro"] * train_df["log_days_open"]
test_df["is_major_metro_log_days"] = test_df["is_major_metro"] * test_df["log_days_open"]

# ---------------------------------------------------------
# 4. P-Variable Subgroup Aggregations & Interactions
# ---------------------------------------------------------
p_cols = [f"P{i}" for i in range(1, 38)]
p_demo_cols = [f"P{i}" for i in range(1, 9)]
p_re_cols = [f"P{i}" for i in range(9, 30)]
p_comm_cols = [f"P{i}" for i in range(30, 38)]


def extract_p_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    p_matrix = df[p_cols].values

    # Global P statistics per restaurant
    df["p_sum"] = np.sum(p_matrix, axis=1)
    df["p_mean"] = np.mean(p_matrix, axis=1)
    df["p_std"] = np.std(p_matrix, axis=1)
    df["p_max"] = np.max(p_matrix, axis=1)
    df["p_min"] = np.min(p_matrix, axis=1)
    df["p_median"] = np.median(p_matrix, axis=1)
    df["p_zeros_count"] = np.sum(p_matrix == 0, axis=1)

    # Subgroup statistics
    df["p_demo_sum"] = df[p_demo_cols].sum(axis=1)
    df["p_demo_mean"] = df[p_demo_cols].mean(axis=1)
    df["p_demo_std"] = df[p_demo_cols].std(axis=1)

    df["p_re_sum"] = df[p_re_cols].sum(axis=1)
    df["p_re_mean"] = df[p_re_cols].mean(axis=1)
    df["p_re_std"] = df[p_re_cols].std(axis=1)

    df["p_comm_sum"] = df[p_comm_cols].sum(axis=1)
    df["p_comm_mean"] = df[p_comm_cols].mean(axis=1)
    df["p_comm_std"] = df[p_comm_cols].std(axis=1)

    # Domain interaction ratios & products
    df["p_ratio_demo_re"] = (df["p_demo_sum"] + 1.0) / (df["p_re_sum"] + 1.0)
    df["p_ratio_comm_re"] = (df["p_comm_sum"] + 1.0) / (df["p_re_sum"] + 1.0)
    df["p_prod_p1_p2"] = df["P1"] * df["P2"]
    df["p_ratio_p1_p2"] = df["P1"] / (df["P2"] + 1e-5)
    df["p_sum_p11_p12"] = df["P11"] + df["P12"]
    df["p_ratio_p19_p20"] = df["P19"] / (df["P20"] + 1e-5)
    df["p_ratio_p28_p29"] = df["P28"] / (df["P29"] + 1e-5)

    return df


train_df = extract_p_aggregations(train_df)
test_df = extract_p_aggregations(test_df)

# ---------------------------------------------------------
# 5. Dimensionality Reduction (PCA & TruncatedSVD on P-Variables)
# ---------------------------------------------------------
pca_scaler = StandardScaler()
p_train_scaled = pca_scaler.fit_transform(train_df[p_cols])
p_test_scaled = pca_scaler.transform(test_df[p_cols])

pca = PCA(n_components=5, random_state=42)
pca_train = pca.fit_transform(p_train_scaled)
pca_test = pca.transform(p_test_scaled)

svd = TruncatedSVD(n_components=3, random_state=42)
svd_train = svd.fit_transform(p_train_scaled)
svd_test = svd.transform(p_test_scaled)

for i in range(5):
    train_df[f"pca_p_{i}"] = pca_train[:, i]
    test_df[f"pca_p_{i}"] = pca_test[:, i]

for i in range(3):
    train_df[f"svd_p_{i}"] = svd_train[:, i]
    test_df[f"svd_p_{i}"] = svd_test[:, i]

# ---------------------------------------------------------
# 6. Target Transformation & Stratified Cross-Validation Split
# ---------------------------------------------------------
train_df["log_revenue"] = np.log1p(train_df["revenue"])

n_splits = 5
revenue_bins = pd.qcut(train_df["revenue"], q=n_splits, labels=False)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, revenue_bins)):
    train_df.loc[val_idx, "fold"] = fold_idx

cols_to_drop = ["Open Date", "City", "City Group", "Type"]
train_processed = train_df.drop(
    columns=[c for c in cols_to_drop if c in train_df.columns]
)
test_processed = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])


# =========================================================
# 7. Model Architectures & Factory Definitions
# =========================================================
class RevenueMLP(nn.Module):
    """Compact Regularized Multi-Layer Perceptron for Small-Sample Tabular Regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32, 16],
        dropout_rate: float = 0.25,
    ):
        super(RevenueMLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class LogRMSELoss(nn.Module):
    def __init__(self):
        super(LogRMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + 1e-8)


def get_torch_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-2,
    epochs: int = 150,
) -> Tuple[optim.Optimizer, Any]:
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    return optimizer, scheduler


class ModelFactory:
    @staticmethod
    def get_lightgbm_params(seed: int = 42) -> Dict[str, Any]:
        return {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 5,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "random_state": seed,
            "verbose": -1,
            "n_jobs": -1,
        }

    @staticmethod
    def get_xgboost_params(seed: int = 42) -> Dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 1.5,
            "random_state": seed,
            "n_jobs": -1,
        }

    @staticmethod
    def get_linear_models(seed: int = 42) -> Dict[str, Any]:
        return {
            "ridge": Ridge(alpha=10.0, random_state=seed),
            "elastic_net": ElasticNet(
                alpha=0.1, l1_ratio=0.5, random_state=seed, max_iter=2000
            ),
            "huber": HuberRegressor(epsilon=1.35, max_iter=1000, alpha=1.0),
        }

    @staticmethod
    def get_ensemble_trees(seed: int = 42) -> Dict[str, Any]:
        return {
            "random_forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=300,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=3,
                subsample=0.8,
                random_state=seed,
            ),
        }


def compute_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, is_log_scale: bool = False
) -> float:
    if is_log_scale:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    y_pred_orig = np.clip(y_pred_orig, a_min=0.0, a_max=None)
    mse = np.mean((y_true_orig - y_pred_orig) ** 2)
    return float(np.sqrt(mse))


# =========================================================
# 8. 5-Fold Cross-Validation Training Loop
# =========================================================
feature_cols = [
    c
    for c in train_processed.columns
    if c not in ["id", "revenue", "log_revenue", "fold"]
]
n_features = len(feature_cols)
n_folds = int(train_processed["fold"].nunique())
y_true_orig = train_processed["revenue"].values

model_keys = [
    "lightgbm",
    "xgboost",
    "ridge",
    "elastic_net",
    "huber",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "pytorch_mlp",
]

oof_predictions = {m: np.zeros(len(train_processed)) for m in model_keys}
test_predictions = {m: np.zeros(len(test_processed)) for m in model_keys}

torch.manual_seed(42)
np.random.seed(42)

for fold in range(n_folds):
    train_mask = train_processed["fold"] != fold
    val_mask = train_processed["fold"] == fold

    # Leak-free PCA computation per fold
    pca_scaler = StandardScaler()
    p_tr_scaled = pca_scaler.fit_transform(train_processed.loc[train_mask, p_cols].values)
    p_val_scaled = pca_scaler.transform(train_processed.loc[val_mask, p_cols].values)
    p_te_scaled = pca_scaler.transform(test_processed[p_cols].values)

    pca = PCA(n_components=5, random_state=42)
    pca_tr = pca.fit_transform(p_tr_scaled)
    pca_val = pca.transform(p_val_scaled)
    pca_te = pca.transform(p_te_scaled)

    pca_col_indices = [feature_cols.index(f"pca_p_{i}") for i in range(5)]

    X_tr = train_processed.loc[train_mask, feature_cols].values
    X_val = train_processed.loc[val_mask, feature_cols].values
    X_te = test_processed[feature_cols].values

    for idx_pca, col_idx in enumerate(pca_col_indices):
        X_tr[:, col_idx] = pca_tr[:, idx_pca]
        X_val[:, col_idx] = pca_val[:, idx_pca]
        X_te[:, col_idx] = pca_te[:, idx_pca]

    y_tr_log = train_processed.loc[train_mask, "log_revenue"].values
    y_val_log = train_processed.loc[val_mask, "log_revenue"].values

    # Strict training fold scaling for linear & neural models
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_te_scaled = scaler.transform(X_te)

    # --- A. LightGBM ---
    lgb_params = ModelFactory.get_lightgbm_params(seed=42 + fold)
    lgb_reg = lgb.LGBMRegressor(**lgb_params)
    lgb_reg.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_predictions["lightgbm"][val_mask] = lgb_reg.predict(X_val)
    test_predictions["lightgbm"] += lgb_reg.predict(X_te) / n_folds

    # --- B. XGBoost ---
    xgb_params = ModelFactory.get_xgboost_params(seed=42 + fold)
    xgb_reg = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_reg.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], verbose=False)
    oof_predictions["xgboost"][val_mask] = xgb_reg.predict(X_val)
    test_predictions["xgboost"] += xgb_reg.predict(X_te) / n_folds

    # --- C. Linear Models (Ridge, ElasticNet, Huber) ---
    linear_models = ModelFactory.get_linear_models(seed=42 + fold)
    for m_name in ["ridge", "elastic_net", "huber"]:
        lin_model = linear_models[m_name]
        lin_model.fit(X_tr_scaled, y_tr_log)
        oof_predictions[m_name][val_mask] = lin_model.predict(X_val_scaled)
        test_predictions[m_name] += lin_model.predict(X_te_scaled) / n_folds

    # --- D. Tree Ensemble Regressors ---
    tree_models = ModelFactory.get_ensemble_trees(seed=42 + fold)
    for m_name in ["random_forest", "extra_trees", "gradient_boosting"]:
        t_model = tree_models[m_name]
        t_model.fit(X_tr, y_tr_log)
        oof_predictions[m_name][val_mask] = t_model.predict(X_val)
        test_predictions[m_name] += t_model.predict(X_te) / n_folds

    # --- E. PyTorch Regularized MLP ---
    mlp = RevenueMLP(input_dim=n_features, hidden_dims=[64, 32, 16], dropout_rate=0.25)
    mlp_criterion = LogRMSELoss()
    mlp_opt, mlp_sched = get_torch_optimizer_and_scheduler(
        mlp, lr=1.5e-3, weight_decay=1e-2, epochs=150
    )

    t_X_tr = torch.tensor(X_tr_scaled, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr_log, dtype=torch.float32)
    t_X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
    t_y_val = torch.tensor(y_val_log, dtype=torch.float32)
    t_X_te = torch.tensor(X_te_scaled, dtype=torch.float32)

    best_val_loss = float("inf")
    best_mlp_state = copy.deepcopy(mlp.state_dict())

    mlp.train()
    for epoch in range(150):
        mlp_opt.zero_grad()
        out = mlp(t_X_tr)
        loss = mlp_criterion(out, t_y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
        mlp_opt.step()
        mlp_sched.step()

        mlp.eval()
        with torch.no_grad():
            val_out = mlp(t_X_val)
            val_loss = float(mlp_criterion(val_out, t_y_val).item())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_mlp_state = copy.deepcopy(mlp.state_dict())
        mlp.train()

    mlp.load_state_dict(best_mlp_state)
    mlp.eval()
    with torch.no_grad():
        oof_predictions["pytorch_mlp"][val_mask] = mlp(t_X_val).cpu().numpy().flatten()
        test_predictions["pytorch_mlp"] += mlp(t_X_te).cpu().numpy().flatten() / n_folds

# ---------------------------------------------------------
# 9. Ensemble Blending & Validation Evaluation
# ---------------------------------------------------------
best_single_name = None
best_single_rmse = float("inf")

for m_name in model_keys:
    rmse_m = compute_rmse(y_true_orig, oof_predictions[m_name], is_log_scale=True)
    if rmse_m < best_single_rmse:
        best_single_rmse = rmse_m
        best_single_name = m_name

oof_mat = np.column_stack([oof_predictions[m] for m in model_keys])
test_mat = np.column_stack([test_predictions[m] for m in model_keys])


def blend_objective(w: np.ndarray) -> float:
    w = np.maximum(0.0, w)
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    log_preds = oof_mat @ w
    rev_preds = np.clip(np.expm1(log_preds), a_min=0.0, a_max=None)
    return float(np.sqrt(np.mean((y_true_orig - rev_preds) ** 2)))


init_weights = np.ones(len(model_keys)) / len(model_keys)
bounds = [(0.0, 1.0) for _ in model_keys]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-9},
)

optimal_weights = np.maximum(0.0, opt_res.x)
optimal_weights = optimal_weights / np.sum(optimal_weights)

oof_ensemble_rev = np.clip(np.expm1(oof_mat @ optimal_weights), a_min=0.0, a_max=None)
test_ensemble_rev = np.clip(np.expm1(test_mat @ optimal_weights), a_min=0.0, a_max=None)
ensemble_rmse = compute_rmse(y_true_orig, oof_ensemble_rev, is_log_scale=False)

if ensemble_rmse <= best_single_rmse:
    final_test_preds = test_ensemble_rev
    final_validation_score = ensemble_rmse
else:
    final_test_preds = np.clip(
        np.expm1(test_predictions[best_single_name]), a_min=0.0, a_max=None
    )
    final_validation_score = best_single_rmse

# ---------------------------------------------------------
# 10. Generate Submission & Print Metric
# ---------------------------------------------------------
submission_path = Path("./submission/submission.csv")
submission_df = pd.DataFrame(
    {"id": test_processed["id"].astype(int), "revenue": final_test_preds}
)
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_validation_score}")