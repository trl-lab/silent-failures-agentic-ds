import os
import random
import warnings
from typing import Any, Dict, List

from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. Global Setup and Reproducibility
# ==============================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# ==============================================================================
# 1. Feature Engineering Utilities
# ==============================================================================
def engineer_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    id_col: str = None,
    group_col: str = None,
    y_train: Any = None,
):
    """Applies strictly isolated feature engineering fitted on df_train:

    - Row-level missingness counts and indicator flags
    - Skewness log1p / power transforms
    - Frequency, label, and regularized smoothed target encoding
    - Correlation-screened domain ratios & differences
    - Hierarchical group summary aggregations
    """
    train_feat = df_train.copy()
    test_feat = df_test.copy()

    # 1.1 Row-wise missingness count
    train_feat["row_missing_count"] = df_train.isnull().sum(axis=1).astype(np.float32)
    test_feat["row_missing_count"] = df_test.isnull().sum(axis=1).astype(np.float32)

    # 1.2 Missing value indicators and train-based median imputation
    for col in num_cols:
        if train_feat[col].isnull().sum() > 0 or test_feat[col].isnull().sum() > 0:
            train_feat[f"{col}_isna"] = train_feat[col].isnull().astype(np.float32)
            test_feat[f"{col}_isna"] = test_feat[col].isnull().astype(np.float32)

            med_val = float(train_feat[col].median())
            if np.isnan(med_val):
                med_val = 0.0
            train_feat[col] = train_feat[col].fillna(med_val)
            test_feat[col] = test_feat[col].fillna(med_val)

    # 1.3 Skewness transforms on non-negative continuous features
    for col in num_cols:
        min_val = train_feat[col].min()
        if min_val >= 0:
            skew = train_feat[col].skew()
            if abs(skew) > 1.2:
                train_feat[f"{col}_log1p"] = np.log1p(train_feat[col].clip(lower=0)).astype(np.float32)
                test_feat[f"{col}_log1p"] = np.log1p(test_feat[col].clip(lower=0)).astype(np.float32)

    # 1.4 Frequency, Label, and Smoothed Target Encoding for Categoricals
    global_mean = float(np.mean(y_train)) if y_train is not None else 50.0
    for col in cat_cols:
        freq = train_feat[col].value_counts(normalize=True).to_dict()
        train_feat[f"{col}_freq"] = (
            train_feat[col].map(freq).fillna(0).astype(np.float32)
        )
        test_feat[f"{col}_freq"] = test_feat[col].map(freq).fillna(0).astype(np.float32)

        # Smoothed Target Encoding with smoothing parameter m=10
        if y_train is not None:
            m = 10.0
            te_df = pd.DataFrame({"cat": train_feat[col].astype(str), "target": np.array(y_train, dtype=np.float32)})
            grp = te_df.groupby("cat")["target"].agg(["count", "mean"])
            te_map = ((grp["count"] * grp["mean"] + m * global_mean) / (grp["count"] + m)).to_dict()
            train_feat[f"{col}_te"] = train_feat[col].astype(str).map(te_map).fillna(global_mean).astype(np.float32)
            test_feat[f"{col}_te"] = test_feat[col].astype(str).map(te_map).fillna(global_mean).astype(np.float32)

        train_feat[col] = train_feat[col].astype(str).fillna("__missing__")
        test_feat[col] = test_feat[col].astype(str).fillna("__missing__")

        le = LabelEncoder()
        all_vals = list(train_feat[col].unique()) + ["__unseen__"]
        le.fit(all_vals)

        train_feat[f"{col}_encoded"] = train_feat[col].apply(
            lambda s: s if s in le.classes_ else "__unseen__"
        )
        train_feat[f"{col}_encoded"] = le.transform(train_feat[f"{col}_encoded"])

        test_feat[f"{col}_encoded"] = test_feat[col].apply(
            lambda s: s if s in le.classes_ else "__unseen__"
        )
        test_feat[f"{col}_encoded"] = le.transform(test_feat[f"{col}_encoded"])

    # 1.5 Correlation-Screened Ratios, Differences & Multiplicative Interactions
    if len(num_cols) >= 2:
        top_num = num_cols[: min(16, len(num_cols))]
        corr_matrix = train_feat[top_num].corr().abs()
        for i in range(len(top_num)):
            for j in range(i + 1, len(top_num)):
                col1 = top_num[i]
                col2 = top_num[j]
                corr_val = corr_matrix.loc[col1, col2] if (col1 in corr_matrix.index and col2 in corr_matrix.columns) else 0.5
                if 0.10 <= corr_val <= 0.90:
                    train_feat[f"{col1}_minus_{col2}"] = (train_feat[col1] - train_feat[col2]).astype(np.float32)
                    test_feat[f"{col1}_minus_{col2}"] = (test_feat[col1] - test_feat[col2]).astype(np.float32)

                    train_feat[f"{col1}_div_{col2}"] = (
                        train_feat[col1] / (train_feat[col2].abs() + 1e-4)
                    ).astype(np.float32)
                    test_feat[f"{col1}_div_{col2}"] = (
                        test_feat[col1] / (test_feat[col2].abs() + 1e-4)
                    ).astype(np.float32)

                    train_feat[f"{col1}_mult_{col2}"] = (
                        train_feat[col1] * train_feat[col2]
                    ).astype(np.float32)
                    test_feat[f"{col1}_mult_{col2}"] = (
                        test_feat[col1] * test_feat[col2]
                    ).astype(np.float32)

    # 1.6 Group Aggregations & State-Relative Standardized Z-Scores
    if group_col is not None and group_col in df_train.columns:
        for num_c in num_cols:
            grp_stats = (
                train_feat.groupby(group_col)[num_c]
                .agg(["mean", "std", "min", "max"])
                .reset_index()
            )
            grp_stats.columns = [
                group_col,
                f"{num_c}_grp_mean",
                f"{num_c}_grp_std",
                f"{num_c}_grp_min",
                f"{num_c}_grp_max",
            ]
            grp_stats[f"{num_c}_grp_std"] = grp_stats[f"{num_c}_grp_std"].fillna(0)

            train_feat = train_feat.merge(grp_stats, on=group_col, how="left")
            test_feat = test_feat.merge(grp_stats, on=group_col, how="left")

            mean_fallback = float(train_feat[num_c].mean())
            std_fallback = float(train_feat[num_c].std()) if float(train_feat[num_c].std()) > 0 else 1.0
            min_fallback = float(train_feat[num_c].min())
            max_fallback = float(train_feat[num_c].max())

            train_feat[f"{num_c}_grp_mean"] = train_feat[f"{num_c}_grp_mean"].fillna(mean_fallback)
            train_feat[f"{num_c}_grp_std"] = train_feat[f"{num_c}_grp_std"].fillna(std_fallback)
            train_feat[f"{num_c}_grp_min"] = train_feat[f"{num_c}_grp_min"].fillna(min_fallback)
            train_feat[f"{num_c}_grp_max"] = train_feat[f"{num_c}_grp_max"].fillna(max_fallback)

            test_feat[f"{num_c}_grp_mean"] = test_feat[f"{num_c}_grp_mean"].fillna(mean_fallback)
            test_feat[f"{num_c}_grp_std"] = test_feat[f"{num_c}_grp_std"].fillna(std_fallback)
            test_feat[f"{num_c}_grp_min"] = test_feat[f"{num_c}_grp_min"].fillna(min_fallback)
            test_feat[f"{num_c}_grp_max"] = test_feat[f"{num_c}_grp_max"].fillna(max_fallback)

            train_feat[f"{num_c}_state_zscore"] = (
                (train_feat[num_c] - train_feat[f"{num_c}_grp_mean"])
                / (train_feat[f"{num_c}_grp_std"] + 1e-4)
            ).astype(np.float32)
            test_feat[f"{num_c}_state_zscore"] = (
                (test_feat[num_c] - test_feat[f"{num_c}_grp_mean"])
                / (test_feat[f"{num_c}_grp_std"] + 1e-4)
            ).astype(np.float32)

            train_feat[f"{num_c}_diff_grp_mean"] = (
                train_feat[num_c] - train_feat[f"{num_c}_grp_mean"]
            ).astype(np.float32)
            test_feat[f"{num_c}_diff_grp_mean"] = (
                test_feat[num_c] - test_feat[f"{num_c}_grp_mean"]
            ).astype(np.float32)

    # 1.7 Clean infinite values and drop non-numeric / raw categoricals
    train_feat = train_feat.replace([np.inf, -np.inf], np.nan)
    test_feat = test_feat.replace([np.inf, -np.inf], np.nan)

    for col in cat_cols:
        if col in train_feat.columns:
            train_feat = train_feat.drop(columns=[col])
        if col in test_feat.columns:
            test_feat = test_feat.drop(columns=[col])

    if id_col is not None:
        if id_col in train_feat.columns:
            train_feat = train_feat.drop(columns=[id_col])
        if id_col in test_feat.columns:
            test_feat = test_feat.drop(columns=[id_col])

    for c in train_feat.columns:
        if train_feat[c].isnull().sum() > 0 or test_feat[c].isnull().sum() > 0:
            med = float(train_feat[c].median())
            if np.isnan(med):
                med = 0.0
            train_feat[c] = train_feat[c].fillna(med)
            test_feat[c] = test_feat[c].fillna(med)

    return train_feat, test_feat


# ==============================================================================
# 2. Model Architectures & Configuration Factories
# ==============================================================================
class TabularResidualBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)

        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features), nn.LayerNorm(out_features)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.dropout1(self.act1(self.ln1(self.fc1(x))))
        out = self.dropout2(self.act2(self.ln2(self.fc2(out))))
        return out + residual


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.15,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                TabularResidualBlock(
                    in_features=hidden_dims[i],
                    out_features=hidden_dims[i + 1],
                    dropout_rate=dropout_rate,
                )
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        out = self.head(x)
        return out.squeeze(-1)


def get_catboost_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": seed,
        "verbose": False,
        "early_stopping_rounds": 50,
    }


def get_et_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": seed,
        "n_jobs": -1,
    }


def get_lgb_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": seed,
        "n_estimators": 2000,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> Dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_estimators": 2000,
        "n_jobs": -1,
        "tree_method": "hist",
    }


# ==============================================================================
# 3. Data Ingestion & Pipeline Orchestration
# ==============================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")
target_col = "Population for whom broadband available, 2019 (%)"

y_train_raw = train_df[target_col].copy()
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

id_col = None
for col in ["id", "ID", "Id", "FIPS", "GEOID"]:
    if col in X_train_raw.columns and col in X_test_raw.columns:
        id_col = col
        break

categorical_cols = []
numerical_cols = []
for col in X_train_raw.columns:
    if col == id_col:
        continue
    if (
        X_train_raw[col].dtype == "object"
        or pd.api.types.is_categorical_dtype(X_train_raw[col])
        or pd.api.types.is_string_dtype(X_train_raw[col])
    ):
        categorical_cols.append(col)
    else:
        if X_train_raw[col].nunique() < 10 and X_train_raw[col].dtype in [
            "int64",
            "int32",
            "int16",
            "int8",
        ]:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

group_column = None
for candidate in ["State", "state", "STATE", "state_abbr", "StateAbbr", "Region"]:
    if candidate in X_train_raw.columns:
        group_column = candidate
        break
if group_column is None and len(categorical_cols) > 0:
    group_column = categorical_cols[0]

# Generate Full Test Feature Matrix
X_train_full_engineered, X_test_engineered = engineer_features(
    X_train_raw,
    X_test_raw,
    numerical_cols,
    categorical_cols,
    id_col=id_col,
    group_col=group_column,
    y_train=y_train_raw,
)

common_cols = [
    c for c in X_train_full_engineered.columns if c in X_test_engineered.columns
]
X_test_engineered = X_test_engineered[common_cols].astype(np.float32)

# Stratified K-Fold CV
N_SPLITS = 10
num_bins = min(10, max(2, len(y_train_raw) // 100))
y_binned = pd.qcut(y_train_raw, q=num_bins, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

fold_assignments = np.zeros(len(train_df), dtype=int)
for fold, (_, val_idx) in enumerate(skf.split(X_train_raw, y_binned)):
    fold_assignments[val_idx] = fold

# ==============================================================================
# 4. Model Training & Out-of-Fold Cross-Validation
# ==============================================================================
n_train_samples = len(train_df)
y_all_true = np.zeros(n_train_samples, dtype=np.float32)

oof_lgb = np.zeros(n_train_samples, dtype=np.float32)
oof_xgb = np.zeros(n_train_samples, dtype=np.float32)
oof_cat = np.zeros(n_train_samples, dtype=np.float32)
oof_et = np.zeros(n_train_samples, dtype=np.float32)
oof_nn = np.zeros(n_train_samples, dtype=np.float32)

test_preds_lgb = np.zeros((len(test_df), N_SPLITS), dtype=np.float32)
test_preds_xgb = np.zeros((len(test_df), N_SPLITS), dtype=np.float32)
test_preds_cat = np.zeros((len(test_df), N_SPLITS), dtype=np.float32)
test_preds_et = np.zeros((len(test_df), N_SPLITS), dtype=np.float32)
test_preds_nn = np.zeros((len(test_df), N_SPLITS), dtype=np.float32)

for fold in range(N_SPLITS):
    train_idx = np.where(fold_assignments != fold)[0]
    val_idx = np.where(fold_assignments == fold)[0]

    df_tr_fold = X_train_raw.iloc[train_idx].copy()
    df_val_fold = X_train_raw.iloc[val_idx].copy()
    y_tr_fold = y_train_raw.iloc[train_idx].values.astype(np.float32)
    y_val_fold = y_train_raw.iloc[val_idx].values.astype(np.float32)

    tr_feat_fold, val_feat_fold = engineer_features(
        df_tr_fold,
        df_val_fold,
        numerical_cols,
        categorical_cols,
        id_col=id_col,
        group_col=group_column,
    )

    feature_cols = [c for c in common_cols if c in tr_feat_fold.columns]
    X_train_f = tr_feat_fold[feature_cols].values.astype(np.float32)
    X_val_f = val_feat_fold[feature_cols].values.astype(np.float32)
    X_test_f = X_test_engineered[feature_cols].values.astype(np.float32)

    y_all_true[val_idx] = y_val_fold

    # 4.1 LightGBM Regressor
    lgb_params = get_lgb_params(seed=RANDOM_SEED + fold)
    dtrain = lgb.Dataset(X_train_f, label=y_tr_fold)
    dval = lgb.Dataset(X_val_f, label=y_val_fold, reference=dtrain)
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=lgb_callbacks,
    )

    pred_val_lgb = lgb_model.predict(X_val_f, num_iteration=lgb_model.best_iteration)
    oof_lgb[val_idx] = pred_val_lgb
    test_preds_lgb[:, fold] = lgb_model.predict(
        X_test_f, num_iteration=lgb_model.best_iteration
    )

    # 4.2 XGBoost Regressor
    xgb_params = get_xgb_params(seed=RANDOM_SEED + fold)
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(
        X_train_f,
        y_tr_fold,
        eval_set=[(X_val_f, y_val_fold)],
        verbose=False,
    )

    pred_val_xgb = xgb_model.predict(X_val_f)
    oof_xgb[val_idx] = pred_val_xgb
    test_preds_xgb[:, fold] = xgb_model.predict(X_test_f)

    # 4.3 CatBoost Regressor
    cat_params = get_catboost_params(seed=RANDOM_SEED + fold)
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train_f,
        y_tr_fold,
        eval_set=(X_val_f, y_val_fold),
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict(X_val_f)
    test_preds_cat[:, fold] = cat_model.predict(X_test_f)

    # 4.4 ExtraTrees Regressor
    et_params = get_et_params(seed=RANDOM_SEED + fold)
    et_model = ExtraTreesRegressor(**et_params)
    et_model.fit(X_train_f, y_tr_fold)
    oof_et[val_idx] = et_model.predict(X_val_f)
    test_preds_et[:, fold] = et_model.predict(X_test_f)

    # 4.5 TabularResNet
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_f)
    X_val_scaled = scaler.transform(X_val_f)
    X_test_scaled = scaler.transform(X_test_f)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr_fold, dtype=torch.float32),
        ),
        batch_size=64,
        shuffle=True,
        drop_last=False,
    )

    val_tensor_x = torch.tensor(X_val_scaled, dtype=torch.float32)
    test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32)

    nn_model = TabularResNet(
        input_dim=len(feature_cols), hidden_dims=[256, 128, 64], dropout_rate=0.15
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=3e-3, weight_decay=1e-4)

    num_epochs = 70
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5
    )

    best_val_rmse = float("inf")
    best_model_weights = None
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = nn_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_out = nn_model(val_tensor_x.to(device)).cpu().numpy()
            val_rmse = np.sqrt(mean_squared_error(y_val_fold, val_out))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
    nn_model.eval()
    with torch.no_grad():
        best_val_pred_nn = nn_model(val_tensor_x.to(device)).cpu().numpy()
        best_test_pred_nn = nn_model(test_tensor_x.to(device)).cpu().numpy()

    oof_nn[val_idx] = best_val_pred_nn
    test_preds_nn[:, fold] = best_test_pred_nn


# ==============================================================================
# 5. Ensemble Optimization, Submission Generation & Metric Reporting
# ==============================================================================
OOF_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_nn])


def rmse_objective(weights):
    weights = np.array(weights)
    pred_blend = np.dot(OOF_matrix, weights)
    pred_blend_clipped = np.clip(pred_blend, 0.0, 100.0)
    return np.sqrt(mean_squared_error(y_all_true, pred_blend_clipped))


initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
bounds = [(0.0, 1.0)] * 5
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    rmse_objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x

oof_ensemble = np.dot(OOF_matrix, best_weights)
oof_ensemble_clipped = np.clip(oof_ensemble, 0.0, 100.0)
final_score = np.sqrt(mean_squared_error(y_all_true, oof_ensemble_clipped))

test_lgb_mean = np.mean(test_preds_lgb, axis=1)
test_xgb_mean = np.mean(test_preds_xgb, axis=1)
test_cat_mean = np.mean(test_preds_cat, axis=1)
test_et_mean = np.mean(test_preds_et, axis=1)
test_nn_mean = np.mean(test_preds_nn, axis=1)

Test_matrix = np.column_stack([test_lgb_mean, test_xgb_mean, test_cat_mean, test_et_mean, test_nn_mean])
final_test_preds = np.dot(Test_matrix, best_weights)
final_test_preds_clipped = np.clip(final_test_preds, 0.0, 100.0)

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(final_test_preds_clipped)),
        target_col: final_test_preds_clipped,
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
