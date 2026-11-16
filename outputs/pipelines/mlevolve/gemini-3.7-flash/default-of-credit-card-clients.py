import os
import random
from typing import Any, Dict, List, Tuple
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# =========================================================================
# 0. Global Setup and Reproducibility
# =========================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 1. Data Processing and Feature Engineering
# =========================================================================
def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    col_rename_map = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "PAY_1": "PAY_0",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": "default payment next month",
    }

    train_df.rename(columns=col_rename_map, inplace=True)
    test_df.rename(columns=col_rename_map, inplace=True)

    target_col = "default payment next month"
    if target_col not in train_df.columns:
        for candidate in ["default", "target", "default_payment_next_month"]:
            if candidate in train_df.columns:
                train_df.rename(columns={candidate: target_col}, inplace=True)
                break

    if "PAY_1" in train_df.columns and "PAY_0" not in train_df.columns:
        train_df.rename(columns={"PAY_1": "PAY_0"}, inplace=True)
    if "PAY_1" in test_df.columns and "PAY_0" not in test_df.columns:
        test_df.rename(columns={"PAY_1": "PAY_0"}, inplace=True)

    return train_df, test_df, target_col


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean categorical anomalies
    if "EDUCATION" in df.columns:
        df["EDUCATION_CLEAN"] = df["EDUCATION"].apply(
            lambda x: 4 if x not in [1, 2, 3] else x
        )
    if "MARRIAGE" in df.columns:
        df["MARRIAGE_CLEAN"] = df["MARRIAGE"].apply(
            lambda x: 3 if x not in [1, 2] else x
        )

    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    # Delinquency and Repayment Status Features
    df["pay_max"] = df[pay_cols].max(axis=1)
    df["pay_min"] = df[pay_cols].min(axis=1)
    df["pay_mean"] = df[pay_cols].mean(axis=1)
    df["pay_std"] = df[pay_cols].std(axis=1).fillna(0)
    df["pay_sum"] = df[pay_cols].sum(axis=1)
    df["pay_trend"] = df["PAY_0"] - df["PAY_6"]
    df["pay_recent_diff"] = df["PAY_0"] - df["PAY_2"]
    df["num_delayed_months"] = (df[pay_cols] > 0).sum(axis=1)
    df["has_severe_delay"] = (df[pay_cols] >= 2).any(axis=1).astype(int)
    df["never_delayed"] = (df[pay_cols] <= 0).all(axis=1).astype(int)

    # Financial & Credit Limit Utilization Features
    limit = df["LIMIT_BAL"] + 1.0
    for i, b_col in enumerate(bill_cols, 1):
        df[f"util_{i}"] = df[b_col] / limit
        df[f"over_limit_{i}"] = (df[b_col] > df["LIMIT_BAL"]).astype(int)

    util_cols = [f"util_{i}" for i in range(1, 7)]
    df["util_mean"] = df[util_cols].mean(axis=1)
    df["util_std"] = df[util_cols].std(axis=1).fillna(0)
    df["util_max"] = df[util_cols].max(axis=1)
    df["util_min"] = df[util_cols].min(axis=1)
    df["util_trend"] = df["util_1"] - df["util_6"]
    df["over_limit_count"] = df[[f"over_limit_{i}" for i in range(1, 7)]].sum(axis=1)

    # Payment-to-Bill Ratios and Unpaid Balances
    for i in range(1, 6):
        df[f"pay_ratio_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i+1}"].abs() + 1.0)
        df[f"unpaid_amt_{i}"] = df[f"BILL_AMT{i}"] - df[f"PAY_AMT{i}"]

    pay_ratio_cols = [f"pay_ratio_{i}" for i in range(1, 6)]
    df["pay_ratio_mean"] = df[pay_ratio_cols].mean(axis=1)
    df["pay_ratio_max"] = df[pay_ratio_cols].max(axis=1)
    df["pay_ratio_min"] = df[pay_ratio_cols].min(axis=1)

    unpaid_cols = [f"unpaid_amt_{i}" for i in range(1, 6)]
    df["unpaid_amt_mean"] = df[unpaid_cols].mean(axis=1)
    df["unpaid_amt_max"] = df[unpaid_cols].max(axis=1)

    # Decay-Weighted Temporal Aggregates
    decay_w = np.exp(-0.35 * np.arange(6))
    decay_w = decay_w / decay_w.sum()
    df["pay_decay_weighted"] = sum(w * df[col] for w, col in zip(decay_w, pay_cols))
    df["bill_decay_weighted"] = sum(w * df[col] for w, col in zip(decay_w, bill_cols))
    df["pay_amt_decay_weighted"] = sum(w * df[col] for w, col in zip(decay_w, pay_amt_cols))

    # Delinquency Escalation & Acceleration
    df["pay_acceleration"] = (df["PAY_0"] - df["PAY_2"]) - (df["PAY_2"] - df["PAY_4"])
    df["pay_recent_to_old_ratio"] = (df["PAY_0"] + 3.0) / (df["PAY_6"] + 3.0)
    df["num_severe_delayed"] = (df[pay_cols] >= 2).sum(axis=1)

    # Payment Stress and Overlimit Delinquency Indicators
    for i in range(1, 7):
        df[f"zero_pay_stress_{i}"] = ((df[f"BILL_AMT{i}"] > 0) & (df[f"PAY_AMT{i}"] == 0)).astype(int)
        df[f"overlimit_and_delinq_{i}"] = (
            (df[f"BILL_AMT{i}"] > df["LIMIT_BAL"]) & (df[pay_cols[i - 1]] > 0)
        ).astype(int)
    zero_stress_cols = [f"zero_pay_stress_{i}" for i in range(1, 7)]
    df["zero_pay_stress_sum"] = df[zero_stress_cols].sum(axis=1)
    df["zero_pay_stress_recent"] = df["zero_pay_stress_1"]

    overlimit_delinq_cols = [f"overlimit_and_delinq_{i}" for i in range(1, 7)]
    df["overlimit_delinq_sum"] = df[overlimit_delinq_cols].sum(axis=1)
    df["overlimit_delinq_recent"] = df["overlimit_and_delinq_1"]

    # Month-over-Month Differentials & Sequential Status Changes
    for i in range(1, 6):
        df[f"bill_mom_diff_{i}"] = df[f"BILL_AMT{i}"] - df[f"BILL_AMT{i+1}"]
        df[f"pay_amt_mom_diff_{i}"] = df[f"PAY_AMT{i}"] - df[f"PAY_AMT{i+1}"]
    for i in range(5):
        df[f"pay_status_diff_{i}"] = df[pay_cols[i]] - df[pay_cols[i+1]]

    # Exact Linear Trend Slopes (over months 6 -> 1, ordered chronologically)
    slope_weights = np.array([-5, -3, -1, 1, 3, 5]) / 35.0
    df["pay_slope"] = sum(w * df[col] for w, col in zip(slope_weights, reversed(pay_cols)))
    df["bill_slope"] = sum(w * df[col] for w, col in zip(slope_weights, reversed(bill_cols)))
    df["pay_amt_slope"] = sum(w * df[col] for w, col in zip(slope_weights, reversed(pay_amt_cols)))

    # Sequential Risk & Delinquency Duration Features
    d0 = (df["PAY_0"] > 0).astype(int)
    d1 = (df["PAY_2"] > 0).astype(int)
    d2 = (df["PAY_3"] > 0).astype(int)
    d3 = (df["PAY_4"] > 0).astype(int)
    d4 = (df["PAY_5"] > 0).astype(int)
    d5 = (df["PAY_6"] > 0).astype(int)
    df["consecutive_delinq_duration"] = d0 * (1 + d1 * (1 + d2 * (1 + d3 * (1 + d4 * (1 + d5)))))

    # Monthly Payment Coverage Ratios
    for i in range(1, 7):
        df[f"payment_coverage_{i}"] = df[f"PAY_AMT{i}"] / (df[f"BILL_AMT{i}"].abs() + 1.0)
    coverage_cols = [f"payment_coverage_{i}" for i in range(1, 7)]
    df["payment_coverage_mean"] = df[coverage_cols].mean(axis=1)
    df["payment_coverage_min"] = df[coverage_cols].min(axis=1)

    # Non-linear Risk Interactions
    df["log_limit_x_recent_pay"] = np.log1p(np.maximum(df["LIMIT_BAL"], 0)) * np.maximum(df["PAY_0"], 0)
    df["unpaid_to_limit_ratio_1"] = (df["BILL_AMT1"] - df["PAY_AMT1"]) / (df["LIMIT_BAL"] + 1.0)
    df["unpaid_to_limit_ratio_mean"] = df[unpaid_cols].mean(axis=1) / (df["LIMIT_BAL"] + 1.0)

    # Aggregate Bill and Payment Trajectories
    df["bill_mean"] = df[bill_cols].mean(axis=1)
    df["bill_std"] = df[bill_cols].std(axis=1).fillna(0)
    df["bill_max"] = df[bill_cols].max(axis=1)
    df["bill_min"] = df[bill_cols].min(axis=1)
    df["bill_sum"] = df[bill_cols].sum(axis=1)
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]

    df["pay_amt_mean"] = df[pay_amt_cols].mean(axis=1)
    df["pay_amt_std"] = df[pay_amt_cols].std(axis=1).fillna(0)
    df["pay_amt_max"] = df[pay_amt_cols].max(axis=1)
    df["pay_amt_min"] = df[pay_amt_cols].min(axis=1)
    df["pay_amt_sum"] = df[pay_amt_cols].sum(axis=1)
    df["pay_amt_trend"] = df["PAY_AMT1"] - df["PAY_AMT6"]
    df["total_pay_to_total_bill"] = df["pay_amt_sum"] / (df["bill_sum"].abs() + 1.0)

    # Demographic and Interaction Features
    df["age_to_limit"] = df["AGE"] / limit
    df["limit_per_age"] = df["LIMIT_BAL"] / (df["AGE"] + 1.0)
    df["age_group"] = pd.cut(
        df["AGE"], bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df["sex_edu"] = df["SEX"].astype(str) + "_" + df["EDUCATION_CLEAN"].astype(str)
    df["marriage_edu"] = (
        df["MARRIAGE_CLEAN"].astype(str) + "_" + df["EDUCATION_CLEAN"].astype(str)
    )
    df["demog_group"] = (
        df["SEX"].astype(str)
        + "_"
        + df["EDUCATION_CLEAN"].astype(str)
        + "_"
        + df["MARRIAGE_CLEAN"].astype(str)
    )
    df["pay_status_pair"] = df["PAY_0"].astype(str) + "_" + df["PAY_2"].astype(str)

    return df


train_raw, test_raw, target_col = load_and_preprocess_data()
train_processed = engineer_features(train_raw)
test_processed = engineer_features(test_raw)

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
train_processed["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_processed, train_processed[target_col])
):
    train_processed.loc[val_idx, "fold"] = fold


def add_group_aggregations(
    tr_df: pd.DataFrame, te_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for grp_col in ["demog_group", "age_group"]:
        for num_col in ["LIMIT_BAL", "PAY_0", "bill_decay_weighted", "pay_amt_decay_weighted"]:
            grp_stats = (
                tr_df.groupby(grp_col)[num_col].agg(["mean", "std"]).reset_index()
            )
            grp_stats.columns = [
                grp_col,
                f"{grp_col}_{num_col}_mean",
                f"{grp_col}_{num_col}_std",
            ]
            grp_stats[f"{grp_col}_{num_col}_std"] = grp_stats[
                f"{grp_col}_{num_col}_std"
            ].fillna(0)

            tr_df = tr_df.merge(grp_stats, on=grp_col, how="left")
            te_df = te_df.merge(grp_stats, on=grp_col, how="left")

            tr_df[f"{num_col}_diff_{grp_col}_mean"] = (
                tr_df[num_col] - tr_df[f"{grp_col}_{num_col}_mean"]
            )
            te_df[f"{num_col}_diff_{grp_col}_mean"] = (
                te_df[num_col] - te_df[f"{grp_col}_{num_col}_mean"]
            )
            tr_df[f"{num_col}_ratio_{grp_col}_mean"] = tr_df[num_col] / (
                tr_df[f"{grp_col}_{num_col}_mean"].abs() + 1.0
            )
            te_df[f"{num_col}_ratio_{grp_col}_mean"] = te_df[num_col] / (
                te_df[f"{grp_col}_{num_col}_mean"].abs() + 1.0
            )
    return tr_df, te_df


train_processed, test_processed = add_group_aggregations(train_processed, test_processed)


def compute_oof_target_encoding(
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    te_cols: List[str],
    target: str,
    smoothing: float = 15.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global_mean = tr_df[target].mean()
    for col in te_cols:
        new_col = f"te_{col}"
        tr_df[new_col] = np.nan
        for f in tr_df["fold"].unique():
            tr_mask = tr_df["fold"] != f
            va_mask = tr_df["fold"] == f
            stats = tr_df.loc[tr_mask].groupby(col)[target].agg(["count", "mean"])
            smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            tr_df.loc[va_mask, new_col] = (
                tr_df.loc[va_mask, col].map(smoothed).fillna(global_mean)
            )

        stats_all = tr_df.groupby(col)[target].agg(["count", "mean"])
        smoothed_all = (
            stats_all["count"] * stats_all["mean"] + smoothing * global_mean
        ) / (stats_all["count"] + smoothing)
        te_df[new_col] = te_df[col].map(smoothed_all).fillna(global_mean)
    return tr_df, te_df


te_feature_cols = ["demog_group", "PAY_0", "pay_status_pair", "EDUCATION_CLEAN", "MARRIAGE_CLEAN"]
train_processed, test_processed = compute_oof_target_encoding(
    train_processed, test_processed, te_feature_cols, target_col
)

cat_cols = ["sex_edu", "marriage_edu", "demog_group", "pay_status_pair"]
for col in cat_cols:
    all_cats = sorted(
        list(
            set(train_processed[col].astype(str).unique()).union(
                set(test_processed[col].astype(str).unique())
            )
        )
    )
    cat_type = pd.CategoricalDtype(categories=all_cats, ordered=True)
    train_processed[col] = train_processed[col].astype(str).astype(cat_type).cat.codes
    test_processed[col] = test_processed[col].astype(str).astype(cat_type).cat.codes


# =========================================================================
# 2. Model Architectures & Configurations
# =========================================================================
def create_lightgbm_model(
    custom_params: Dict[str, Any] = None, random_state: int = 42
) -> lgb.LGBMClassifier:
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 25,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        params.update(custom_params)
    return lgb.LGBMClassifier(**params)


def create_xgboost_model(
    custom_params: Dict[str, Any] = None, random_state: int = 42
) -> xgb.XGBClassifier:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    if custom_params:
        params.update(custom_params)
    return xgb.XGBClassifier(**params)


def create_catboost_model(
    custom_params: Dict[str, Any] = None, random_state: int = 42
) -> cb.CatBoostClassifier:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": random_state,
        "verbose": False,
        "thread_count": -1,
        "early_stopping_rounds": 50,
    }
    if custom_params:
        params.update(custom_params)
    return cb.CatBoostClassifier(**params)


class TabularResBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        logits = self.head(x)
        return logits.squeeze(-1)


def create_tabular_nn_components(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    model = TabularResNet(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=1e-5
    )
    return model, criterion, optimizer, scheduler


# =========================================================================
# 3. Stratified CV Training & Inference Pipeline
# =========================================================================
feature_cols = [
    col
    for col in train_processed.columns
    if col not in [target_col, "fold", "ID", "id"]
]

X_test_raw = test_processed[feature_cols].values
n_train = len(train_processed)
n_test = len(test_processed)
n_splits = train_processed["fold"].nunique()

oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cb = np.zeros(n_train)
oof_nn = np.zeros(n_train)

test_preds_lgb = np.zeros(n_test)
test_preds_xgb = np.zeros(n_test)
test_preds_cb = np.zeros(n_test)
test_preds_nn = np.zeros(n_test)

y_all = train_processed[target_col].values

for fold in range(n_splits):
    train_mask = train_processed["fold"] != fold
    val_mask = train_processed["fold"] == fold

    X_tr = train_processed.loc[train_mask, feature_cols].values
    y_tr = train_processed.loc[train_mask, target_col].values
    X_va = train_processed.loc[val_mask, feature_cols].values
    y_va = train_processed.loc[val_mask, target_col].values
    val_indices = np.where(val_mask)[0]

    # --- LightGBM ---
    lgb_model = create_lightgbm_model(random_state=SEED + fold)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_lgb[val_indices] = lgb_model.predict_proba(X_va)[:, 1]
    test_preds_lgb += lgb_model.predict_proba(X_test_raw)[:, 1] / n_splits

    # --- XGBoost ---
    xgb_model = create_xgboost_model(random_state=SEED + fold)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_indices] = xgb_model.predict_proba(X_va)[:, 1]
    test_preds_xgb += xgb_model.predict_proba(X_test_raw)[:, 1] / n_splits

    # --- CatBoost ---
    cb_model = create_catboost_model(random_state=SEED + fold)
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[val_indices] = cb_model.predict_proba(X_va)[:, 1]
    test_preds_cb += cb_model.predict_proba(X_test_raw)[:, 1] / n_splits

    # --- Deep Tabular ResNet ---
    scaler = QuantileTransformer(
        n_quantiles=min(2000, len(X_tr)),
        output_distribution="normal",
        random_state=SEED + fold,
        subsample=200000,
    )
    X_tr_sc = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0)
    X_va_sc = np.nan_to_num(scaler.transform(X_va), nan=0.0)
    X_te_sc = np.nan_to_num(scaler.transform(X_test_raw), nan=0.0)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )

    nn_model, criterion, optimizer, scheduler = create_tabular_nn_components(
        input_dim=len(feature_cols), lr=1e-3, weight_decay=1e-4, device=device
    )

    best_val_auc = 0.0
    best_nn_val_probs = None
    best_nn_state = None

    for epoch in range(25):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            va_tensor = torch.tensor(X_va_sc, dtype=torch.float32).to(device)
            va_probs = torch.sigmoid(nn_model(va_tensor)).cpu().numpy()
            ep_auc = roc_auc_score(y_va, va_probs)
            if ep_auc > best_val_auc:
                best_val_auc = ep_auc
                best_nn_val_probs = va_probs
                best_nn_state = {k: v.cpu() for k, v in nn_model.state_dict().items()}

    oof_nn[val_indices] = best_nn_val_probs
    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_state.items()})
    nn_model.eval()
    with torch.no_grad():
        te_tensor = torch.tensor(X_te_sc, dtype=torch.float32).to(device)
        test_preds_nn += torch.sigmoid(nn_model(te_tensor)).cpu().numpy() / n_splits


# =========================================================================
# 4. Ensemble Weight Optimization & Submission Generation
# =========================================================================
oof_rank_matrix = np.column_stack([
    rankdata(oof_lgb) / len(oof_lgb),
    rankdata(oof_xgb) / len(oof_xgb),
    rankdata(oof_cb) / len(oof_cb),
    rankdata(oof_nn) / len(oof_nn),
])

test_rank_matrix = np.column_stack([
    rankdata(test_preds_lgb) / len(test_preds_lgb),
    rankdata(test_preds_xgb) / len(test_preds_xgb),
    rankdata(test_preds_cb) / len(test_preds_cb),
    rankdata(test_preds_nn) / len(test_preds_nn),
])


def rank_loss_func(weights):
    w = np.array(weights)
    w = np.maximum(w, 0)
    w_sum = w.sum()
    if w_sum == 0:
        return 0.0
    w = w / w_sum
    blend = oof_rank_matrix @ w
    return -roc_auc_score(y_all, blend)


init_weights = [0.35, 0.25, 0.30, 0.10]
bounds = [(0, 1)] * 4
res = minimize(
    rank_loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0},
)

opt_weights = res.x / np.sum(res.x)

oof_ensemble = oof_rank_matrix @ opt_weights
final_validation_score = float(roc_auc_score(y_all, oof_ensemble))

test_predictions = test_rank_matrix @ opt_weights
test_predictions = np.clip(test_predictions, 0.0, 1.0)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_predictions)), target_col: test_predictions}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_validation_score}")
