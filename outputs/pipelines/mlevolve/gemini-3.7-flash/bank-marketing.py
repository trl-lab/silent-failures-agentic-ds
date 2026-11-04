import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# ------------------------------------------------------------------------------
# 1. Reproducibility & Device Configuration
# ------------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. Data Loading & Preprocessing
# ------------------------------------------------------------------------------
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = (
    "Class"
    if "Class" in train_df.columns
    else ("y" if "y" in train_df.columns else None)
)
if target_col is None:
    raise ValueError("Target column ('Class' or 'y') not found in training dataset.")

y_raw = train_df[target_col]
if (
    y_raw.dtype == object
    or isinstance(y_raw.iloc[0], str)
    or isinstance(y_raw.dtype, pd.CategoricalDtype)
):
    unique_vals = set(y_raw.dropna().astype(str).unique())
    if unique_vals.issubset({"1", "2"}):
        y_train = (y_raw.astype(str) == "2").astype(int)
    elif unique_vals.issubset({"yes", "no"}):
        y_train = (y_raw.astype(str).str.lower() == "yes").astype(int)
    else:
        y_train = (
            y_raw.map({"2": 1, "1": 0, "yes": 1, "no": 0, True: 1, False: 0})
            .fillna(0)
            .astype(int)
        )
elif y_raw.min() == 1 and y_raw.max() == 2:
    y_train = (y_raw == 2).astype(int)
else:
    y_train = y_raw.astype(int)

y_train = pd.Series(y_train).reset_index(drop=True)

drop_cols_train = [c for c in [target_col, "id", "ID"] if c in train_df.columns]
drop_cols_test = [c for c in [target_col, "id", "ID"] if c in test_df.columns]

train_features = train_df.drop(columns=drop_cols_train).reset_index(drop=True)
test_features = test_df.drop(columns=drop_cols_test).reset_index(drop=True)

# Standardize column names from OpenML format (V1..V16) to standard domain attribute names
DOMAIN_COL_NAMES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

v_cols_map = {f"V{i+1}": name for i, name in enumerate(DOMAIN_COL_NAMES)}
v_cols_map.update({f"v{i+1}": name for i, name in enumerate(DOMAIN_COL_NAMES)})

train_features = train_features.rename(columns=v_cols_map)
test_features = test_features.rename(columns=v_cols_map)

if "month" not in train_features.columns and len(train_features.columns) == len(DOMAIN_COL_NAMES):
    train_features.columns = DOMAIN_COL_NAMES
    test_features.columns = DOMAIN_COL_NAMES

# ------------------------------------------------------------------------------
# 3. Domain Feature Engineering
# ------------------------------------------------------------------------------
MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "11": 11,
    "12": 12,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
}


def engineer_base_features(df):
    df = df.copy()

    # Temporal & Month Features
    month_num = df["month"].astype(str).str.lower().map(MONTH_MAP).fillna(6).astype(int)
    df["month_num"] = month_num
    df["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)
    df["quarter"] = ((month_num - 1) // 3) + 1

    day_num = pd.to_numeric(df["day"], errors="coerce").fillna(15)
    df["day_sin"] = np.sin(2 * np.pi * day_num / 31.0)
    df["day_cos"] = np.cos(2 * np.pi * day_num / 31.0)
    df["is_month_start"] = (day_num <= 5).astype(int)
    df["is_month_end"] = (day_num >= 25).astype(int)
    df["is_mid_month"] = ((day_num >= 10) & (day_num <= 20)).astype(int)

    # Call Duration Features
    duration = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    df["duration_log"] = np.log1p(np.maximum(0, duration))
    df["duration_min"] = duration / 60.0
    df["duration_is_zero"] = (duration == 0).astype(int)
    df["duration_over_5min"] = (duration > 300).astype(int)
    df["duration_over_10min"] = (duration > 600).astype(int)

    # Campaign & Contact Intensity
    campaign = pd.to_numeric(df["campaign"], errors="coerce").fillna(1)
    df["duration_per_campaign"] = duration / (campaign + 1e-5)
    df["campaign_log"] = np.log1p(campaign)
    df["is_first_contact_campaign"] = (campaign == 1).astype(int)
    df["is_frequent_campaign"] = (campaign > 3).astype(int)

    # Past Campaign History
    pdays = pd.to_numeric(df["pdays"], errors="coerce").fillna(-1)
    previous = pd.to_numeric(df["previous"], errors="coerce").fillna(0)
    df["was_previously_contacted"] = (
        (pdays != -1) & (pdays != 999) & (previous > 0)
    ).astype(int)
    df["pdays_cleaned"] = np.where((pdays == -1) | (pdays == 999), -1, pdays)
    df["pdays_log"] = np.log1p(np.maximum(0, df["pdays_cleaned"]))
    df["previous_log"] = np.log1p(previous)
    df["campaign_to_previous_ratio"] = campaign / (previous + 1.0)
    df["pdays_per_previous"] = np.where(
        previous > 0, df["pdays_cleaned"] / (previous + 1.0), -1
    )

    poutcome_str = df["poutcome"].astype(str).str.lower()
    df["prev_campaign_success"] = (poutcome_str == "success").astype(int)
    df["prev_campaign_failure"] = (poutcome_str == "failure").astype(int)
    df["prev_campaign_unknown"] = (poutcome_str.isin(["unknown", "other"])).astype(int)

    # Client Financial Profile & Demographics
    age = pd.to_numeric(df["age"], errors="coerce").fillna(40)
    balance = pd.to_numeric(df["balance"], errors="coerce").fillna(0)

    df["age_group"] = pd.cut(
        age, bins=[0, 25, 35, 50, 65, 120], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    df["is_senior"] = (age >= 60).astype(int)
    df["is_young"] = (age < 30).astype(int)

    job_str = df["job"].astype(str).str.lower()
    df["is_retired_or_student"] = job_str.isin(["retired", "student"]).astype(int)
    df["is_unemployed"] = (job_str == "unemployed").astype(int)
    df["is_white_collar"] = job_str.isin(["management", "admin.", "technician"]).astype(
        int
    )

    housing_str = df["housing"].astype(str).str.lower()
    loan_str = df["loan"].astype(str).str.lower()
    default_str = df["default"].astype(str).str.lower()

    has_housing = housing_str.isin(["yes", "1", "true", "y", "2"]).astype(int)
    has_loan = loan_str.isin(["yes", "1", "true", "y", "2"]).astype(int)
    has_default = default_str.isin(["yes", "1", "true", "y", "2"]).astype(int)

    df["has_housing"] = has_housing
    df["has_loan"] = has_loan
    df["has_default"] = has_default
    df["total_loans"] = has_housing + has_loan
    df["has_any_loan"] = ((has_housing == 1) | (has_loan == 1)).astype(int)
    df["has_both_loans"] = ((has_housing == 1) & (has_loan == 1)).astype(int)
    df["is_debt_free"] = (
        (has_housing == 0) & (has_loan == 0) & (has_default == 0)
    ).astype(int)

    df["balance_sign"] = np.sign(balance)
    df["is_balance_negative"] = (balance < 0).astype(int)
    df["is_balance_zero"] = (balance == 0).astype(int)
    df["balance_log_pos"] = np.log1p(np.maximum(0, balance))
    df["balance_log_neg"] = np.log1p(np.maximum(0, -balance))
    df["balance_to_age"] = balance / (age + 1.0)
    df["balance_per_campaign"] = balance / (campaign + 1.0)

    # Categorical Interactions / Crosses
    education_str = df["education"].astype(str).str.lower()
    marital_str = df["marital"].astype(str).str.lower()
    contact_str = df["contact"].astype(str).str.lower()

    df["job_education"] = job_str + "__" + education_str
    df["housing_loan"] = housing_str + "__" + loan_str
    df["marital_education"] = marital_str + "__" + education_str
    df["month_contact"] = df["month"].astype(str).str.lower() + "__" + contact_str
    df["job_marital"] = job_str + "__" + marital_str
    df["poutcome_contact"] = poutcome_str + "__" + contact_str

    return df


X_train_base = engineer_base_features(train_features)
X_test_base = engineer_base_features(test_features)


# ------------------------------------------------------------------------------
# 4. Group Aggregations (Strictly Computed on Train)
# ------------------------------------------------------------------------------
def compute_group_aggregations(train_df, test_df):
    train_out = train_df.copy()
    test_out = test_df.copy()

    agg_specs = [
        ("job", "balance", ["mean", "median", "std"]),
        ("job", "duration", ["mean", "median", "std"]),
        ("education", "balance", ["mean", "std"]),
        ("age_group", "balance", ["mean", "std"]),
        ("month_num", "duration", ["mean", "median"]),
        ("housing_loan", "balance", ["mean", "median"]),
    ]

    for group_col, num_col, aggs in agg_specs:
        grouped = train_df.groupby(group_col)[num_col].agg(aggs)
        grouped.columns = [f"{group_col}_{num_col}_{agg}" for agg in aggs]
        grouped = grouped.reset_index()

        train_out = train_out.merge(grouped, on=group_col, how="left")
        test_out = test_out.merge(grouped, on=group_col, how="left")

        mean_col = f"{group_col}_{num_col}_mean"
        if mean_col in train_out.columns:
            diff_col_name = f"{num_col}_diff_from_{group_col}_mean"
            train_out[diff_col_name] = train_out[num_col] - train_out[mean_col]
            test_out[diff_col_name] = test_out[num_col] - test_out[mean_col]

    return train_out, test_out


X_train_agg, X_test_agg = compute_group_aggregations(X_train_base, X_test_base)

# ------------------------------------------------------------------------------
# 5. Categorical Encoding & Leak-Free Target Encoding
# ------------------------------------------------------------------------------
categorical_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "job_education",
    "housing_loan",
    "marital_education",
    "month_contact",
    "job_marital",
    "poutcome_contact",
]

for col in categorical_cols:
    freq = X_train_agg[col].astype(str).value_counts(normalize=True).to_dict()
    X_train_agg[f"{col}_freq"] = (
        X_train_agg[col].astype(str).map(freq).fillna(0.0).astype(float)
    )
    X_test_agg[f"{col}_freq"] = (
        X_test_agg[col].astype(str).map(freq).fillna(0.0).astype(float)
    )

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat(
        [X_train_agg[col].astype(str), X_test_agg[col].astype(str)], axis=0
    ).unique()
    le.fit(combined)
    X_train_agg[col] = le.transform(X_train_agg[col].astype(str))
    X_test_agg[col] = le.transform(X_test_agg[col].astype(str))

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = np.zeros(len(y_train), dtype=int)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_agg, y_train)):
    folds[val_idx] = fold_idx

X_train_final = X_train_agg.reset_index(drop=True)
X_test_final = X_test_agg.reset_index(drop=True)

target_enc_cols = [
    "job",
    "education",
    "marital",
    "month",
    "poutcome",
    "job_education",
    "housing_loan",
]
global_target_mean = float(y_train.mean())
smoothing_m = 10.0

for col in target_enc_cols:
    oof_target_enc = np.zeros(len(X_train_final), dtype=float)
    for fold_idx in range(N_SPLITS):
        tr_idx = np.where(folds != fold_idx)[0]
        va_idx = np.where(folds == fold_idx)[0]

        train_cat = X_train_final.iloc[tr_idx][col]
        train_y = y_train.iloc[tr_idx]

        stats = (
            pd.DataFrame({"cat": train_cat, "y": train_y})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        counts = stats["count"]
        means = stats["mean"]
        smoothed_vals = (counts * means + smoothing_m * global_target_mean) / (
            counts + smoothing_m
        )
        enc_map = smoothed_vals.to_dict()

        val_cat = X_train_final.iloc[va_idx][col]
        oof_target_enc[va_idx] = val_cat.map(enc_map).fillna(global_target_mean).values

    X_train_final[f"{col}_target_enc"] = oof_target_enc

    stats_all = (
        pd.DataFrame({"cat": X_train_final[col], "y": y_train})
        .groupby("cat")["y"]
        .agg(["count", "mean"])
    )
    counts_all = stats_all["count"]
    means_all = stats_all["mean"]
    smoothed_vals_all = (counts_all * means_all + smoothing_m * global_target_mean) / (
        counts_all + smoothing_m
    )
    test_enc_map = smoothed_vals_all.to_dict()
    X_test_final[f"{col}_target_enc"] = (
        X_test_final[col].map(test_enc_map).fillna(global_target_mean).values
    )

for col in X_train_final.columns:
    if X_train_final[col].dtype == object:
        X_train_final[col] = pd.to_numeric(X_train_final[col], errors="coerce")
        X_test_final[col] = pd.to_numeric(X_test_final[col], errors="coerce")

    col_median = X_train_final[col].median()
    if pd.isna(col_median):
        col_median = 0.0
    X_train_final[col] = X_train_final[col].fillna(col_median)
    X_test_final[col] = X_test_final[col].fillna(col_median)


# ------------------------------------------------------------------------------
# 6. Model Architectures & Factory
# ------------------------------------------------------------------------------
class TabularResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out = self.dropout2(out)
        return self.activation(out + residual)


class TabularResNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(num_features)
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        layers = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(
                TabularResidualBlock(
                    in_features=in_dim,
                    out_features=out_dim,
                    dropout_rate=dropout_rate,
                )
            )
        self.res_blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_layer(x)
        x = self.res_blocks(x)
        logits = self.head(x)
        return logits.squeeze(-1)


def build_pytorch_criterion():
    return nn.BCEWithLogitsLoss()


def build_pytorch_optimizer(
    model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4
):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_pytorch_scheduler(
    optimizer: optim.Optimizer, num_epochs: int, eta_min: float = 1e-6
):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=eta_min
    )


class ModelFactory:
    @staticmethod
    def create_neural_network(
        num_features: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        model = TabularResNet(
            num_features=num_features,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )
        criterion = build_pytorch_criterion()
        optimizer = build_pytorch_optimizer(model, lr=lr, weight_decay=weight_decay)
        return model, criterion, optimizer

    @staticmethod
    def create_lgb_classifier(random_state: int = 42, **kwargs):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)

    @staticmethod
    def create_xgb_classifier(random_state: int = 42, **kwargs):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_estimators": 1500,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return xgb.XGBClassifier(**params)

    @staticmethod
    def create_catboost_classifier(random_state: int = 42, **kwargs):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": random_state,
            "iterations": 1500,
            "verbose": False,
            "thread_count": -1,
        }
        params.update(kwargs)
        return CatBoostClassifier(**params)


# ------------------------------------------------------------------------------
# 7. Cross-Validation Training Loop
# ------------------------------------------------------------------------------
X_tr_mat = X_train_final.values
X_te_mat = X_test_final.values
y_true = np.array(y_train).ravel().astype(int)
n_samples = len(X_tr_mat)
n_test_samples = len(X_te_mat)

oof_lgb = np.zeros(n_samples, dtype=float)
oof_xgb = np.zeros(n_samples, dtype=float)
oof_cat = np.zeros(n_samples, dtype=float)
oof_nn = np.zeros(n_samples, dtype=float)

test_lgb = np.zeros(n_test_samples, dtype=float)
test_xgb = np.zeros(n_test_samples, dtype=float)
test_cat = np.zeros(n_test_samples, dtype=float)
test_nn = np.zeros(n_test_samples, dtype=float)

for fold in range(N_SPLITS):
    idx_train = np.where(folds != fold)[0]
    idx_val = np.where(folds == fold)[0]

    X_fold_train, y_fold_train = X_tr_mat[idx_train], y_true[idx_train]
    X_fold_val, y_fold_val = X_tr_mat[idx_val], y_true[idx_val]

    # --- 1. LightGBM ---
    lgb_model = ModelFactory.create_lgb_classifier(random_state=42 + fold)
    lgb_model.fit(
        X_fold_train,
        y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[idx_val] = lgb_model.predict_proba(X_fold_val)[:, 1]
    test_lgb += lgb_model.predict_proba(X_te_mat)[:, 1] / N_SPLITS

    # --- 2. XGBoost ---
    xgb_model = ModelFactory.create_xgb_classifier(
        random_state=42 + fold, early_stopping_rounds=50
    )
    xgb_model.fit(
        X_fold_train,
        y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False,
    )
    oof_xgb[idx_val] = xgb_model.predict_proba(X_fold_val)[:, 1]
    test_xgb += xgb_model.predict_proba(X_te_mat)[:, 1] / N_SPLITS

    # --- 3. CatBoost ---
    cat_model = ModelFactory.create_catboost_classifier(
        random_state=42 + fold, early_stopping_rounds=50
    )
    cat_model.fit(
        X_fold_train,
        y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        verbose=False,
    )
    oof_cat[idx_val] = cat_model.predict_proba(X_fold_val)[:, 1]
    test_cat += cat_model.predict_proba(X_te_mat)[:, 1] / N_SPLITS

    # --- 4. PyTorch Tabular ResNet ---
    scaler = StandardScaler()
    X_nn_train = scaler.fit_transform(X_fold_train)
    X_nn_val = scaler.transform(X_fold_val)
    X_nn_test = scaler.transform(X_te_mat)

    train_tensor_x = torch.tensor(X_nn_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_fold_train, dtype=torch.float32)
    val_tensor_x = torch.tensor(X_nn_val, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_fold_val, dtype=torch.float32)
    test_tensor_x = torch.tensor(X_nn_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor_x, train_tensor_y),
        batch_size=256,
        shuffle=True,
        drop_last=False,
    )

    nn_model, criterion, optimizer = ModelFactory.create_neural_network(
        num_features=X_nn_train.shape[1],
        hidden_dims=[256, 128, 64],
        dropout_rate=0.25,
        lr=1e-3,
        weight_decay=1e-4,
    )
    nn_model = nn_model.to(device)
    criterion = criterion.to(device)
    scheduler = build_pytorch_scheduler(optimizer, num_epochs=35)

    best_val_auc = -1.0
    best_val_preds = None
    best_test_preds = None
    patience = 8
    patience_counter = 0

    for epoch in range(35):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            out_logits = nn_model(batch_x)
            loss = criterion(out_logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_logits = nn_model(val_tensor_x.to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            epoch_val_auc = roc_auc_score(y_fold_val, val_probs)

            if epoch_val_auc > best_val_auc:
                best_val_auc = epoch_val_auc
                best_val_preds = val_probs
                test_logits = nn_model(test_tensor_x.to(device))
                best_test_preds = torch.sigmoid(test_logits).cpu().numpy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    oof_nn[idx_val] = best_val_preds
    test_nn += best_test_preds / N_SPLITS

    fold_lgb_auc = roc_auc_score(y_fold_val, oof_lgb[idx_val])
    fold_xgb_auc = roc_auc_score(y_fold_val, oof_xgb[idx_val])
    fold_cat_auc = roc_auc_score(y_fold_val, oof_cat[idx_val])
    fold_nn_auc = roc_auc_score(y_fold_val, oof_nn[idx_val])
    print(
        f"Fold {fold+1}/{N_SPLITS} - LGB: {fold_lgb_auc:.5f} | XGB: {fold_xgb_auc:.5f} | CAT: {fold_cat_auc:.5f} | NN: {fold_nn_auc:.5f}"
    )

# ------------------------------------------------------------------------------
# 8. Out-Of-Fold Evaluation & Optimal Ensemble Blending
# ------------------------------------------------------------------------------
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn])


def blend_loss(weights):
    w = np.maximum(np.array(weights), 0.0)
    w_sum = np.sum(w)
    if w_sum == 0:
        return 1.0
    w_norm = w / w_sum
    pred_blend = np.dot(oof_matrix, w_norm)
    return -roc_auc_score(y_true, pred_blend)


initial_weights = [0.35, 0.35, 0.20, 0.10]
bounds = [(0.0, 1.0)] * 4
opt_result = minimize(
    blend_loss,
    initial_weights,
    method="Powell",
    bounds=bounds,
    options={"maxiter": 200},
)

opt_weights = np.maximum(opt_result.x, 0.0)
opt_weights = opt_weights / np.sum(opt_weights)
print(
    f"Ensemble Weights: LGB={opt_weights[0]:.3f}, XGB={opt_weights[1]:.3f}, CAT={opt_weights[2]:.3f}, NN={opt_weights[3]:.3f}"
)

final_oof_preds = np.dot(oof_matrix, opt_weights)
final_test_preds = np.dot(test_matrix, opt_weights)
final_score = roc_auc_score(y_true, final_oof_preds)

# ------------------------------------------------------------------------------
# 9. Submission Output
# ------------------------------------------------------------------------------
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds), dtype=int), "Class": final_test_preds}
)
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_score}")
