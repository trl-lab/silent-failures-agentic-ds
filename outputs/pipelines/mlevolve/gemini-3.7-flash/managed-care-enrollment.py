import math
import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 1. Configuration and Paths
# --------------------------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train.parquet")
TEST_PATH = os.path.join(INPUT_DIR, "test.parquet")
TARGET_COL = "Total Enrollment"
N_SPLITS = 10
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# --------------------------------------------------------------------------
# 2. Data Cleaning & Feature Engineering
# --------------------------------------------------------------------------
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

y_train = train_df[TARGET_COL].copy().astype(np.float32)
train_df = train_df.drop(columns=[TARGET_COL])

train_df["is_train"] = 1
test_df["is_train"] = 0
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Clean string columns
for col in combined.columns:
    if col == "is_train":
        continue
    if combined[col].dtype == "object":
        combined[col] = combined[col].astype(str).str.strip()
        combined[col] = combined[col].replace(
            ["nan", "None", "NULL", "N/A", ""], np.nan
        )

# Temporal decomposition & cyclical calendar encodings
date_cols = []
for col in combined.columns:
    if col == "is_train":
        continue
    if any(kw in col.lower() for kw in ["date", "time", "period", "month", "year"]):
        try:
            sample_non_null = combined[col].dropna().iloc[:100]
            if len(sample_non_null) > 0:
                parsed = pd.to_datetime(sample_non_null, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    date_cols.append(col)
        except Exception:
            pass

for col in date_cols:
    dt_series = pd.to_datetime(combined[col], errors="coerce")
    combined[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(int)
    combined[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
    combined[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(int)
    combined[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(int)
    combined[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(int)
    combined[f"{col}_is_month_end"] = dt_series.dt.is_month_end.astype(int)

    month_valid = dt_series.dt.month.fillna(1)
    combined[f"{col}_sin_month"] = np.sin(2 * np.pi * month_valid / 12.0)
    combined[f"{col}_cos_month"] = np.cos(2 * np.pi * month_valid / 12.0)

    valid_years = combined[f"{col}_year"][combined[f"{col}_year"] > 0]
    min_year = valid_years.min() if len(valid_years) > 0 else 2000
    combined[f"{col}_elapsed_months"] = np.where(
        combined[f"{col}_year"] > 0,
        (combined[f"{col}_year"] - min_year) * 12 + dt_series.dt.month.fillna(1),
        -1,
    )

categorical_cols = [
    col
    for col in combined.columns
    if col not in ["is_train"] + date_cols
    and (combined[col].dtype == "object" or combined[col].dtype.name == "category")
]
numerical_cols = [
    col
    for col in combined.columns
    if col not in ["is_train"] + date_cols + categorical_cols
    and pd.api.types.is_numeric_dtype(combined[col])
]

for col in categorical_cols + numerical_cols:
    if combined[col].isna().sum() > 0:
        combined[f"{col}_isna"] = combined[col].isna().astype(int)

train_mask = combined["is_train"] == 1
for col in categorical_cols:
    freq_map = combined.loc[train_mask, col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq_map).fillna(0.0).astype(np.float32)

if len(categorical_cols) >= 2:
    top_cats = sorted(categorical_cols, key=lambda c: combined[c].nunique())[:4]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            inter_col = f"{c1}_x_{c2}"
            combined[inter_col] = (
                combined[c1].astype(str) + "_" + combined[c2].astype(str)
            )
            categorical_cols.append(inter_col)
            inter_freq = (
                combined.loc[train_mask, inter_col]
                .value_counts(normalize=True)
                .to_dict()
            )
            combined[f"{inter_col}_freq"] = (
                combined[inter_col].map(inter_freq).fillna(0.0).astype(np.float32)
            )

for cat_col in categorical_cols[:6]:
    group_counts = combined.groupby(cat_col)[cat_col].transform("count")
    combined[f"{cat_col}_group_count"] = group_counts.astype(np.float32)

for col in categorical_cols:
    combined[f"{col}_strlen"] = combined[col].astype(str).str.len().astype(np.float32)
    combined[f"{col}_word_count"] = (
        combined[col].astype(str).apply(lambda x: len(x.split())).astype(np.float32)
    )

# Hierarchical and multi-key composite interaction group aggregations
base_agg_num_cols = [
    col
    for col in numerical_cols
    if not col.endswith("_isna") and not col.endswith("_freq")
]
date_num_cols = [
    f"{col}_elapsed_months"
    for col in date_cols
    if f"{col}_elapsed_months" in combined.columns
] + [f"{col}_year" for col in date_cols if f"{col}_year" in combined.columns]
agg_targets = list(dict.fromkeys(base_agg_num_cols + date_num_cols))[:6]
single_key_cats = [c for c in categorical_cols if "_x_" not in c][:4]
composite_key_cats = [c for c in categorical_cols if "_x_" in c][:4]
all_agg_cats = single_key_cats + composite_key_cats

for cat_col in all_agg_cats:
    for num_col in agg_targets:
        if num_col in combined.columns:
            grp = combined.groupby(cat_col)[num_col]
            grp_mean = grp.transform("mean").astype(np.float32)
            grp_std = grp.transform("std").fillna(0.0).astype(np.float32)
            grp_min = grp.transform("min").astype(np.float32)
            grp_max = grp.transform("max").astype(np.float32)

            combined[f"{num_col}_mean_by_{cat_col}"] = grp_mean
            combined[f"{num_col}_std_by_{cat_col}"] = grp_std
            combined[f"{num_col}_min_by_{cat_col}"] = grp_min
            combined[f"{num_col}_max_by_{cat_col}"] = grp_max
            combined[f"{num_col}_diff_mean_by_{cat_col}"] = (
                combined[num_col] - grp_mean
            ).astype(np.float32)
            combined[f"{num_col}_ratio_mean_by_{cat_col}"] = (
                combined[num_col] / (grp_mean.abs() + 1.0)
            ).astype(np.float32)

for col in categorical_cols:
    le = LabelEncoder()
    str_vals = combined[col].fillna("__MISSING__").astype(str)
    combined[f"{col}_le"] = le.fit_transform(str_vals).astype(np.int32)

# Impute all numerical features
all_num_cols = [
    col
    for col in combined.columns
    if col not in ["is_train"] + date_cols + categorical_cols
    and pd.api.types.is_numeric_dtype(combined[col])
]
for col in all_num_cols:
    train_median = combined.loc[train_mask, col].median()
    if pd.isna(train_median):
        train_median = 0.0
    combined[col] = (
        combined[col].replace([np.inf, -np.inf], np.nan).fillna(train_median)
    )

train_processed = (
    combined[combined["is_train"] == 1]
    .drop(columns=["is_train"])
    .reset_index(drop=True)
)
test_processed = (
    combined[combined["is_train"] == 0]
    .drop(columns=["is_train"])
    .reset_index(drop=True)
)
train_processed[TARGET_COL] = y_train.values

# --------------------------------------------------------------------------
# 3. Stratified Cross-Validation Splitting
# --------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
target_quantiles = pd.qcut(
    train_processed[TARGET_COL],
    q=min(10, len(np.unique(train_processed[TARGET_COL]))),
    labels=False,
    duplicates="drop",
)

train_processed["fold"] = -1
for fold_idx, (train_idx, val_idx) in enumerate(
    skf.split(train_processed, target_quantiles)
):
    train_processed.loc[val_idx, "fold"] = fold_idx


# --------------------------------------------------------------------------
# 4. Model Architectures & Builders
# --------------------------------------------------------------------------
class TabularResNet(nn.Module):
    def __init__(
        self,
        cat_cardinalities,
        num_continuous_features,
        embedding_dim_factor=0.5,
        hidden_dims=(256, 128, 64),
        dropout_rate=0.2,
        output_dim=1,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for card in cat_cardinalities:
            emb_dim = max(
                4, min(50, int(math.ceil(math.pow(card, embedding_dim_factor))))
            )
            self.embeddings.append(nn.Embedding(card + 1, emb_dim))
            total_emb_dim += emb_dim

        self.has_categorical = len(cat_cardinalities) > 0
        self.has_continuous = num_continuous_features > 0

        if self.has_continuous:
            self.num_bn = nn.BatchNorm1d(num_continuous_features)
            in_dim = total_emb_dim + num_continuous_features
        else:
            in_dim = total_emb_dim

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            dim_in = hidden_dims[i]
            dim_out = hidden_dims[i + 1]
            block = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.LayerNorm(dim_out),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(dim_out, dim_out),
                nn.LayerNorm(dim_out),
            )
            shortcut = (
                nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
            )
            self.blocks.append(nn.ModuleDict({"block": block, "shortcut": shortcut}))

        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dims[-1], output_dim),
        )

    def forward(self, x_num=None, x_cat=None):
        components = []
        if self.has_categorical and x_cat is not None:
            emb_outs = [
                emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)
            ]
            components.append(torch.cat(emb_outs, dim=1))
        if self.has_continuous and x_num is not None:
            components.append(self.num_bn(x_num))

        x = torch.cat(components, dim=1) if len(components) > 1 else components[0]
        x = self.input_layer(x)
        for block_dict in self.blocks:
            residual = block_dict["shortcut"](x)
            x = block_dict["block"](x)
            x = torch.relu(x + residual)
        return self.head(x).squeeze(-1)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def build_lightgbm_regressor(random_state=42):
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgboost_regressor(random_state=42):
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def build_catboost_regressor(cat_features=None, random_state=42):
    return cb.CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        cat_features=cat_features,
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
        early_stopping_rounds=50,
    )


def build_extratrees_regressor(random_state=42):
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=random_state,
    )


def build_dart_regressor(random_state=42):
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="dart",
        n_estimators=800,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        drop_rate=0.1,
        skip_drop=0.5,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_ridge_regressor(random_state=42):
    return Ridge(alpha=10.0, random_state=random_state)


def compute_oof_target_encodings(
    X_train_df,
    y_train_series,
    X_val_df,
    X_test_df,
    te_columns,
    n_inner_splits=5,
    smoothings=(5.0, 25.0),
    random_state=42,
):
    global_mean = float(y_train_series.mean())
    global_std = float(y_train_series.std()) if len(y_train_series) > 1 else 0.0
    global_min = float(y_train_series.min())
    global_max = float(y_train_series.max())

    X_tr_te = pd.DataFrame(index=X_train_df.index)
    X_val_te = pd.DataFrame(index=X_val_df.index)
    X_te_te = pd.DataFrame(index=X_test_df.index)

    inner_kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)

    for c in te_columns:
        if c not in X_train_df.columns:
            continue

        grp_full = y_train_series.groupby(X_train_df[c])
        cnt_full = grp_full.count()
        grp_m_full = grp_full.mean()
        grp_s_full = grp_full.std().fillna(0.0)
        grp_mi_full = grp_full.min()
        grp_ma_full = grp_full.max()

        min_tr = np.full(len(X_train_df), global_min, dtype=np.float32)
        max_tr = np.full(len(X_train_df), global_max, dtype=np.float32)

        for in_tr_idx, in_val_idx in inner_kf.split(X_train_df):
            in_tr_cat = X_train_df.iloc[in_tr_idx][c]
            in_tr_y = y_train_series.iloc[in_tr_idx]

            grp = in_tr_y.groupby(in_tr_cat)
            s_min = grp.min().to_dict()
            s_max = grp.max().to_dict()

            in_val_cats = X_train_df.iloc[in_val_idx][c]
            min_tr[in_val_idx] = in_val_cats.map(s_min).fillna(global_min).values
            max_tr[in_val_idx] = in_val_cats.map(s_max).fillna(global_max).values

        X_tr_te[f"{c}_te_min"] = min_tr
        X_tr_te[f"{c}_te_max"] = max_tr
        X_val_te[f"{c}_te_min"] = X_val_df[c].map(grp_mi_full.to_dict()).fillna(global_min).astype(np.float32).values
        X_val_te[f"{c}_te_max"] = X_val_df[c].map(grp_ma_full.to_dict()).fillna(global_max).astype(np.float32).values
        X_te_te[f"{c}_te_min"] = X_test_df[c].map(grp_mi_full.to_dict()).fillna(global_min).astype(np.float32).values
        X_te_te[f"{c}_te_max"] = X_test_df[c].map(grp_ma_full.to_dict()).fillna(global_max).astype(np.float32).values

        for s in smoothings:
            s_tag = int(s)
            mean_tr = np.full(len(X_train_df), global_mean, dtype=np.float32)
            std_tr = np.full(len(X_train_df), global_std, dtype=np.float32)

            for in_tr_idx, in_val_idx in inner_kf.split(X_train_df):
                in_tr_cat = X_train_df.iloc[in_tr_idx][c]
                in_tr_y = y_train_series.iloc[in_tr_idx]

                grp = in_tr_y.groupby(in_tr_cat)
                cnt = grp.count()
                grp_m = grp.mean()
                grp_s = grp.std().fillna(0.0)

                s_mean = ((cnt * grp_m + s * global_mean) / (cnt + s)).to_dict()
                s_std = ((cnt * grp_s + s * global_std) / (cnt + s)).to_dict()

                in_val_cats = X_train_df.iloc[in_val_idx][c]
                mean_tr[in_val_idx] = in_val_cats.map(s_mean).fillna(global_mean).values
                std_tr[in_val_idx] = in_val_cats.map(s_std).fillna(global_std).values

            X_tr_te[f"{c}_te_mean_s{s_tag}"] = mean_tr
            X_tr_te[f"{c}_te_std_s{s_tag}"] = std_tr

            full_s_mean = ((cnt_full * grp_m_full + s * global_mean) / (cnt_full + s)).to_dict()
            full_s_std = ((cnt_full * grp_s_full + s * global_std) / (cnt_full + s)).to_dict()

            X_val_te[f"{c}_te_mean_s{s_tag}"] = X_val_df[c].map(full_s_mean).fillna(global_mean).astype(np.float32).values
            X_val_te[f"{c}_te_std_s{s_tag}"] = X_val_df[c].map(full_s_std).fillna(global_std).astype(np.float32).values

            X_te_te[f"{c}_te_mean_s{s_tag}"] = X_test_df[c].map(full_s_mean).fillna(global_mean).astype(np.float32).values
            X_te_te[f"{c}_te_std_s{s_tag}"] = X_test_df[c].map(full_s_std).fillna(global_std).astype(np.float32).values

    return X_tr_te, X_val_te, X_te_te


# --------------------------------------------------------------------------
# 5. Training, Out-Of-Fold Evaluation, and Blending
# --------------------------------------------------------------------------
base_feature_cols = [
    col
    for col in train_processed.columns
    if col not in [TARGET_COL, "fold"]
    and pd.api.types.is_numeric_dtype(train_processed[col])
]
cat_cols = [col for col in base_feature_cols if col.endswith("_le")]
te_candidate_cols = cat_cols[:8]

folds = sorted(train_processed["fold"].unique())
n_splits = len(folds)
n_train = len(train_processed)
n_test = len(test_processed)

cat_cardinalities = [
    int(max(train_processed[c].max(), test_processed[c].max()) + 1) for c in cat_cols
]

oof_lgb = np.zeros(n_train, dtype=np.float32)
oof_xgb = np.zeros(n_train, dtype=np.float32)
oof_cat = np.zeros(n_train, dtype=np.float32)
oof_et = np.zeros(n_train, dtype=np.float32)
oof_dart = np.zeros(n_train, dtype=np.float32)
oof_ridge = np.zeros(n_train, dtype=np.float32)
oof_nn = np.zeros(n_train, dtype=np.float32)

test_lgb = np.zeros(n_test, dtype=np.float32)
test_xgb = np.zeros(n_test, dtype=np.float32)
test_cat = np.zeros(n_test, dtype=np.float32)
test_et = np.zeros(n_test, dtype=np.float32)
test_dart = np.zeros(n_test, dtype=np.float32)
test_ridge = np.zeros(n_test, dtype=np.float32)
test_nn = np.zeros(n_test, dtype=np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
y_all = train_processed[TARGET_COL].values.astype(np.float32)

for fold in folds:
    tr_idx = train_processed["fold"] != fold
    val_idx = train_processed["fold"] == fold

    X_tr_base = train_processed.loc[tr_idx, base_feature_cols].copy()
    y_tr = pd.Series(y_all[tr_idx], index=X_tr_base.index)
    X_val_base = train_processed.loc[val_idx, base_feature_cols].copy()
    y_val = y_all[val_idx]
    X_te_base = test_processed[base_feature_cols].copy()

    # Leak-free out-of-fold target encodings per fold
    X_tr_te, X_val_te, X_te_te = compute_oof_target_encodings(
        X_tr_base,
        y_tr,
        X_val_base,
        X_te_base,
        te_candidate_cols,
        n_inner_splits=5,
        smoothings=(5.0, 25.0),
        random_state=RANDOM_STATE + fold,
    )

    X_tr = pd.concat([X_tr_base, X_tr_te], axis=1)
    X_val = pd.concat([X_val_base, X_val_te], axis=1)
    X_te = pd.concat([X_te_base, X_te_te], axis=1)
    y_tr_arr = y_tr.values

    # 1. LightGBM with native categorical features
    lgb_reg = build_lightgbm_regressor(random_state=RANDOM_STATE + fold)
    lgb_reg.fit(
        X_tr,
        y_tr_arr,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = lgb_reg.predict(X_val)
    test_lgb += lgb_reg.predict(X_te) / n_splits

    # 2. XGBoost
    xgb_reg = build_xgboost_regressor(random_state=RANDOM_STATE + fold)
    xgb_reg.fit(X_tr, y_tr_arr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_reg.predict(X_val)
    test_xgb += xgb_reg.predict(X_te) / n_splits

    # 3. CatBoost with native categorical handling
    cb_reg = build_catboost_regressor(cat_features=cat_cols, random_state=RANDOM_STATE + fold)
    cb_reg.fit(X_tr, y_tr_arr, eval_set=(X_val, y_val), verbose=False)
    oof_cat[val_idx] = cb_reg.predict(X_val)
    test_cat += cb_reg.predict(X_te) / n_splits

    # 4. ExtraTrees Regressor
    et_reg = build_extratrees_regressor(random_state=RANDOM_STATE + fold)
    et_reg.fit(X_tr, y_tr_arr)
    oof_et[val_idx] = et_reg.predict(X_val)
    test_et += et_reg.predict(X_te) / n_splits

    # 5. LightGBM DART booster
    dart_reg = build_dart_regressor(random_state=RANDOM_STATE + fold)
    dart_reg.fit(
        X_tr,
        y_tr_arr,
        categorical_feature=cat_cols,
    )
    oof_dart[val_idx] = dart_reg.predict(X_val)
    test_dart += dart_reg.predict(X_te) / n_splits

    # 6. Ridge Regressor
    num_cols_ridge = [c for c in X_tr.columns if c not in cat_cols]
    scaler_ridge = StandardScaler()
    X_tr_ridge = scaler_ridge.fit_transform(X_tr[num_cols_ridge].values)
    X_val_ridge = scaler_ridge.transform(X_val[num_cols_ridge].values)
    X_te_ridge = scaler_ridge.transform(X_te[num_cols_ridge].values)
    ridge_reg = build_ridge_regressor(random_state=RANDOM_STATE + fold)
    ridge_reg.fit(X_tr_ridge, y_tr_arr)
    oof_ridge[val_idx] = ridge_reg.predict(X_val_ridge)
    test_ridge += ridge_reg.predict(X_te_ridge) / n_splits

    # 7. PyTorch TabularResNet
    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    scaler = StandardScaler()
    X_tr_num_scaled = scaler.fit_transform(X_tr[num_cols].values)
    X_val_num_scaled = scaler.transform(X_val[num_cols].values)
    X_te_num_scaled = scaler.transform(X_te[num_cols].values)

    y_scaler = StandardScaler()
    y_tr_scaled = y_scaler.fit_transform(y_tr_arr.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    X_tr_num_t = torch.tensor(X_tr_num_scaled, dtype=torch.float32)
    X_val_num_t = torch.tensor(X_val_num_scaled, dtype=torch.float32)
    X_te_num_t = torch.tensor(X_te_num_scaled, dtype=torch.float32)

    X_tr_cat_t = torch.tensor(X_tr[cat_cols].values, dtype=torch.long)
    X_val_cat_t = torch.tensor(X_val[cat_cols].values, dtype=torch.long)
    X_te_cat_t = torch.tensor(X_te[cat_cols].values, dtype=torch.long)

    y_tr_t = torch.tensor(y_tr_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32)

    train_ds = TensorDataset(X_tr_num_t, X_tr_cat_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)

    nn_model = TabularResNet(
        cat_cardinalities=cat_cardinalities,
        num_continuous_features=len(num_cols),
        hidden_dims=(256, 128, 64),
        dropout_rate=0.15,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-5)

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(45):
        nn_model.train()
        for batch_num, batch_cat, batch_y in train_loader:
            batch_num, batch_cat, batch_y = (
                batch_num.to(device),
                batch_cat.to(device),
                batch_y.to(device),
            )
            optimizer.zero_grad()
            preds = nn_model(batch_num, batch_cat)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_preds_nn_t = nn_model(
                X_val_num_t.to(device), X_val_cat_t.to(device)
            ).cpu()
            val_loss = criterion(val_preds_nn_t, y_val_t).item()
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_weights = {
                    k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                }

    if best_nn_weights is not None:
        nn_model.load_state_dict(best_nn_weights)

    nn_model.eval()
    with torch.no_grad():
        val_scaled_preds = (
            nn_model(X_val_num_t.to(device), X_val_cat_t.to(device)).cpu().numpy()
        )
        oof_nn[val_idx] = y_scaler.inverse_transform(
            val_scaled_preds.reshape(-1, 1)
        ).ravel()
        test_scaled_preds = (
            nn_model(X_te_num_t.to(device), X_te_cat_t.to(device)).cpu().numpy()
        )
        test_nn += (
            y_scaler.inverse_transform(test_scaled_preds.reshape(-1, 1)).ravel()
            / n_splits
        )

# Clean and guard OOF and test predictions against NaNs
oof_lgb = np.nan_to_num(oof_lgb, nan=np.nanmedian(y_all))
oof_xgb = np.nan_to_num(oof_xgb, nan=np.nanmedian(y_all))
oof_cat = np.nan_to_num(oof_cat, nan=np.nanmedian(y_all))
oof_et = np.nan_to_num(oof_et, nan=np.nanmedian(y_all))
oof_dart = np.nan_to_num(oof_dart, nan=np.nanmedian(y_all))
oof_ridge = np.nan_to_num(oof_ridge, nan=np.nanmedian(y_all))
oof_nn = np.nan_to_num(oof_nn, nan=np.nanmedian(y_all))

test_lgb = np.nan_to_num(test_lgb, nan=np.nanmedian(y_all))
test_xgb = np.nan_to_num(test_xgb, nan=np.nanmedian(y_all))
test_cat = np.nan_to_num(test_cat, nan=np.nanmedian(y_all))
test_et = np.nan_to_num(test_et, nan=np.nanmedian(y_all))
test_dart = np.nan_to_num(test_dart, nan=np.nanmedian(y_all))
test_ridge = np.nan_to_num(test_ridge, nan=np.nanmedian(y_all))
test_nn = np.nan_to_num(test_nn, nan=np.nanmedian(y_all))

# Blend Optimization via SLSQP with L2 regularization
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_dart, oof_ridge, oof_nn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_et, test_dart, test_ridge, test_nn])


def blend_objective(weights):
    w = np.array(weights)
    pred = np.clip(np.dot(oof_matrix, w), 0.0, None)
    pred = np.nan_to_num(pred, nan=np.nanmedian(y_all))
    mse = mean_squared_error(y_all, pred)
    l2_reg = 1e-4 * np.sum(w ** 2)
    return np.sqrt(mse) + l2_reg


init_w = np.ones(oof_matrix.shape[1]) / oof_matrix.shape[1]
bounds = [(0.0, 1.0) for _ in range(oof_matrix.shape[1])]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    init_w,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = (
    opt_res.x if opt_res.success else np.ones(oof_matrix.shape[1]) / oof_matrix.shape[1]
)
best_weights = best_weights / np.sum(best_weights)

oof_blended = np.clip(np.dot(oof_matrix, best_weights), 0.0, None)
final_score = float(np.sqrt(mean_squared_error(y_all, oof_blended)))

# --------------------------------------------------------------------------
# 6. Submission Generation & Metric Reporting
# --------------------------------------------------------------------------
test_blended = np.clip(np.dot(test_matrix, best_weights), 0.0, None)

submission_df = pd.DataFrame(
    {"id": np.arange(n_test, dtype=int), TARGET_COL: test_blended}
)
sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(sub_path, index=False)

print(f"Final Validation Score: {final_score}")
