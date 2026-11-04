import os
import re
import warnings
from typing import Any, Dict, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from catboost import CatBoostRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Setup paths and load data
# =============================================================================
input_dir = "./input"
submission_dir = "./submission"
working_dir = "./working"

os.makedirs(submission_dir, exist_ok=True)
os.makedirs(working_dir, exist_ok=True)

train_path = os.path.join(input_dir, "train.parquet")
test_path = os.path.join(input_dir, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Clean column names by stripping trailing/leading whitespaces
train_df.columns = [str(c).strip() for c in train_df.columns]
test_df.columns = [str(c).strip() for c in test_df.columns]

target_col = "Adjusted MLR"
if target_col not in train_df.columns:
    matches = [
        c for c in train_df.columns if "adjusted" in c.lower() and "mlr" in c.lower()
    ]
    if matches:
        target_col = matches[0]

# Preserve ID or create 0-based integer index
if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))
if "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df))

y_train = train_df[target_col].copy().astype(np.float32)
train_features = train_df.drop(columns=[target_col]).copy()
test_features = test_df.copy()

print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")
print(
    f"Target column: '{target_col}', Target summary: min={y_train.min():.4f}, mean={y_train.mean():.4f}, max={y_train.max():.4f}"
)

# =============================================================================
# 2. String-to-Numeric Cleaning & Type Detection
# =============================================================================
combined_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)

ignore_cols = {"id"}
feature_cols = [c for c in train_features.columns if c not in ignore_cols]


def clean_numeric_str(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in ["nan", "none", "null", "n/a", "na", "-", ".", "unknown"]:
        return np.nan
    # Remove currency, percentage, commas and extra spaces
    s = s.replace("$", "").replace("%", "").replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if match:
            try:
                val_extracted = float(match.group(0))
                if "(" in str(val) and ")" in str(val):
                    val_extracted = -abs(val_extracted)
                return val_extracted
            except (ValueError, TypeError):
                return np.nan
        return np.nan


# Robust parsing across object columns to convert formatted financial figures to float
for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(combined_df[col]):
        converted = combined_df[col].apply(clean_numeric_str)
        non_null_mask = combined_df[col].notnull() & (
            combined_df[col].astype(str).str.strip().str.lower().isin(["", "nan", "none", "null", "n/a", "na"]) == False
        )
        if non_null_mask.sum() > 0:
            success_ratio = (
                converted[non_null_mask].notnull().sum() / non_null_mask.sum()
            )
            if success_ratio >= 0.40:
                train_features[col] = (
                    train_features[col].apply(clean_numeric_str).astype(np.float32)
                )
                test_features[col] = (
                    test_features[col].apply(clean_numeric_str).astype(np.float32)
                )
                combined_df[col] = (
                    combined_df[col].apply(clean_numeric_str).astype(np.float32)
                )

numeric_cols = []
categorical_cols = []
text_cols = []

for c in feature_cols:
    if pd.api.types.is_numeric_dtype(combined_df[c]):
        numeric_cols.append(c)
    else:
        nunique = combined_df[c].nunique(dropna=True)
        if nunique < 50:
            categorical_cols.append(c)
        else:
            text_cols.append(c)

print(
    f"Detected {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(text_cols)} text columns."
)


# =============================================================================
# 3. Domain & Actuarial Feature Engineering
# =============================================================================
def engineer_domain_features(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    df_out = df.copy()

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df_out[f"{col}_isnan"] = df[col].isnull().astype(np.float32)

    claim_cols = [
        c
        for c in num_cols
        if re.search(r"claim|incurred|clinical|service|paid|numerator", c, re.I)
    ]
    premium_cols = [
        c for c in num_cols if re.search(r"premium|revenue|earned|rate|denominator", c, re.I)
    ]
    expense_cols = [
        c
        for c in num_cols
        if re.search(r"expense|admin|quality|tax|fee|cost|qia|hcqi", c, re.I)
    ]
    member_cols = [
        c
        for c in num_cols
        if re.search(r"member|month|enrol|life|count|covered", c, re.I)
    ]
    cred_cols = [
        c
        for c in num_cols
        if re.search(r"credib|adjust|factor|base|target", c, re.I)
    ]

    eps = 1e-6
    for p_col in premium_cols:
        p_val = df[p_col].fillna(0)
        for c_col in claim_cols:
            c_val = df[c_col].fillna(0)
            df_out[f"ratio_{c_col}_over_{p_col}"] = (c_val + eps) / (p_val + eps)
            df_out[f"diff_{p_col}_minus_{c_col}"] = p_val - c_val
        for e_col in expense_cols:
            e_val = df[e_col].fillna(0)
            df_out[f"ratio_{e_col}_over_{p_col}"] = (e_val + eps) / (p_val + eps)
            for c_col in claim_cols:
                c_val = df[c_col].fillna(0)
                df_out[f"mlr_total_{c_col}_{e_col}_over_{p_col}"] = (c_val + e_val + eps) / (p_val + eps)

    for m_col in member_cols:
        m_val = df[m_col].fillna(0) + eps
        for c_col in claim_cols:
            df_out[f"pmpm_{c_col}"] = df[c_col].fillna(0) / m_val
        for p_col in premium_cols:
            df_out[f"pmpm_{p_col}"] = df[p_col].fillna(0) / m_val
        for e_col in expense_cols:
            df_out[f"pmpm_{e_col}"] = df[e_col].fillna(0) / m_val

    for c in num_cols:
        val = df[c].fillna(0)
        if (val >= 0).all() and (val.max() > 100):
            df_out[f"log1p_{c}"] = np.log1p(val.clip(lower=0))

    if len(num_cols) <= 25:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col1, col2 = num_cols[i], num_cols[j]
                v1, v2 = df[col1].fillna(0), df[col2].fillna(0)
                df_out[f"ratio_{col1}_div_{col2}"] = (v1 + eps) / (v2 + eps)
                df_out[f"prod_{col1}_mul_{col2}"] = v1 * v2
                df_out[f"sub_{col1}_minus_{col2}"] = v1 - v2

    return df_out


train_feat_eng = engineer_domain_features(train_features, numeric_cols)
test_feat_eng = engineer_domain_features(test_features, numeric_cols)

# Categorical Encodings & Text TF-IDF + SVD
all_cat_cols = categorical_cols + text_cols
for cat in all_cat_cols:
    train_feat_eng[cat] = train_feat_eng[cat].astype(str).fillna("MISSING")
    test_feat_eng[cat] = test_feat_eng[cat].astype(str).fillna("MISSING")

    freq_map = (
        pd.concat([train_feat_eng[cat], test_feat_eng[cat]])
        .value_counts(normalize=True)
        .to_dict()
    )
    train_feat_eng[f"{cat}_freq"] = (
        train_feat_eng[cat].map(freq_map).fillna(0).astype(np.float32)
    )
    test_feat_eng[f"{cat}_freq"] = (
        test_feat_eng[cat].map(freq_map).fillna(0).astype(np.float32)
    )

    le = LabelEncoder()
    unique_vals = list(
        set(train_feat_eng[cat].unique()).union(set(test_feat_eng[cat].unique()))
    )
    le.fit(unique_vals)
    train_feat_eng[f"{cat}_le"] = le.transform(train_feat_eng[cat])
    test_feat_eng[f"{cat}_le"] = le.transform(test_feat_eng[cat])

# Text sub-string embeddings using TF-IDF + TruncatedSVD
for t_col in text_cols:
    corpus = pd.concat([train_feat_eng[t_col], test_feat_eng[t_col]]).fillna("")
    n_components = min(4, corpus.nunique() - 1)
    if n_components >= 2:
        tfidf = TfidfVectorizer(
            ngram_range=(2, 4),
            analyzer="char_wb",
            max_features=250,
            sublinear_tf=True,
        )
        tfidf_mat = tfidf.fit_transform(corpus)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for comp_i in range(n_components):
            train_feat_eng[f"{t_col}_svd_{comp_i}"] = svd_mat[: len(train_feat_eng), comp_i].astype(np.float32)
            test_feat_eng[f"{t_col}_svd_{comp_i}"] = svd_mat[len(train_feat_eng) :, comp_i].astype(np.float32)

# Multi-column categorical interaction frequency features
cat_pair_list = categorical_cols[:4]
for i in range(len(cat_pair_list)):
    for j in range(i + 1, len(cat_pair_list)):
        c1, c2 = cat_pair_list[i], cat_pair_list[j]
        pair_tr = train_feat_eng[c1].astype(str) + "_" + train_feat_eng[c2].astype(str)
        pair_te = test_feat_eng[c1].astype(str) + "_" + test_feat_eng[c2].astype(str)
        pair_freq = pd.concat([pair_tr, pair_te]).value_counts(normalize=True).to_dict()
        train_feat_eng[f"pair_{c1}_{c2}_freq"] = pair_tr.map(pair_freq).fillna(0).astype(np.float32)
        test_feat_eng[f"pair_{c1}_{c2}_freq"] = pair_te.map(pair_freq).fillna(0).astype(np.float32)

# 10-Fold Leakage-Free Stratified Cross-Validation Splitting
n_splits = 10
num_bins = min(10, max(2, len(train_df) // 20))
try:
    binned_y = pd.qcut(y_train, q=num_bins, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = list(cv.split(train_feat_eng, binned_y))
except Exception:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = list(cv.split(train_feat_eng, y_train))

# Out-of-fold target encoding (mean, std, min, max) for all categorical & text columns
for cat in all_cat_cols:
    oof_target_mean = np.zeros(len(train_feat_eng), dtype=np.float32)
    oof_target_std = np.zeros(len(train_feat_eng), dtype=np.float32)
    oof_target_min = np.zeros(len(train_feat_eng), dtype=np.float32)
    oof_target_max = np.zeros(len(train_feat_eng), dtype=np.float32)
    test_target_mean = np.zeros(len(test_feat_eng), dtype=np.float32)
    test_target_std = np.zeros(len(test_feat_eng), dtype=np.float32)
    test_target_min = np.zeros(len(test_feat_eng), dtype=np.float32)
    test_target_max = np.zeros(len(test_feat_eng), dtype=np.float32)

    global_mean = float(y_train.mean())
    global_std = float(y_train.std()) if len(y_train) > 1 else 0.0
    global_min = float(y_train.min())
    global_max = float(y_train.max())

    for trn_idx, val_idx in cv_splits:
        X_tr_cat = train_feat_eng.iloc[trn_idx][cat]
        y_tr_fold = y_train.iloc[trn_idx]

        stats = y_tr_fold.groupby(X_tr_cat).agg(["count", "mean", "std", "min", "max"])
        stats["std"] = stats["std"].fillna(0.0)
        stats["min"] = stats["min"].fillna(global_min)
        stats["max"] = stats["max"].fillna(global_max)

        smooth = 10.0
        smoothed_mean = (stats["count"] * stats["mean"] + smooth * global_mean) / (
            stats["count"] + smooth
        )
        smoothed_std = (stats["count"] * stats["std"] + smooth * global_std) / (
            stats["count"] + smooth
        )

        mean_map = smoothed_mean.to_dict()
        std_map = smoothed_std.to_dict()
        min_map = stats["min"].to_dict()
        max_map = stats["max"].to_dict()

        oof_target_mean[val_idx] = train_feat_eng.iloc[val_idx][cat].map(mean_map).fillna(global_mean)
        oof_target_std[val_idx] = train_feat_eng.iloc[val_idx][cat].map(std_map).fillna(global_std)
        oof_target_min[val_idx] = train_feat_eng.iloc[val_idx][cat].map(min_map).fillna(global_min)
        oof_target_max[val_idx] = train_feat_eng.iloc[val_idx][cat].map(max_map).fillna(global_max)

        test_target_mean += test_feat_eng[cat].map(mean_map).fillna(global_mean).values / n_splits
        test_target_std += test_feat_eng[cat].map(std_map).fillna(global_std).values / n_splits
        test_target_min += test_feat_eng[cat].map(min_map).fillna(global_min).values / n_splits
        test_target_max += test_feat_eng[cat].map(max_map).fillna(global_max).values / n_splits

    train_feat_eng[f"{cat}_te_mean"] = oof_target_mean
    train_feat_eng[f"{cat}_te_std"] = oof_target_std
    train_feat_eng[f"{cat}_te_min"] = oof_target_min
    train_feat_eng[f"{cat}_te_max"] = oof_target_max
    test_feat_eng[f"{cat}_te_mean"] = test_target_mean
    test_feat_eng[f"{cat}_te_std"] = test_target_std
    test_feat_eng[f"{cat}_te_min"] = test_target_min
    test_feat_eng[f"{cat}_te_max"] = test_target_max

cols_to_drop = [c for c in all_cat_cols if c in train_feat_eng.columns]
X_train_final = train_feat_eng.drop(columns=cols_to_drop + ["id"], errors="ignore")
X_test_final = test_feat_eng.drop(columns=cols_to_drop + ["id"], errors="ignore")

for col in X_train_final.columns:
    if col in X_test_final.columns:
        med_val = X_train_final[col].median()
        if pd.isna(med_val):
            med_val = 0.0
        X_train_final[col] = X_train_final[col].fillna(med_val)
        X_test_final[col] = X_test_final[col].fillna(med_val)

feature_names = [c for c in X_train_final.columns if c in X_test_final.columns]
X_train_final = X_train_final[feature_names].astype(np.float32)
X_test_final = X_test_final[feature_names].astype(np.float32)

print(
    f"Engineered Features count: {len(feature_names)}, Processed train shape: {X_train_final.shape}"
)

# =============================================================================
# 4. Model Architectures and Factories
# =============================================================================


def get_lgb_model(
    seed: int = 42, custom_params: Dict[str, Any] = None
) -> lgb.LGBMRegressor:
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return lgb.LGBMRegressor(**default_params)


def get_xgb_model(
    seed: int = 42, custom_params: Dict[str, Any] = None
) -> xgb.XGBRegressor:
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return xgb.XGBRegressor(**default_params)


def get_cat_model(
    seed: int = 42, custom_params: Dict[str, Any] = None
) -> CatBoostRegressor:
    default_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
    }
    if custom_params:
        default_params.update(custom_params)
    return CatBoostRegressor(**default_params)


class TabularMLPBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + res


class TabularMLPNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                TabularMLPBlock(
                    hidden_dims[i], hidden_dims[i + 1], dropout_rate=dropout_rate
                )
            )
        self.hidden_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_layer(x)
        feat = self.hidden_blocks(feat)
        out = self.head(feat)
        return out.squeeze(-1)


def build_neural_pipeline(
    input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Any]:
    model = TabularMLPNet(
        input_dim=input_dim, hidden_dims=(256, 128, 64), dropout_rate=0.15
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-5
    )
    return model, criterion, optimizer, scheduler


# =============================================================================
# 5. Training and Evaluation Loops
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_samples = len(X_train_final)
n_test = len(X_test_final)
num_folds = len(cv_splits)

model_names = ["lgb", "xgb", "cat", "ridge", "mlp"]
oof_preds = {m: np.zeros(n_samples, dtype=np.float32) for m in model_names}
test_preds = {m: np.zeros(n_test, dtype=np.float32) for m in model_names}

y_true_arr = (
    y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train
)
X_train_arr = (
    X_train_final.values if isinstance(X_train_final, pd.DataFrame) else X_train_final
)
X_test_arr = (
    X_test_final.values if isinstance(X_test_final, pd.DataFrame) else X_test_final
)

for fold, (trn_idx, val_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_train_arr[trn_idx], y_true_arr[trn_idx]
    X_va, y_va = X_train_arr[val_idx], y_true_arr[val_idx]

    # 1. LightGBM
    lgb_model = get_lgb_model(seed=42 + fold)
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=lgb_callbacks)

    val_pred_lgb = lgb_model.predict(X_va)
    oof_preds["lgb"][val_idx] = val_pred_lgb
    test_preds["lgb"] += lgb_model.predict(X_test_arr) / num_folds

    # 2. XGBoost
    xgb_model = get_xgb_model(
        seed=42 + fold, custom_params={"early_stopping_rounds": 50}
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_pred_xgb = xgb_model.predict(X_va)
    oof_preds["xgb"][val_idx] = val_pred_xgb
    test_preds["xgb"] += xgb_model.predict(X_test_arr) / num_folds

    # 3. CatBoost
    cat_model = get_cat_model(
        seed=42 + fold, custom_params={"early_stopping_rounds": 50}
    )
    cat_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_pred_cat = cat_model.predict(X_va)
    oof_preds["cat"][val_idx] = val_pred_cat
    test_preds["cat"] += cat_model.predict(X_test_arr) / num_folds

    # 4. Regularized RidgeCV
    scaler_ridge = StandardScaler()
    X_tr_s_ridge = np.nan_to_num(scaler_ridge.fit_transform(X_tr), nan=0.0)
    X_va_s_ridge = np.nan_to_num(scaler_ridge.transform(X_va), nan=0.0)
    X_te_s_ridge = np.nan_to_num(scaler_ridge.transform(X_test_arr), nan=0.0)

    ridge_model = RidgeCV(alphas=np.logspace(-3, 4, 30))
    ridge_model.fit(X_tr_s_ridge, y_tr)

    val_pred_ridge = ridge_model.predict(X_va_s_ridge)
    oof_preds["ridge"][val_idx] = val_pred_ridge
    test_preds["ridge"] += ridge_model.predict(X_te_s_ridge) / num_folds

    # 5. Deep Tabular MLP
    scaler_mlp = StandardScaler()
    X_tr_s = np.nan_to_num(scaler_mlp.fit_transform(X_tr), nan=0.0)
    X_va_s = np.nan_to_num(scaler_mlp.transform(X_va), nan=0.0)
    X_te_s = np.nan_to_num(scaler_mlp.transform(X_test_arr), nan=0.0)

    train_ds = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_s, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    batch_size = min(64, max(16, len(X_tr_s) // 8))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    mlp_model, criterion, optimizer, scheduler = build_neural_pipeline(
        input_dim=X_tr_s.shape[1], lr=1e-3, weight_decay=1e-4
    )
    mlp_model.to(device)

    best_val_loss = float("inf")
    best_weights = None
    patience = 20
    patience_counter = 0

    for epoch in range(1, 81):
        mlp_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = mlp_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        mlp_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_out = mlp_model(batch_x)
                val_loss = criterion(val_out, batch_y)
                val_losses.append(val_loss.item() * len(batch_x))

        mean_val_loss = np.sum(val_losses) / len(X_va_s)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in mlp_model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    mlp_model.load_state_dict(best_weights)
    mlp_model.to(device)
    mlp_model.eval()

    with torch.no_grad():
        val_mlp_pred = (
            mlp_model(torch.tensor(X_va_s, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
        test_mlp_pred = (
            mlp_model(torch.tensor(X_te_s, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )

    oof_preds["mlp"][val_idx] = val_mlp_pred
    test_preds["mlp"] += test_mlp_pred / num_folds

    fold_lgb_rmse = np.sqrt(np.mean((y_va - val_pred_lgb) ** 2))
    fold_xgb_rmse = np.sqrt(np.mean((y_va - val_pred_xgb) ** 2))
    fold_cat_rmse = np.sqrt(np.mean((y_va - val_pred_cat) ** 2))
    fold_ridge_rmse = np.sqrt(np.mean((y_va - val_pred_ridge) ** 2))
    fold_mlp_rmse = np.sqrt(np.mean((y_va - val_mlp_pred) ** 2))
    print(
        f"Fold {fold+1}/{num_folds} - LGB: {fold_lgb_rmse:.5f} | XGB: {fold_xgb_rmse:.5f} | CAT: {fold_cat_rmse:.5f} | Ridge: {fold_ridge_rmse:.5f} | MLP: {fold_mlp_rmse:.5f}"
    )

# =============================================================================
# 6. Constrained Ensemble Optimization & Final Metric
# =============================================================================
oof_matrix = np.column_stack([oof_preds[m] for m in model_names])
test_matrix = np.column_stack([test_preds[m] for m in model_names])

init_weights = np.ones(len(model_names)) / len(model_names)
bounds = [(0.0, 1.0) for _ in model_names]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def regularized_ensemble_loss(w):
    pred = oof_matrix @ w
    rmse = np.sqrt(np.mean((y_true_arr - pred) ** 2))
    ridge_penalty = 1e-4 * np.sum(w ** 2)
    return rmse + ridge_penalty


opt_result = minimize(
    regularized_ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
if opt_result.success:
    optimal_weights = opt_result.x
else:
    optimal_weights = init_weights

print(f"Optimal Ensemble Weights: {dict(zip(model_names, optimal_weights))}")

final_oof_pred = oof_matrix @ optimal_weights
final_val_score = float(np.sqrt(np.mean((y_true_arr - final_oof_pred) ** 2)))

# =============================================================================
# 7. Generate Submission File
# =============================================================================
final_test_pred = test_matrix @ optimal_weights
y_min, y_max = float(y_true_arr.min()), float(y_true_arr.max())
margin = 0.05 * (y_max - y_min)
final_test_pred = np.clip(final_test_pred, y_min - margin, y_max + margin)

test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(n_test)

submission_df = pd.DataFrame({"id": test_ids, "Adjusted MLR": final_test_pred})
submission_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Submission shape: {submission_df.shape}")
print(f"Final Validation Score: {final_val_score}")