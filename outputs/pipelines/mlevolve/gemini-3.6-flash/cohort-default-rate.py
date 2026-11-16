import json
import os
import pickle
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Ensure required output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Load dataset
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Average or Greater than 30"
if target_col not in train_df.columns:
    possible_targets = [
        c
        for c in train_df.columns
        if "30" in c or "target" in c.lower() or "default" in c.lower()
    ]
    if possible_targets:
        target_col = possible_targets[0]

# Standardize ID column
id_col = None
for c in ["id", "ID", "Id", "row_id"]:
    if c in train_df.columns and c in test_df.columns:
        id_col = c
        break

if id_col is None:
    train_df["id"] = np.arange(len(train_df))
    test_df["id"] = np.arange(len(test_df))
    id_col = "id"

test_ids = test_df[id_col].values

# Standardize target values into 0/1 binary integers
if train_df[target_col].dtype == object or train_df[target_col].dtype == bool:
    train_df[target_col] = (
        train_df[target_col]
        .astype(str)
        .str.lower()
        .isin(["1", "true", "yes", "y"])
        .astype(int)
    )
else:
    train_df[target_col] = train_df[target_col].astype(int)

y = train_df[target_col].values

# 3. Dynamic Feature Identification
feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]

num_cols = []
cat_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train_df[c]):
        if train_df[c].nunique() <= 10 and train_df[c].dtype == "object":
            cat_cols.append(c)
        else:
            num_cols.append(c)
    else:
        cat_cols.append(c)

# 4. Feature Engineering: Missingness Indicators
for df in [train_df, test_df]:
    df["null_count"] = df[feature_cols].isnull().sum(axis=1)
    df["null_ratio"] = df["null_count"] / max(1, len(feature_cols))

missing_cols = [
    c
    for c in feature_cols
    if train_df[c].isnull().sum() > 0 or test_df[c].isnull().sum() > 0
]
for c in missing_cols:
    train_df[f"{c}_isna"] = train_df[c].isnull().astype(int)
    test_df[f"{c}_isna"] = test_df[c].isnull().astype(int)

# 5. Categorical Encoding & Frequency Features
for c in cat_cols:
    train_df[c] = train_df[c].astype(str).fillna("missing")
    test_df[c] = test_df[c].astype(str).fillna("missing")

    freq = train_df[c].value_counts(normalize=True).to_dict()
    train_df[f"{c}_freq"] = train_df[c].map(freq).fillna(0.0).astype(float)
    test_df[f"{c}_freq"] = test_df[c].map(freq).fillna(0.0).astype(float)

# 6. Skewness Transformation for Heavy-Tailed Numeric Features
for c in num_cols:
    val = train_df[c].dropna()
    if len(val) > 0 and abs(val.skew()) > 1.5:
        min_val = min(train_df[c].min(), test_df[c].min())
        shift = abs(min_val) + 1.0 if min_val < 0 else 1.0
        train_df[f"{c}_log1p"] = np.log1p(np.maximum(0, train_df[c] + shift - 1.0))
        test_df[f"{c}_log1p"] = np.log1p(np.maximum(0, test_df[c] + shift - 1.0))

# 7. Pairwise Interaction Ratios & Differences
top_num = num_cols[:6] if len(num_cols) >= 2 else []
for i in range(len(top_num)):
    for j in range(i + 1, len(top_num)):
        c1, c2 = top_num[i], top_num[j]
        train_df[f"{c1}_div_{c2}"] = train_df[c1] / (train_df[c2].abs() + 1e-5)
        test_df[f"{c1}_div_{c2}"] = test_df[c1] / (test_df[c2].abs() + 1e-5)

        train_df[f"{c1}_sub_{c2}"] = train_df[c1] - train_df[c2]
        test_df[f"{c1}_sub_{c2}"] = test_df[c1] - test_df[c2]

# 8. Row-Level Summary Statistics
if len(num_cols) > 1:
    train_df["row_mean"] = train_df[num_cols].mean(axis=1)
    test_df["row_mean"] = test_df[num_cols].mean(axis=1)
    train_df["row_std"] = train_df[num_cols].std(axis=1)
    test_df["row_std"] = test_df[num_cols].std(axis=1)
    train_df["row_min"] = train_df[num_cols].min(axis=1)
    test_df["row_min"] = test_df[num_cols].min(axis=1)
    train_df["row_max"] = train_df[num_cols].max(axis=1)
    test_df["row_max"] = test_df[num_cols].max(axis=1)

# 9. Group Aggregations
if len(cat_cols) > 0 and len(num_cols) > 0:
    grp_cat = cat_cols[0]
    for n_col in num_cols[:3]:
        grp_stats = train_df.groupby(grp_cat)[n_col].agg(["mean", "std"]).reset_index()
        grp_stats.columns = [
            grp_cat,
            f"{grp_cat}_{n_col}_grp_mean",
            f"{grp_cat}_{n_col}_grp_std",
        ]

        train_df = train_df.merge(grp_stats, on=grp_cat, how="left")
        test_df = test_df.merge(grp_stats, on=grp_cat, how="left")

# 10. Stratified K-Fold Split & Out-of-Fold Target Encoding (Strict Leakage Prevention)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_df["fold"] = -1

for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_df, y)):
    train_df.loc[val_idx, "fold"] = fold_idx

global_mean = y.mean()
smoothing = 10.0

for c in cat_cols:
    te_col = f"{c}_target_enc"
    train_df[te_col] = np.nan
    test_df[te_col] = np.nan

    for fold_idx in range(5):
        val_mask = train_df["fold"] == fold_idx
        tr_fold = train_df[train_df["fold"] != fold_idx]

        stats = tr_fold.groupby(c)[target_col].agg(["count", "mean"])
        smooth_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        train_df.loc[val_mask, te_col] = (
            train_df.loc[val_mask, c].map(smooth_te).fillna(global_mean)
        )

    full_stats = train_df.groupby(c)[target_col].agg(["count", "mean"])
    full_smooth_te = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    test_df[te_col] = test_df[c].map(full_smooth_te).fillna(global_mean)

# Clean inf/nan in all engineered features
all_features = [c for c in train_df.columns if c not in [id_col, target_col, "fold"]]

for df in [train_df, test_df]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Convert string categoricals to pandas category type for GBDT support
for c in all_features:
    if train_df[c].dtype == "object":
        train_df[c] = train_df[c].astype("category")
        test_df[c] = pd.Categorical(test_df[c], categories=train_df[c].cat.categories)


# Helper function to prepare numeric matrices for models requiring continuous matrices
def prepare_numeric_df(df, features):
    df_num = pd.DataFrame(index=df.index)
    for col in features:
        if isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype in [
            "object",
            "category",
        ]:
            df_num[col] = df[col].astype("category").cat.codes
        else:
            df_num[col] = df[col]
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce").fillna(0.0)
    return df_num


train_num_df = prepare_numeric_df(train_df, all_features)
test_num_df = prepare_numeric_df(test_df, all_features)


# 11. PyTorch Gated Tabular Architecture & Loss Definition
class GatedTabularBlock(nn.Module):

    def __init__(self, in_features, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.gate = nn.Linear(in_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = (
            nn.Linear(in_features, hidden_dim)
            if in_features != hidden_dim
            else nn.Identity()
        )

    def forward(self, x):
        res = self.residual(x)
        h = self.act(self.fc1(x))
        g = torch.sigmoid(self.gate(x))
        h = h * g
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(res + h)


class TabularGatedNet(nn.Module):

    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(GatedTabularBlock(curr_dim, h_dim, dropout_rate))
            curr_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(curr_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits.squeeze(-1)


class BinaryFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()


# 12. Model Multi-Architecture Training & OOF Predictions
model_names = ["lgb_gbdt1", "lgb_gbdt2", "xgb_hist"]
if HAS_CATBOOST:
    model_names.append("catboost")

oof_preds = {name: np.zeros(len(train_df)) for name in model_names}
test_preds = {name: np.zeros(len(test_df)) for name in model_names}

num_folds = 5

for fold_idx in range(num_folds):
    tr_mask = train_df["fold"] != fold_idx
    val_mask = train_df["fold"] == fold_idx

    X_tr, y_tr = train_df.loc[tr_mask, all_features], y[tr_mask]
    X_va, y_va = train_df.loc[val_mask, all_features], y[val_mask]

    X_tr_num, X_va_num = (
        train_num_df.loc[tr_mask].values,
        train_num_df.loc[val_mask].values,
    )
    X_te_num = test_num_df.values

    # Model A: LightGBM GBDT Standard Deep
    lgb_gbdt1 = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold_idx,
        n_estimators=1000,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_gbdt1.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds["lgb_gbdt1"][val_mask] = lgb_gbdt1.predict_proba(X_va)[:, 1]
    test_preds["lgb_gbdt1"] += (
        lgb_gbdt1.predict_proba(test_df[all_features])[:, 1] / num_folds
    )

    # Model B: LightGBM GBDT Shallow Regularized
    lgb_gbdt2 = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.04,
        num_leaves=15,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=100 + fold_idx,
        n_estimators=1000,
        verbose=-1,
        n_jobs=-1,
    )
    lgb_gbdt2.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_preds["lgb_gbdt2"][val_mask] = lgb_gbdt2.predict_proba(X_va)[:, 1]
    test_preds["lgb_gbdt2"] += (
        lgb_gbdt2.predict_proba(test_df[all_features])[:, 1] / num_folds
    )

    # Model C: XGBoost Hist
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=200 + fold_idx,
        early_stopping_rounds=50,
        eval_metric="auc",
        n_jobs=-1,
    )
    xgb_model.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_va)], verbose=False)
    oof_preds["xgb_hist"][val_mask] = xgb_model.predict_proba(X_va_num)[:, 1]
    test_preds["xgb_hist"] += xgb_model.predict_proba(X_te_num)[:, 1] / num_folds

    # Model D: CatBoost Classifier
    if HAS_CATBOOST:
        cb_model = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3.0,
            eval_metric="AUC",
            random_seed=300 + fold_idx,
            early_stopping_rounds=50,
            verbose=0,
        )
        cb_model.fit(X_tr_num, y_tr, eval_set=(X_va_num, y_va))
        oof_preds["catboost"][val_mask] = cb_model.predict_proba(X_va_num)[:, 1]
        test_preds["catboost"] += cb_model.predict_proba(X_te_num)[:, 1] / num_folds

# 13. Rank Percentile Transformation & Metric-Aligned Optimization
oof_ranks = {}
test_ranks = {}

for name in model_names:
    oof_ranks[name] = rankdata(oof_preds[name]) / len(y)
    test_ranks[name] = rankdata(test_preds[name]) / len(test_preds[name])


def roc_auc_optimization_objective(weights):
    weights = np.maximum(0, weights)
    total = np.sum(weights)
    if total == 0:
        return 0.0
    weights /= total
    blend = np.zeros(len(y))
    for idx, name in enumerate(model_names):
        blend += weights[idx] * oof_ranks[name]
    return -roc_auc_score(y, blend)


init_weights = np.ones(len(model_names)) / len(model_names)
res = minimize(
    roc_auc_optimization_objective,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
)

opt_weights = np.maximum(0, res.x)
opt_weights /= np.sum(opt_weights)

blended_oof = np.zeros(len(y))
final_test_preds = np.zeros(len(test_df))

for idx, name in enumerate(model_names):
    blended_oof += opt_weights[idx] * oof_ranks[name]
    final_test_preds += opt_weights[idx] * test_ranks[name]

final_auc = roc_auc_score(y, blended_oof)

# 14. Save Final Submission
submission_df = pd.DataFrame({"id": test_ids, target_col: final_test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

# Required final output line
print(f"Final Validation Score: {final_auc}")