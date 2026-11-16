from catboost import CatBoostRegressor
import json
import os
import random
import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import FeatureUnion
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb


# 0. Deterministic Reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# 1. Setup paths and load data
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"
os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

train_path = os.path.join(input_dir, "train.parquet")
test_path = os.path.join(input_dir, "test.parquet")

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = (
    "Price"
    if "Price" in train_df.columns
    else [c for c in train_df.columns if c.lower() == "price"][0]
)

n_train = len(train_df)
n_test = len(test_df)

if "id" not in train_df.columns:
    train_df["id"] = np.arange(n_train)
if "id" not in test_df.columns:
    test_df["id"] = np.arange(n_test)

train_ids = train_df["id"].values
test_ids = test_df["id"].values
y_train = train_df[target_col].values.astype(np.float32)

# 2. Identify column types
feature_cols_raw = [c for c in train_df.columns if c not in [target_col, "id"]]

text_cols = []
cat_cols = []
num_cols = []

for c in feature_cols_raw:
    if pd.api.types.is_numeric_dtype(train_df[c]):
        num_cols.append(c)
    else:
        sample_vals = train_df[c].dropna().astype(str)
        if (
            len(sample_vals) > 0
            and sample_vals.apply(lambda x: len(x.split())).mean() > 4
        ):
            text_cols.append(c)
        else:
            cat_cols.append(c)

# 3. Text Feature Extraction & Domain-Specific Wine Regex Parsing
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
train_folds = np.zeros(n_train, dtype=int)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_folds[val_idx] = fold

train_text_features = pd.DataFrame(index=train_df.index)
test_text_features = pd.DataFrame(index=test_df.index)

wine_quality_keywords = [
    "reserve", "reserva", "gran reserva", "grand cru", "premier cru", "estate",
    "single vineyard", "old vine", "vieilles vignes", "oak", "barrel", "barrique",
    "cuvee", "vintage", "rare", "collection", "signature", "special", "selection",
    "aged", "prestige", "chateau", "domaine", "classic", "superior", "superiore",
    "classico", "riserva", "rich", "complex", "elegant", "intense", "full-bodied",
    "concentrated", "silky", "tannins", "acidity", "finish", "aroma", "crisp",
    "balanced", "deep", "structured", "opulent", "fresh", "lush", "velvety"
]

all_str_cols = text_cols + [c for c in cat_cols if train_df[c].dtype == object]
full_train_text = train_df[all_str_cols].fillna("").astype(str).agg(" ".join, axis=1)
full_test_text = test_df[all_str_cols].fillna("").astype(str).agg(" ".join, axis=1)

# Domain signal regex extraction
for col in all_str_cols:
    train_str = train_df[col].fillna("").astype(str)
    test_str = test_df[col].fillna("").astype(str)

    extracted_train_year = train_str.str.extract(r"\b(18\d\d|19\d\d|20[0-2]\d)\b")[0].astype(float)
    extracted_test_year = test_str.str.extract(r"\b(18\d\d|19\d\d|20[0-2]\d)\b")[0].astype(float)
    if extracted_train_year.notna().sum() > (0.02 * n_train):
        train_text_features[f"{col}_extracted_vintage"] = extracted_train_year
        test_text_features[f"{col}_extracted_vintage"] = extracted_test_year
        v_age_tr = (2024.0 - extracted_train_year).clip(0, 100)
        v_age_te = (2024.0 - extracted_test_year).clip(0, 100)
        train_text_features[f"{col}_vintage_age"] = v_age_tr
        test_text_features[f"{col}_vintage_age"] = v_age_te
        train_text_features[f"{col}_vintage_age_sq"] = v_age_tr ** 2
        test_text_features[f"{col}_vintage_age_sq"] = v_age_te ** 2
        train_text_features[f"{col}_log_vintage_age"] = np.log1p(v_age_tr)
        test_text_features[f"{col}_log_vintage_age"] = np.log1p(v_age_te)

    extracted_train_pct = train_str.str.extract(r"(\d{1,2}(?:\.\d+)?)\s*%")[0].astype(float)
    extracted_test_pct = test_str.str.extract(r"(\d{1,2}(?:\.\d+)?)\s*%")[0].astype(float)
    if extracted_train_pct.notna().sum() > (0.01 * n_train):
        train_text_features[f"{col}_extracted_alcohol"] = extracted_train_pct
        test_text_features[f"{col}_extracted_alcohol"] = extracted_test_pct

    extracted_train_vol = train_str.str.extract(r"(\d{3,4})\s*(?:ml|cl)")[0].astype(float)
    extracted_test_vol = test_str.str.extract(r"(\d{3,4})\s*(?:ml|cl)")[0].astype(float)
    if extracted_train_vol.notna().sum() > (0.01 * n_train):
        train_text_features[f"{col}_extracted_volume"] = extracted_train_vol
        test_text_features[f"{col}_extracted_volume"] = extracted_test_vol

    extracted_train_barrel = train_str.str.extract(r"(\d{1,2})\s*(?:months?|mo|years?|yr)\s*(?:in|of)?\s*(?:oak|barrel|barrique|wood)", flags=re.IGNORECASE)[0].astype(float)
    extracted_test_barrel = test_str.str.extract(r"(\d{1,2})\s*(?:months?|mo|years?|yr)\s*(?:in|of)?\s*(?:oak|barrel|barrique|wood)", flags=re.IGNORECASE)[0].astype(float)
    if extracted_train_barrel.notna().sum() > (0.005 * n_train):
        train_text_features[f"{col}_barrel_months"] = extracted_train_barrel
        test_text_features[f"{col}_barrel_months"] = extracted_test_barrel

train_text_features["quality_kw_count"] = full_train_text.str.lower().apply(
    lambda s: sum(1 for kw in wine_quality_keywords if kw in s)
)
test_text_features["quality_kw_count"] = full_test_text.str.lower().apply(
    lambda s: sum(1 for kw in wine_quality_keywords if kw in s)
)

for tier_kw in ["reserve", "reserva", "grand cru", "premier cru", "single vineyard", "old vine", "estate", "cuvee", "oak", "barrique"]:
    kw_slug = tier_kw.replace(" ", "_")
    train_text_features[f"flag_{kw_slug}"] = full_train_text.str.lower().str.contains(tier_kw, regex=False).astype(float)
    test_text_features[f"flag_{kw_slug}"] = full_test_text.str.lower().str.contains(tier_kw, regex=False).astype(float)

for col in text_cols:
    train_s = train_df[col].fillna("").astype(str)
    test_s = test_df[col].fillna("").astype(str)

    train_text_features[f"{col}_char_len"] = train_s.str.len()
    test_text_features[f"{col}_char_len"] = test_s.str.len()
    train_text_features[f"{col}_word_count"] = train_s.apply(lambda x: len(x.split()))
    test_text_features[f"{col}_word_count"] = test_s.apply(lambda x: len(x.split()))
    train_text_features[f"{col}_has_excl"] = train_s.apply(lambda x: int("!" in x))
    test_text_features[f"{col}_has_excl"] = test_s.apply(lambda x: int("!" in x))

# Combined Word + Char TF-IDF
word_vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=25000,
    stop_words="english",
    token_pattern=r"(?u)\b\w+\b",
    sublinear_tf=True,
)
char_vec = TfidfVectorizer(
    ngram_range=(3, 5),
    analyzer="char_wb",
    max_features=20000,
    sublinear_tf=True,
)
tfidf_global = FeatureUnion([("word", word_vec), ("char", char_vec)])
train_tfidf_global = tfidf_global.fit_transform(full_train_text)
test_tfidf_global = tfidf_global.transform(full_test_text)

# Leak-free OOF TF-IDF Ridge Text Model Predictions as Tabular Features
for alpha_val in [0.5, 2.0, 10.0]:
    oof_ridge_feat = np.zeros(n_train, dtype=np.float32)
    test_ridge_feat = np.zeros(n_test, dtype=np.float32)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(train_df)):
        ridge_m = Ridge(alpha=alpha_val, random_state=42)
        ridge_m.fit(train_tfidf_global[trn_idx], y_train[trn_idx])
        oof_ridge_feat[val_idx] = ridge_m.predict(train_tfidf_global[val_idx])
        test_ridge_feat += ridge_m.predict(test_tfidf_global) / n_folds
    train_text_features[f"oof_ridge_pred_a{alpha_val}"] = oof_ridge_feat
    test_text_features[f"oof_ridge_pred_a{alpha_val}"] = test_ridge_feat

n_svd_components = min(40, max(10, int(np.sqrt(n_train) // 4)))
svd_global = TruncatedSVD(n_components=n_svd_components, random_state=42)
train_svd_global = svd_global.fit_transform(train_tfidf_global)
test_svd_global = svd_global.transform(test_tfidf_global)

for i in range(n_svd_components):
    train_text_features[f"global_svd_{i}"] = train_svd_global[:, i]
    test_text_features[f"global_svd_{i}"] = test_svd_global[:, i]

# 4. Numerical Preprocessing & Interaction Features
train_num = train_df[num_cols].copy()
test_num = test_df[num_cols].copy()

for c in num_cols:
    train_missing = train_num[c].isna()
    test_missing = test_num[c].isna()
    if train_missing.sum() > 0 or test_missing.sum() > 0:
        train_num[f"{c}_isna"] = train_missing.astype(int)
        test_num[f"{c}_isna"] = test_missing.astype(int)

    med = train_num[c].median()
    train_num[c] = train_num[c].fillna(med)
    test_num[c] = test_num[c].fillna(med)

base_num_cols = list(num_cols)
for c in base_num_cols:
    if (train_num[c] >= 0).all() and (test_num[c] >= 0).all():
        train_num[f"{c}_log1p"] = np.log1p(train_num[c].clip(lower=0))
        test_num[f"{c}_log1p"] = np.log1p(test_num[c].clip(lower=0))

    # Non-linear rating & score transformations
    c_std = float(train_num[c].std() + 1e-5)
    c_mean = float(train_num[c].mean())
    train_num[f"{c}_sq"] = train_num[c] ** 2
    test_num[f"{c}_sq"] = test_num[c] ** 2
    train_num[f"{c}_cube"] = (train_num[c] / (c_mean + 1e-5)) ** 3
    test_num[f"{c}_cube"] = (test_num[c] / (c_mean + 1e-5)) ** 3
    train_num[f"{c}_pow1p5"] = np.power(np.clip(train_num[c], 0, None), 1.5)
    test_num[f"{c}_pow1p5"] = np.power(np.clip(test_num[c], 0, None), 1.5)
    train_num[f"{c}_exp_norm"] = np.exp(np.clip((train_num[c] - c_mean) / c_std, -5.0, 5.0))
    test_num[f"{c}_exp_norm"] = np.exp(np.clip((test_num[c] - c_mean) / c_std, -5.0, 5.0))
    train_num[f"{c}_sig_norm"] = 1.0 / (1.0 + np.exp(-np.clip((train_num[c] - c_mean) / c_std, -10.0, 10.0)))
    test_num[f"{c}_sig_norm"] = 1.0 / (1.0 + np.exp(-np.clip((test_num[c] - c_mean) / c_std, -10.0, 10.0)))

for i in range(len(base_num_cols)):
    for j in range(i + 1, min(len(base_num_cols), i + 6)):
        c1, c2 = base_num_cols[i], base_num_cols[j]
        train_num[f"{c1}_ratio_{c2}"] = train_num[c1] / (train_num[c2].abs() + 1e-6)
        test_num[f"{c1}_ratio_{c2}"] = test_num[c1] / (test_num[c2].abs() + 1e-6)
        train_num[f"{c1}_mult_{c2}"] = train_num[c1] * train_num[c2]
        test_num[f"{c1}_mult_{c2}"] = test_num[c1] * test_num[c2]

# 5. Categorical Encodings (Frequency + Group Aggregations + Out-of-Fold Target Encoding)
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
train_folds = np.zeros(n_train, dtype=int)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_folds[val_idx] = fold

train_cat_features = pd.DataFrame(index=train_df.index)
test_cat_features = pd.DataFrame(index=test_df.index)

# Generate higher-order categorical pairs (e.g. winery+variety, country+province)
interaction_cats = []
for i in range(len(cat_cols)):
    for j in range(i + 1, min(len(cat_cols), i + 4)):
        c1, c2 = cat_cols[i], cat_cols[j]
        pair_col = f"{c1}__x__{c2}"
        interaction_cats.append(pair_col)
        train_df[pair_col] = (
            train_df[c1].fillna("__MISSING__").astype(str)
            + "___"
            + train_df[c2].fillna("__MISSING__").astype(str)
        )
        test_df[pair_col] = (
            test_df[c1].fillna("__MISSING__").astype(str)
            + "___"
            + test_df[c2].fillna("__MISSING__").astype(str)
        )

all_cat_cols_to_encode = cat_cols + interaction_cats

for c in all_cat_cols_to_encode:
    s_train = train_df[c].fillna("__MISSING__").astype(str)
    s_test = test_df[c].fillna("__MISSING__").astype(str)

    freq = s_train.value_counts(normalize=True).to_dict()
    train_cat_features[f"{c}_freq"] = s_train.map(freq).fillna(0.0)
    test_cat_features[f"{c}_freq"] = s_test.map(freq).fillna(0.0)

    for num_c in base_num_cols[:4]:
        grouped = train_df.groupby(s_train)[num_c].agg(["mean", "std"])
        mean_map = grouped["mean"].to_dict()
        std_map = grouped["std"].fillna(0.0).to_dict()

        train_cat_features[f"{c}_{num_c}_mean"] = s_train.map(mean_map).fillna(
            train_df[num_c].mean()
        )
        test_cat_features[f"{c}_{num_c}_mean"] = s_test.map(mean_map).fillna(
            train_df[num_c].mean()
        )
        train_cat_features[f"{c}_{num_c}_std"] = s_train.map(std_map).fillna(0.0)
        test_cat_features[f"{c}_{num_c}_std"] = s_test.map(std_map).fillna(0.0)

    log_y = np.log1p(np.clip(y_train, a_min=0, a_max=None))
    global_mean_log_y = float(np.mean(log_y))
    smoothing = 10.0

    oof_te = np.zeros(n_train, dtype=np.float32)
    for fold, (t_idx, v_idx) in enumerate(kf.split(train_df)):
        fold_train_s = s_train.iloc[t_idx]
        fold_train_y = log_y[t_idx]

        stats = (
            pd.DataFrame({"cat": fold_train_s, "y": fold_train_y})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean_log_y) / (
            stats["count"] + smoothing
        )
        oof_te[v_idx] = s_train.iloc[v_idx].map(smoothed).fillna(global_mean_log_y)

    full_stats = (
        pd.DataFrame({"cat": s_train, "y": log_y})
        .groupby("cat")["y"]
        .agg(["count", "mean"])
    )
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean_log_y
    ) / (full_stats["count"] + smoothing)

    train_cat_features[f"{c}_te_log_y"] = oof_te
    test_cat_features[f"{c}_te_log_y"] = s_test.map(full_smoothed).fillna(
        global_mean_log_y
    )

# 6. Combine all processed features
X_train = pd.concat([train_num, train_cat_features, train_text_features], axis=1)
X_test = pd.concat([test_num, test_cat_features, test_text_features], axis=1)

all_features = sorted(list(set(X_train.columns).intersection(set(X_test.columns))))
X_train = X_train[all_features].copy()
X_test = X_test[all_features].copy()

for col in all_features:
    med = X_train[col].median()
    if pd.isna(med):
        med = 0.0
    X_train[col] = X_train[col].fillna(med)
    X_test[col] = X_test[col].fillna(med)


# 7. Model Architectures & Parameter Specifications
class TabularResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [TabularResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def create_nn_components(input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4):
    model = TabularResNet(input_dim=input_dim, hidden_dim=256, num_blocks=3)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def get_lgb_params():
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params():
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }


def get_cat_params():
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
    }


def get_hgb_params():
    return {
        "max_iter": 1500,
        "learning_rate": 0.03,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "n_iter_no_change": 50,
        "random_state": 42,
    }


# 8. Training & Cross-Validation
X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

oof_preds_lgb = np.zeros(n_train, dtype=np.float32)
test_preds_lgb = np.zeros(n_test, dtype=np.float32)

oof_preds_xgb = np.zeros(n_train, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)

oof_preds_cb = np.zeros(n_train, dtype=np.float32)
test_preds_cb = np.zeros(n_test, dtype=np.float32)

oof_preds_hgb = np.zeros(n_train, dtype=np.float32)
test_preds_hgb = np.zeros(n_test, dtype=np.float32)

oof_preds_ridge = np.zeros(n_train, dtype=np.float32)
test_preds_ridge = np.zeros(n_test, dtype=np.float32)

oof_preds_nn = np.zeros(n_train, dtype=np.float32)
test_preds_nn = np.zeros(n_test, dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(n_folds):
    trn_mask = train_folds != fold
    val_mask = train_folds == fold

    X_trn_f = X_train.iloc[trn_mask]
    y_trn_f = y_train[trn_mask]
    X_val_f = X_train.iloc[val_mask]
    y_val_f = y_train[val_mask]

    # LightGBM training
    lgb_reg = lgb.LGBMRegressor(**get_lgb_params())
    lgb_reg.fit(
        X_trn_f,
        y_trn_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_p_lgb = lgb_reg.predict(X_val_f)
    oof_preds_lgb[val_mask] = val_p_lgb
    test_preds_lgb += lgb_reg.predict(X_test) / n_folds

    # XGBoost training
    xgb_reg = xgb.XGBRegressor(
        **get_xgb_params(), early_stopping_rounds=50
    )
    xgb_reg.fit(
        X_trn_f,
        y_trn_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )
    val_p_xgb = xgb_reg.predict(X_val_f)
    oof_preds_xgb[val_mask] = val_p_xgb
    test_preds_xgb += xgb_reg.predict(X_test) / n_folds

    # CatBoost training
    cb_reg = CatBoostRegressor(**get_cat_params())
    cb_reg.fit(
        X_trn_f,
        y_trn_f,
        eval_set=(X_val_f, y_val_f),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_p_cb = cb_reg.predict(X_val_f)
    oof_preds_cb[val_mask] = val_p_cb
    test_preds_cb += cb_reg.predict(X_test) / n_folds

    # HistGradientBoostingRegressor training
    hgb_reg = HistGradientBoostingRegressor(**get_hgb_params())
    hgb_reg.fit(X_trn_f, y_trn_f)
    val_p_hgb = hgb_reg.predict(X_val_f)
    oof_preds_hgb[val_mask] = val_p_hgb
    test_preds_hgb += hgb_reg.predict(X_test) / n_folds

    # Ridge text linear regression
    tfidf_trn_f = train_tfidf_global[trn_mask]
    tfidf_val_f = train_tfidf_global[val_mask]
    ridge_reg = Ridge(alpha=1.0, random_state=42)
    ridge_reg.fit(tfidf_trn_f, y_trn_f)
    val_p_ridge = ridge_reg.predict(tfidf_val_f)
    oof_preds_ridge[val_mask] = val_p_ridge
    test_preds_ridge += ridge_reg.predict(test_tfidf_global) / n_folds

    # PyTorch Tabular ResNet training (Scaling fitted on fold train data only)
    fold_mean = np.nanmean(X_train_np[trn_mask], axis=0, keepdims=True)
    fold_std = np.nanstd(X_train_np[trn_mask], axis=0, keepdims=True) + 1e-6
    X_trn_scaled_f = np.nan_to_num((X_train_np[trn_mask] - fold_mean) / fold_std)
    X_val_scaled_f = np.nan_to_num((X_train_np[val_mask] - fold_mean) / fold_std)
    X_tst_scaled_f = np.nan_to_num((X_test_np - fold_mean) / fold_std)

    X_trn_nn = torch.tensor(X_trn_scaled_f, dtype=torch.float32)
    y_trn_nn = torch.tensor(y_train[trn_mask], dtype=torch.float32)
    X_val_nn = torch.tensor(X_val_scaled_f, dtype=torch.float32)
    y_val_nn = torch.tensor(y_train[val_mask], dtype=torch.float32)
    X_tst_nn = torch.tensor(X_tst_scaled_f, dtype=torch.float32)

    trn_loader = DataLoader(
        TensorDataset(X_trn_nn, y_trn_nn),
        batch_size=128,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_nn, y_val_nn), batch_size=256, shuffle=False
    )
    tst_loader = DataLoader(TensorDataset(X_tst_nn), batch_size=256, shuffle=False)

    nn_model, nn_criterion, nn_optimizer, nn_scheduler = create_nn_components(
        input_dim=len(all_features), lr=1e-3, weight_decay=1e-4
    )
    nn_model.to(device)

    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(35):
        nn_model.train()
        for batch_x, batch_y in trn_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            nn_optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = nn_criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            nn_optimizer.step()

        nn_scheduler.step()

        nn_model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = nn_model(batch_x)
                val_loss_sum += nn_criterion(preds, batch_y).item() * batch_x.size(0)
                val_count += batch_x.size(0)

        current_val_loss = val_loss_sum / val_count
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 8:
                break

    nn_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    nn_model.eval()

    val_nn_preds_list = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            val_nn_preds_list.append(nn_model(batch_x).cpu().numpy())
    oof_preds_nn[val_mask] = np.concatenate(val_nn_preds_list)

    test_nn_preds_list = []
    with torch.no_grad():
        for (batch_x,) in tst_loader:
            batch_x = batch_x.to(device)
            test_nn_preds_list.append(nn_model(batch_x).cpu().numpy())
    test_preds_nn += np.concatenate(test_nn_preds_list) / n_folds

# 9. Ensemble Blending Optimization
oof_matrix = np.column_stack(
    [oof_preds_lgb, oof_preds_xgb, oof_preds_cb, oof_preds_hgb, oof_preds_nn, oof_preds_ridge]
)
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_hgb, test_preds_nn, test_preds_ridge]
)

n_models = oof_matrix.shape[1]


def blend_obj(weights):
    w = weights / (np.sum(weights) + 1e-12)
    pred = oof_matrix @ w
    base_loss = root_mean_squared_error(y_train, pred)
    reg_loss = 1e-4 * np.sum(w ** 2)
    return base_loss + reg_loss


init_weights = np.ones(n_models) / float(n_models)
bounds = [(0.0, 1.0)] * n_models
res = minimize(blend_obj, init_weights, bounds=bounds, method="SLSQP")
optimal_weights = res.x / np.sum(res.x)

oof_ensemble = oof_matrix @ optimal_weights
final_score = root_mean_squared_error(y_train, oof_ensemble)

# 10. Generate Submission File
test_ensemble = test_matrix @ optimal_weights
test_ensemble_clipped = np.clip(test_ensemble, a_min=0.0, a_max=None)

submission_df = pd.DataFrame({"id": test_ids, "Price": test_ensemble_clipped})
submission_path = os.path.join(submission_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_score}")