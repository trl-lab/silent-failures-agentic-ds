import itertools
import math
import os
import random
import re
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


# =========================================================================
# 1. Reproducibility and Device Setup
# =========================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./working", exist_ok=True)
os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


# =========================================================================
# 2. Data Loading & Leak-Free Feature Engineering
# =========================================================================
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Overall"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in train data.")

if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))
if "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df))

print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")

y = train_df[target_col].copy()
train_ids = train_df["id"].copy()
test_ids = test_df["id"].copy()

feature_cols = [c for c in train_df.columns if c not in [target_col, "id"]]
X_train = train_df[feature_cols].copy()
X_test = test_df[[c for c in feature_cols if c in test_df.columns]].copy()

# Ensure target is numeric
y = pd.to_numeric(y, errors="coerce").astype(np.float32)

# Robust Numeric Conversion for Object/String Columns
numeric_cols = []
categorical_cols = []

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(X_train[col]):
        numeric_cols.append(col)
    else:
        # Attempt string-to-numeric coercion
        cleaned_tr = (
            X_train[col]
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        converted_tr = pd.to_numeric(cleaned_tr, errors="coerce")
        orig_null_rate = X_train[col].isna().mean()
        conv_null_rate = converted_tr.isna().mean()

        if (conv_null_rate - orig_null_rate) < 0.4:
            X_train[col] = converted_tr
            if col in X_test.columns:
                cleaned_te = (
                    X_test[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                )
                X_test[col] = pd.to_numeric(cleaned_te, errors="coerce")
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

print(f"Identified {len(numeric_cols)} numeric columns, {len(categorical_cols)} categorical columns.")

# Missingness Indicators
for col in numeric_cols:
    if X_train[col].isna().sum() > 0 or X_test[col].isna().sum() > 0:
        X_train[f"{col}_isna"] = X_train[col].isna().astype(np.float32)
        X_test[f"{col}_isna"] = X_test[col].isna().astype(np.float32)

# Pairwise Interaction Features (Ratios, Differences, Products, Normalized Differences)
if len(numeric_cols) >= 2:
    eps = 1e-5
    for c1, c2 in itertools.combinations(numeric_cols, 2):
        X_train[f"{c1}_add_{c2}"] = X_train[c1] + X_train[c2]
        X_test[f"{c1}_add_{c2}"] = X_test[c1] + X_test[c2]

        X_train[f"{c1}_sub_{c2}"] = X_train[c1] - X_train[c2]
        X_test[f"{c1}_sub_{c2}"] = X_test[c1] - X_test[c2]

        X_train[f"{c1}_mul_{c2}"] = X_train[c1] * X_train[c2]
        X_test[f"{c1}_mul_{c2}"] = X_test[c1] * X_test[c2]

        X_train[f"{c1}_div_{c2}"] = X_train[c1] / (X_train[c2].abs() + eps)
        X_test[f"{c1}_div_{c2}"] = X_test[c1] / (X_test[c2].abs() + eps)

        X_train[f"{c2}_div_{c1}"] = X_train[c2] / (X_train[c1].abs() + eps)
        X_test[f"{c2}_div_{c1}"] = X_test[c2] / (X_test[c1].abs() + eps)

        X_train[f"{c1}_normdiff_{c2}"] = (X_train[c1] - X_train[c2]) / (
            X_train[c1].abs() + X_train[c2].abs() + eps
        )
        X_test[f"{c1}_normdiff_{c2}"] = (X_test[c1] - X_test[c2]) / (
            X_test[c1].abs() + X_test[c2].abs() + eps
        )

# Non-linear Power Transforms & Rank Normalizations
for col in numeric_cols:
    min_val = X_train[col].min()
    if pd.notna(min_val) and min_val >= 0:
        X_train[f"{col}_sqrt"] = np.sqrt(np.maximum(0, X_train[col]))
        X_test[f"{col}_sqrt"] = np.sqrt(np.maximum(0, X_test[col]))

        skew = X_train[col].skew()
        if pd.notna(skew) and skew > 1.0:
            X_train[f"{col}_log1p"] = np.log1p(X_train[col].clip(lower=0))
            X_test[f"{col}_log1p"] = np.log1p(X_test[col].clip(lower=0))

    X_train[f"{col}_sq"] = np.square(X_train[col])
    X_test[f"{col}_sq"] = np.square(X_test[col])

    # Empirical CDF rank normalization
    train_sorted = np.sort(X_train[col].dropna().values)
    if len(train_sorted) > 0:
        X_train[f"{col}_rank_pct"] = np.searchsorted(
            train_sorted, X_train[col].fillna(0).values
        ) / float(len(train_sorted))
        X_test[f"{col}_rank_pct"] = np.searchsorted(
            train_sorted, X_test[col].fillna(0).values
        ) / float(len(train_sorted))

# Quantile-Binned Group-Level Aggregations
if len(numeric_cols) >= 2:
    for c_group in numeric_cols:
        valid_vals = X_train[c_group].dropna()
        if len(valid_vals) > 5:
            qs = np.quantile(valid_vals, [0.2, 0.4, 0.6, 0.8])
            train_bin = np.digitize(
                X_train[c_group].fillna(valid_vals.median()), qs
            )
            test_bin = np.digitize(
                X_test[c_group].fillna(valid_vals.median()), qs
            )

            for c_target in numeric_cols:
                if c_target != c_group:
                    grp_mean_map = X_train.groupby(train_bin)[c_target].mean().to_dict()
                    tr_m = pd.Series(train_bin).map(grp_mean_map).fillna(0).values
                    te_m = pd.Series(test_bin).map(grp_mean_map).fillna(0).values

                    X_train[f"{c_target}_diff_by_{c_group}_bin"] = (
                        X_train[c_target].values - tr_m
                    )
                    X_test[f"{c_target}_diff_by_{c_group}_bin"] = (
                        X_test[c_target].values - te_m
                    )

# K-Means Cluster Distance Embeddings (Only if numeric columns exist)
if len(numeric_cols) > 0:
    km_imputer = SimpleImputer(strategy="median")
    km_scaler = StandardScaler()
    X_tr_num_scaled = km_scaler.fit_transform(
        km_imputer.fit_transform(X_train[numeric_cols])
    )
    X_te_num_scaled = km_scaler.transform(
        km_imputer.transform(X_test[numeric_cols])
    )

    n_clusters = min(6, max(2, len(train_df) // 500))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_tr_num_scaled)
    tr_dists = kmeans.transform(X_tr_num_scaled)
    te_dists = kmeans.transform(X_te_num_scaled)

    for k in range(n_clusters):
        X_train[f"cluster_dist_{k}"] = tr_dists[:, k]
        X_test[f"cluster_dist_{k}"] = te_dists[:, k]

    X_train["min_cluster_dist"] = tr_dists.min(axis=1)
    X_test["min_cluster_dist"] = te_dists.min(axis=1)

# Global Summary Aggregations for Numeric Columns
all_num_now = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
X_train["num_row_sum"] = X_train[all_num_now].sum(axis=1)
X_test["num_row_sum"] = X_test[all_num_now].sum(axis=1)

X_train["num_row_mean"] = X_train[all_num_now].mean(axis=1)
X_test["num_row_mean"] = X_test[all_num_now].mean(axis=1)

X_train["num_row_std"] = X_train[all_num_now].std(axis=1).fillna(0)
X_test["num_row_std"] = X_test[all_num_now].std(axis=1).fillna(0)

# 1. Clean categorical columns as string
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str).fillna("__MISSING__")
    X_test[col] = X_test[col].astype(str).fillna("__MISSING__")

# 2. Pairwise 2-way categorical interaction combinations
cat_pairs = list(itertools.combinations(categorical_cols, 2))
all_cat_cols_to_encode = list(categorical_cols)

for c1, c2 in cat_pairs:
    pair_col = f"{c1}_x_{c2}"
    X_train[pair_col] = X_train[c1].astype(str) + "___" + X_train[c2].astype(str)
    X_test[pair_col] = X_test[c1].astype(str) + "___" + X_test[c2].astype(str)
    all_cat_cols_to_encode.append(pair_col)

# 3. Categorical Frequency & Ordinal Encodings for singles and pairs
for col in all_cat_cols_to_encode:
    freq_map = X_train[col].value_counts(normalize=True).to_dict()
    X_train[f"{col}_freq"] = X_train[col].map(freq_map).fillna(0).astype(np.float32)
    X_test[f"{col}_freq"] = X_test[col].map(freq_map).fillna(0).astype(np.float32)

    cat_map = {val: idx for idx, val in enumerate(X_train[col].unique())}
    X_train[f"{col}_ordinal"] = X_train[col].map(cat_map).fillna(-1).astype(np.float32)
    X_test[f"{col}_ordinal"] = X_test[col].map(cat_map).fillna(-1).astype(np.float32)

# 4. Dense Semantic Text Embeddings & Per-Column N-gram TF-IDF SVD
# Extract contextual sentence embeddings if transformers/sentence-transformers available
dense_text_train = X_train[categorical_cols].astype(str).agg(" | ".join, axis=1)
dense_text_test = X_test[categorical_cols].astype(str).agg(" | ".join, axis=1)

transformer_embedded = False
try:
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    st_tr = st_model.encode(dense_text_train.tolist(), show_progress_bar=False, batch_size=128)
    st_te = st_model.encode(dense_text_test.tolist(), show_progress_bar=False, batch_size=128)
    n_st_dim = min(32, st_tr.shape[1])
    svd_st = TruncatedSVD(n_components=n_st_dim, random_state=42)
    st_tr_red = svd_st.fit_transform(st_tr)
    st_te_red = svd_st.transform(st_te)
    for k in range(n_st_dim):
        X_train[f"st_embed_{k}"] = st_tr_red[:, k].astype(np.float32)
        X_test[f"st_embed_{k}"] = st_te_red[:, k].astype(np.float32)
    transformer_embedded = True
except Exception:
    pass

# Per-column word and char n-gram TF-IDF with SVD embeddings
for c_idx, c_name in enumerate(categorical_cols):
    tr_txt = X_train[c_name].astype(str).values
    te_txt = X_test[c_name].astype(str).values

    # Word level
    tfidf_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=1500)
    w_tr = tfidf_w.fit_transform(tr_txt)
    w_te = tfidf_w.transform(te_txt)
    n_c_svd = min(6, w_tr.shape[1] - 1)
    if n_c_svd > 0:
        svd_w = TruncatedSVD(n_components=n_c_svd, random_state=42 + c_idx)
        w_tr_svd = svd_w.fit_transform(w_tr)
        w_te_svd = svd_w.transform(w_te)
        for k in range(n_c_svd):
            X_train[f"svd_w_{c_name}_{k}"] = w_tr_svd[:, k].astype(np.float32)
            X_test[f"svd_w_{c_name}_{k}"] = w_te_svd[:, k].astype(np.float32)

    # Char level
    tfidf_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=1500)
    c_tr = tfidf_c.fit_transform(tr_txt)
    c_te = tfidf_c.transform(te_txt)
    n_char_svd = min(4, c_tr.shape[1] - 1)
    if n_char_svd > 0:
        svd_c = TruncatedSVD(n_components=n_char_svd, random_state=100 + c_idx)
        c_tr_svd = svd_c.fit_transform(c_tr)
        c_te_svd = svd_c.transform(c_te)
        for k in range(n_char_svd):
            X_train[f"svd_char_{c_name}_{k}"] = c_tr_svd[:, k].astype(np.float32)
            X_test[f"svd_char_{c_name}_{k}"] = c_te_svd[:, k].astype(np.float32)

# Global combined text TF-IDF SVD
tfidf_all = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=4000)
tfidf_all_tr = tfidf_all.fit_transform(dense_text_train)
tfidf_all_te = tfidf_all.transform(dense_text_test)
n_global_svd = min(16, tfidf_all_tr.shape[1] - 1)
if n_global_svd > 1:
    svd_g = TruncatedSVD(n_components=n_global_svd, random_state=42)
    g_tr_svd = svd_g.fit_transform(tfidf_all_tr)
    g_te_svd = svd_g.transform(tfidf_all_te)
    for k in range(n_global_svd):
        X_train[f"svd_global_{k}"] = g_tr_svd[:, k].astype(np.float32)
        X_test[f"svd_global_{k}"] = g_te_svd[:, k].astype(np.float32)

# Cleanup infinite values
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# 5. 10-Fold Cross-Validation & Multi-Scale Bayesian Target Encodings (Strictly Leak-Free)
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_ids = np.zeros(len(train_df), dtype=int)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    fold_ids[val_idx] = fold

global_mean = float(y.mean())
global_std = float(y.std())
global_q25 = float(y.quantile(0.25))
global_q75 = float(y.quantile(0.75))
smoothing_scales = [2.0, 5.0, 15.0, 50.0]

for col in all_cat_cols_to_encode:
    # Multi-scale smoothed means
    for m_smooth in smoothing_scales:
        oof_sm = np.zeros(len(train_df), dtype=np.float32)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
            f_cat = X_train[col].iloc[train_idx]
            f_y = y.iloc[train_idx]
            v_cat = X_train[col].iloc[val_idx]

            stats = f_y.groupby(f_cat).agg(["mean", "count"])
            sm_mean = (stats["count"] * stats["mean"] + m_smooth * global_mean) / (stats["count"] + m_smooth)
            oof_sm[val_idx] = v_cat.map(sm_mean).fillna(global_mean).values

        full_stats = y.groupby(X_train[col]).agg(["mean", "count"])
        full_sm = (
            (full_stats["count"] * full_stats["mean"] + m_smooth * global_mean)
            / (full_stats["count"] + m_smooth)
        ).to_dict()

        X_train[f"{col}_te_mean_m{int(m_smooth)}"] = oof_sm
        X_test[f"{col}_te_mean_m{int(m_smooth)}"] = X_test[col].map(full_sm).fillna(global_mean).astype(np.float32)

    # Multi-scale Bayesian standard deviation & quantiles
    oof_std = np.zeros(len(train_df), dtype=np.float32)
    oof_q25 = np.zeros(len(train_df), dtype=np.float32)
    oof_q75 = np.zeros(len(train_df), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
        f_cat = X_train[col].iloc[train_idx]
        f_y = y.iloc[train_idx]
        v_cat = X_train[col].iloc[val_idx]

        cat_g = f_y.groupby(f_cat)
        c_std = cat_g.std().fillna(0.0)
        c_cnt = cat_g.count()
        c_q25 = cat_g.quantile(0.25)
        c_q75 = cat_g.quantile(0.75)

        sm_std = (c_cnt * c_std + 10.0 * global_std) / (c_cnt + 10.0)
        sm_q25 = (c_cnt * c_q25 + 10.0 * global_q25) / (c_cnt + 10.0)
        sm_q75 = (c_cnt * c_q75 + 10.0 * global_q75) / (c_cnt + 10.0)

        oof_std[val_idx] = v_cat.map(sm_std).fillna(global_std).values
        oof_q25[val_idx] = v_cat.map(sm_q25).fillna(global_q25).values
        oof_q75[val_idx] = v_cat.map(sm_q75).fillna(global_q75).values

    full_g = y.groupby(X_train[col])
    f_std = full_g.std().fillna(0.0)
    f_cnt = full_g.count()
    f_q25 = full_g.quantile(0.25)
    f_q75 = full_g.quantile(0.75)

    full_std_map = ((f_cnt * f_std + 10.0 * global_std) / (f_cnt + 10.0)).to_dict()
    full_q25_map = ((f_cnt * f_q25 + 10.0 * global_q25) / (f_cnt + 10.0)).to_dict()
    full_q75_map = ((f_cnt * f_q75 + 10.0 * global_q75) / (f_cnt + 10.0)).to_dict()

    X_train[f"{col}_te_std"] = oof_std
    X_test[f"{col}_te_std"] = X_test[col].map(full_std_map).fillna(global_std).astype(np.float32)

    X_train[f"{col}_te_q25"] = oof_q25
    X_test[f"{col}_te_q25"] = X_test[col].map(full_q25_map).fillna(global_q25).astype(np.float32)

    X_train[f"{col}_te_q75"] = oof_q75
    X_test[f"{col}_te_q75"] = X_test[col].map(full_q75_map).fillna(global_q75).astype(np.float32)

for col in all_cat_cols_to_encode:
    if col in X_train.columns:
        X_train.drop(columns=[col], inplace=True)
    if col in X_test.columns:
        X_test.drop(columns=[col], inplace=True)

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols].astype(np.float32).copy()
X_test = X_test[common_cols].astype(np.float32).copy()

X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_all = y.values.astype(np.float32)
n_train = len(X_train_np)
n_test = len(X_test_np)
n_features = len(common_cols)

print(f"Engineered features: {n_features}, Train rows: {n_train}, Test rows: {n_test}")


# =========================================================================
# 3. Model Architectures & Definitions
# =========================================================================
class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.ln1(self.fc1(x))))
        out = self.drop2(self.act2(self.ln2(self.fc2(out))))
        return residual + out


class TabularResMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(128, 64),
        dropout_rate: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        layers = []
        for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(ResidualBlock(out_d, dropout_rate=dropout_rate))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.body(x)
        return self.head(x).squeeze(-1)


# =========================================================================
# 4. Model 1: LightGBM 5-Fold Training
# =========================================================================
oof_lgb = np.zeros(n_train, dtype=np.float32)
test_lgb_folds = np.zeros((n_splits, n_test), dtype=np.float32)

print("\n--- Training LightGBM Models ---")
for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    X_train_f, y_train_f = X_train_np[train_mask], y_all[train_mask]
    X_val_f, y_val_f = X_train_np[val_mask], y_all[val_mask]
    X_test_f = X_test_np

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
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
        "n_estimators": 2000,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "verbose": -1,
    }

    model_lgb = lgb.LGBMRegressor(**lgb_params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model_lgb.fit(
        X_train_f,
        y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        eval_names=["val"],
        callbacks=callbacks,
    )

    val_preds = model_lgb.predict(X_val_f)
    oof_lgb[val_mask] = val_preds
    test_lgb_folds[fold] = model_lgb.predict(X_test_f)

    fold_score = root_mean_squared_error(y_val_f, val_preds)
    joblib.dump(model_lgb, f"./working/models/lgb_fold{fold}.joblib")
    print(f"LGB Fold {fold} RMSE: {fold_score:.5f}")

lgb_oof_score = root_mean_squared_error(y_all, oof_lgb)
test_lgb = test_lgb_folds.mean(axis=0)
print(f"LGB Full OOF RMSE: {lgb_oof_score:.5f}")

# =========================================================================
# 5. Model 2: XGBoost 5-Fold Training
# =========================================================================
oof_xgb = np.zeros(n_train, dtype=np.float32)
test_xgb_folds = np.zeros((n_splits, n_test), dtype=np.float32)

print("\n--- Training XGBoost Models ---")
for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    X_train_f, y_train_f = X_train_np[train_mask], y_all[train_mask]
    X_val_f, y_val_f = X_train_np[val_mask], y_all[val_mask]
    X_test_f = X_test_np

    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "random_state": 42 + fold,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }

    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)

    val_preds = model_xgb.predict(X_val_f)
    oof_xgb[val_mask] = val_preds
    test_xgb_folds[fold] = model_xgb.predict(X_test_f)

    fold_score = root_mean_squared_error(y_val_f, val_preds)
    joblib.dump(model_xgb, f"./working/models/xgb_fold{fold}.joblib")
    print(f"XGB Fold {fold} RMSE: {fold_score:.5f}")

xgb_oof_score = root_mean_squared_error(y_all, oof_xgb)
test_xgb = test_xgb_folds.mean(axis=0)
print(f"XGB Full OOF RMSE: {xgb_oof_score:.5f}")

# =========================================================================
# 6. Model 3: CatBoost / Alternative Gradient Boosting
# =========================================================================
oof_cat = np.zeros(n_train, dtype=np.float32)
test_cat_folds = np.zeros((n_splits, n_test), dtype=np.float32)

try:
    from catboost import CatBoostRegressor

    has_catboost = True
except ImportError:
    has_catboost = False

print("\n--- Training CatBoost / Gradient Boosting Alternative ---")
for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    X_train_f, y_train_f = X_train_np[train_mask], y_all[train_mask]
    X_val_f, y_val_f = X_train_np[val_mask], y_all[val_mask]
    X_test_f = X_test_np

    if has_catboost:
        cb_model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            iterations=2000,
            random_seed=42 + fold,
            verbose=False,
            early_stopping_rounds=50,
        )
        cb_model.fit(
            X_train_f,
            y_train_f,
            eval_set=(X_val_f, y_val_f),
            verbose=False,
            use_best_model=True,
        )
        val_preds = cb_model.predict(X_val_f)
        test_preds = cb_model.predict(X_test_f)
        joblib.dump(cb_model, f"./working/models/catboost_fold{fold}.joblib")
    else:
        alt_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.02,
            "num_leaves": 20,
            "max_depth": 5,
            "min_child_samples": 30,
            "subsample": 0.75,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "n_estimators": 2000,
            "random_state": 100 + fold,
            "n_jobs": -1,
            "verbose": -1,
        }
        cb_model = lgb.LGBMRegressor(**alt_params)
        cb_model.fit(
            X_train_f,
            y_train_f,
            eval_set=[(X_val_f, y_val_f)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        val_preds = cb_model.predict(X_val_f)
        test_preds = cb_model.predict(X_test_f)
        joblib.dump(cb_model, f"./working/models/lgb_alt_fold{fold}.joblib")

    oof_cat[val_mask] = val_preds
    test_cat_folds[fold] = test_preds
    fold_score = root_mean_squared_error(y_val_f, val_preds)
    print(f"Cat/GBM2 Fold {fold} RMSE: {fold_score:.5f}")

cat_oof_score = root_mean_squared_error(y_all, oof_cat)
test_cat = test_cat_folds.mean(axis=0)
print(f"Cat/GBM2 Full OOF RMSE: {cat_oof_score:.5f}")

# =========================================================================
# 7. Model 4: PyTorch Tabular Residual MLP
# =========================================================================
oof_nn = np.zeros(n_train, dtype=np.float32)
test_nn_folds = np.zeros((n_splits, n_test), dtype=np.float32)

print("\n--- Training Tabular Neural Network ---")
for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    X_tr_raw, y_tr_raw = X_train_np[train_mask], y_all[train_mask]
    X_va_raw, y_va_raw = X_train_np[val_mask], y_all[val_mask]
    X_te_raw = X_test_np

    # Impute and Standard Scale strictly on training fold
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr = scaler.fit_transform(imputer.fit_transform(X_tr_raw))
    X_va = scaler.transform(imputer.transform(X_va_raw))
    X_te = scaler.transform(imputer.transform(X_te_raw))

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr_raw, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va_raw, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model_nn = TabularResMLP(input_dim=X_tr.shape[1], hidden_dims=(128, 64)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=40, eta_min=1e-5
    )

    best_val_rmse = float("inf")
    best_weights = None
    patience = 8
    patience_counter = 0

    for epoch in range(40):
        model_nn.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model_nn(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model_nn.eval()
        val_preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(device)
                pred = model_nn(bx)
                val_preds_list.extend(pred.cpu().numpy().tolist())

        epoch_val_rmse = root_mean_squared_error(y_va_raw, np.array(val_preds_list))
        if epoch_val_rmse < best_val_rmse:
            best_val_rmse = epoch_val_rmse
            best_weights = {
                k: v.cpu().clone() for k, v in model_nn.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best weights
    model_nn.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model_nn.eval()
    with torch.no_grad():
        val_preds = (
            model_nn(torch.tensor(X_va, dtype=torch.float32).to(device)).cpu().numpy()
        )
        test_preds = (
            model_nn(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
        )

    oof_nn[val_mask] = val_preds
    test_nn_folds[fold] = test_preds
    print(f"NN Fold {fold} RMSE: {best_val_rmse:.5f}")
    torch.save(best_weights, f"./working/models/torch_nn_fold{fold}.pt")

nn_oof_score = root_mean_squared_error(y_all, oof_nn)
test_nn = test_nn_folds.mean(axis=0)
print(f"NN Full OOF RMSE: {nn_oof_score:.5f}")

# =========================================================================
# 7.5 Model 5: Random Forest Regressor 10-Fold Training
# =========================================================================
oof_rf = np.zeros(n_train, dtype=np.float32)
test_rf_folds = np.zeros((n_splits, n_test), dtype=np.float32)

print("\n--- Training Random Forest Models ---")
rf_imputer = SimpleImputer(strategy="median")
X_train_imp = rf_imputer.fit_transform(X_train_np)
X_test_imp = rf_imputer.transform(X_test_np)

for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    X_train_f, y_train_f = X_train_imp[train_mask], y_all[train_mask]
    X_val_f, y_val_f = X_train_imp[val_mask], y_all[val_mask]

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42 + fold,
        n_jobs=-1,
    )
    rf_model.fit(X_train_f, y_train_f)

    val_preds = rf_model.predict(X_val_f)
    oof_rf[val_mask] = val_preds
    test_rf_folds[fold] = rf_model.predict(X_test_imp)

    fold_score = root_mean_squared_error(y_val_f, val_preds)
    joblib.dump(rf_model, f"./working/models/rf_fold{fold}.joblib")
    print(f"RF Fold {fold} RMSE: {fold_score:.5f}")

rf_oof_score = root_mean_squared_error(y_all, oof_rf)
test_rf = test_rf_folds.mean(axis=0)
print(f"RF Full OOF RMSE: {rf_oof_score:.5f}")

# =========================================================================
# 7.6 Model 6: Extra Trees Regressor 10-Fold Training
# =========================================================================
oof_et = np.zeros(n_train, dtype=np.float32)
test_et_folds = np.zeros((n_splits, n_test), dtype=np.float32)

print("\n--- Training Extra Trees Models ---")
for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    X_train_f, y_train_f = X_train_imp[train_mask], y_all[train_mask]
    X_val_f, y_val_f = X_train_imp[val_mask], y_all[val_mask]

    et_model = ExtraTreesRegressor(
        n_estimators=350,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_model.fit(X_train_f, y_train_f)

    val_preds = et_model.predict(X_val_f)
    oof_et[val_mask] = val_preds
    test_et_folds[fold] = et_model.predict(X_test_imp)

    fold_score = root_mean_squared_error(y_val_f, val_preds)
    joblib.dump(et_model, f"./working/models/et_fold{fold}.joblib")
    print(f"ET Fold {fold} RMSE: {fold_score:.5f}")

et_oof_score = root_mean_squared_error(y_all, oof_et)
test_et = test_et_folds.mean(axis=0)
print(f"ET Full OOF RMSE: {et_oof_score:.5f}")

# =========================================================================
# 7.7 Model 7: HistGradientBoosting Regressor 10-Fold Training
# =========================================================================
oof_hgb = np.zeros(n_train, dtype=np.float32)
test_hgb_folds = np.zeros((n_splits, n_test), dtype=np.float32)

print("\n--- Training HistGradientBoosting Models ---")
for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    X_train_f, y_train_f = X_train_imp[train_mask], y_all[train_mask]
    X_val_f, y_val_f = X_train_imp[val_mask], y_all[val_mask]

    hgb_model = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.03,
        l2_regularization=1.0,
        min_samples_leaf=20,
        random_state=42 + fold,
    )
    hgb_model.fit(X_train_f, y_train_f)

    val_preds = hgb_model.predict(X_val_f)
    oof_hgb[val_mask] = val_preds
    test_hgb_folds[fold] = hgb_model.predict(X_test_imp)

    fold_score = root_mean_squared_error(y_val_f, val_preds)
    joblib.dump(hgb_model, f"./working/models/hgb_fold{fold}.joblib")
    print(f"HistGB Fold {fold} RMSE: {fold_score:.5f}")

hgb_oof_score = root_mean_squared_error(y_all, oof_hgb)
test_hgb = test_hgb_folds.mean(axis=0)
print(f"HistGB Full OOF RMSE: {hgb_oof_score:.5f}")

# =========================================================================
# 7.8 Model 8: K-Nearest Neighbors Metric Regressor
# =========================================================================
oof_knn = np.zeros(n_train, dtype=np.float32)
test_knn_folds = np.zeros((n_splits, n_test), dtype=np.float32)

print("\n--- Training K-Nearest Neighbors Metric Regressor ---")
for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    knn_scaler = RobustScaler()
    X_tr_knn = knn_scaler.fit_transform(X_train_imp[train_mask])
    X_va_knn = knn_scaler.transform(X_train_imp[val_mask])
    X_te_knn = knn_scaler.transform(X_test_imp)

    knn_model = KNeighborsRegressor(n_neighbors=25, weights="distance", p=2, n_jobs=-1)
    knn_model.fit(X_tr_knn, y_all[train_mask])

    val_preds = knn_model.predict(X_va_knn)
    oof_knn[val_mask] = val_preds
    test_knn_folds[fold] = knn_model.predict(X_te_knn)

    fold_score = root_mean_squared_error(y_all[val_mask], val_preds)
    joblib.dump(knn_model, f"./working/models/knn_fold{fold}.joblib")
    print(f"KNN Fold {fold} RMSE: {fold_score:.5f}")

knn_oof_score = root_mean_squared_error(y_all, oof_knn)
test_knn = test_knn_folds.mean(axis=0)
print(f"KNN Full OOF RMSE: {knn_oof_score:.5f}")

# =========================================================================
# 8. Multi-Model Regularized Meta-Learners & SLSQP Blending
# =========================================================================
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn, oof_rf, oof_et, oof_hgb, oof_knn])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat, test_nn, test_rf, test_et, test_hgb, test_knn])
n_models = oof_matrix.shape[1]

# Strategy A: SLSQP Constrained Non-Negative Weight Optimization with L2 Shrinkage
def ensemble_loss(weights):
    weights = np.array(weights)
    weights = weights / np.maximum(np.sum(weights), 1e-12)
    blended = np.dot(oof_matrix, weights)
    rmse = root_mean_squared_error(y_all, blended)
    l2_reg = 1e-4 * np.sum(weights ** 2)
    return rmse + l2_reg

init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    ensemble_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = (
    opt_res.x / np.sum(opt_res.x) if opt_res.success else np.ones(n_models) / n_models
)

slsqp_oof = np.dot(oof_matrix, best_weights)
slsqp_test = np.dot(test_matrix, best_weights)
slsqp_score = root_mean_squared_error(y_all, slsqp_oof)
print(f"\nSLSQP Blend Weights (LGB, XGB, Cat, NN, RF, ET, HistGB, KNN): {np.round(best_weights, 4)}")
print(f"SLSQP OOF RMSE: {slsqp_score:.5f}")

# Strategy B: RidgeCV Regularized Meta-Learner (Out-of-Fold)
meta_ridge_oof = np.zeros(n_train, dtype=np.float32)
meta_ridge_test_folds = np.zeros((n_splits, n_test), dtype=np.float32)
alphas = np.logspace(-3, 3, 50)

for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    meta_tr_X, meta_tr_y = oof_matrix[train_mask], y_all[train_mask]
    meta_va_X, meta_va_y = oof_matrix[val_mask], y_all[val_mask]

    ridge_meta = RidgeCV(alphas=alphas, cv=5)
    ridge_meta.fit(meta_tr_X, meta_tr_y)

    meta_ridge_oof[val_mask] = ridge_meta.predict(meta_va_X)
    meta_ridge_test_folds[fold] = ridge_meta.predict(test_matrix)

meta_ridge_score = root_mean_squared_error(y_all, meta_ridge_oof)
meta_ridge_test = meta_ridge_test_folds.mean(axis=0)
print(f"Ridge Meta-Learner OOF RMSE: {meta_ridge_score:.5f}")

# Strategy C: Bayesian Ridge Meta-Learner (Out-of-Fold)
meta_bayes_oof = np.zeros(n_train, dtype=np.float32)
meta_bayes_test_folds = np.zeros((n_splits, n_test), dtype=np.float32)

for fold in range(n_splits):
    train_mask = fold_ids != fold
    val_mask = fold_ids == fold

    meta_tr_X, meta_tr_y = oof_matrix[train_mask], y_all[train_mask]
    meta_va_X, meta_va_y = oof_matrix[val_mask], y_all[val_mask]

    bayes_meta = BayesianRidge()
    bayes_meta.fit(meta_tr_X, meta_tr_y)

    meta_bayes_oof[val_mask] = bayes_meta.predict(meta_va_X)
    meta_bayes_test_folds[fold] = bayes_meta.predict(test_matrix)

meta_bayes_score = root_mean_squared_error(y_all, meta_bayes_oof)
meta_bayes_test = meta_bayes_test_folds.mean(axis=0)
print(f"Bayesian Ridge Meta-Learner OOF RMSE: {meta_bayes_score:.5f}")

# Select the Best Ensembling / Stacking Strategy
candidates = {
    "slsqp": (slsqp_score, slsqp_oof, slsqp_test),
    "ridge": (meta_ridge_score, meta_ridge_oof, meta_ridge_test),
    "bayes": (meta_bayes_score, meta_bayes_oof, meta_bayes_test),
}
best_strategy_name = min(candidates, key=lambda k: candidates[k][0])
best_oof_score, final_oof_preds, final_test_preds = candidates[best_strategy_name]
print(f"\nSelected Best Meta-Strategy: {best_strategy_name.upper()} (OOF RMSE: {best_oof_score:.5f})")

final_val_score = float(best_oof_score)

# =========================================================================
# 9. Submission Output and Validation Checks
# =========================================================================
sub_df = pd.DataFrame({"id": test_ids.values, "Overall": final_test_preds})

sub_path = "./submission/submission.csv"
sub_df.to_csv(sub_path, index=False)

assert os.path.exists(sub_path), f"Submission file {sub_path} was not created!"
assert len(sub_df) == n_test, f"Expected {n_test} rows, found {len(sub_df)}"
assert list(sub_df.columns) == [
    "id",
    "Overall",
], f"Invalid columns: {list(sub_df.columns)}"
assert not sub_df["Overall"].isna().any(), "Submission contains NaN values!"

print(
    f"Submission generated successfully at {sub_path} with {len(sub_df)} predictions."
)
print(f"Final Validation Score: {final_val_score:.6f}")
