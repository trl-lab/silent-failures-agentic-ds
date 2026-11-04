import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

# =========================================================================
# 0. Global Setup & Seed
# =========================================================================
np.random.seed(42)

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Data Ingestion & Target Standardization
# =========================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "num_countries_affected"
if target_col not in train_df.columns:
    for col in train_df.columns:
        if col.lower() == target_col.lower():
            train_df.rename(columns={col: target_col}, inplace=True)
            break

def encode_target(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() == 2:
            unique_sorted = np.sort(series.dropna().unique())
            return (series == unique_sorted[1]).astype(int).values
        else:
            med = series.median()
            return (series > med).astype(int).values

    series_str = series.astype(str).str.strip().str.lower()
    unique_vals = list(series_str.unique())

    if len(unique_vals) == 2:
        pos_keywords = ["more", "multi", "several", "true", "yes", "high", ">", "2"]
        val0, val1 = unique_vals[0], unique_vals[1]

        val0_is_pos = any(kw in val0 for kw in pos_keywords)
        val1_is_pos = any(kw in val1 for kw in pos_keywords)

        if val1_is_pos and not val0_is_pos:
            pos_val = val1
        elif val0_is_pos and not val1_is_pos:
            pos_val = val0
        else:
            neg_keywords = ["one", "single", "0", "false", "no", "low", "none", "1"]
            val0_is_neg = any(kw in val0 for kw in neg_keywords)
            val1_is_neg = any(kw in val1 for kw in neg_keywords)
            if val0_is_neg and not val1_is_neg:
                pos_val = val1
            elif val1_is_neg and not val0_is_neg:
                pos_val = val0
            else:
                pos_val = sorted(unique_vals)[1]
        return (series_str == pos_val).astype(int).values
    else:
        try:
            num_s = pd.to_numeric(series_str, errors="coerce")
            if num_s.notna().sum() > len(series_str) * 0.5:
                med = num_s.median()
                return (num_s > med).astype(int).values
        except Exception:
            pass
        pos_mask = series_str.apply(
            lambda s: any(
                kw in s
                for kw in [
                    "more",
                    "multi",
                    "several",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    ">",
                ]
            )
        )
        if pos_mask.nunique() == 2:
            return pos_mask.astype(int).values
        return (pd.factorize(series_str)[0] > 0).astype(int)


train_df[target_col] = encode_target(train_df[target_col])
y_train = train_df[target_col].values

id_col = None
for col in ["id", "ID", "Id", "index", "notification_id", "reference"]:
    if col in train_df.columns and col in test_df.columns:
        id_col = col
        break

train_features = train_df.drop(columns=[target_col]).copy()
if target_col in test_df.columns:
    test_features = test_df.drop(columns=[target_col]).copy()
else:
    test_features = test_df.copy()

train_features["__is_train__"] = 1
test_features["__is_train__"] = 0

combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# =========================================================================
# 2. Date & Temporal Feature Engineering
# =========================================================================
datetime_cols = []
for col in combined.columns:
    if col in ["__is_train__", id_col]:
        continue
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        datetime_cols.append(col)
    elif combined[col].dtype == "object" or str(combined[col].dtype).startswith(
        "string"
    ):
        sample = combined[col].dropna().head(100)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample)
                datetime_cols.append(col)
            except (ValueError, TypeError):
                pass

for col in datetime_cols:
    dt = pd.to_datetime(combined[col], errors="coerce")
    combined[f"{col}_year"] = dt.dt.year.fillna(-1).astype(int)
    combined[f"{col}_month"] = dt.dt.month.fillna(-1).astype(int)
    combined[f"{col}_day"] = dt.dt.day.fillna(-1).astype(int)
    combined[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(-1).astype(int)
    combined[f"{col}_dayofyear"] = dt.dt.dayofyear.fillna(-1).astype(int)
    combined[f"{col}_quarter"] = dt.dt.quarter.fillna(-1).astype(int)
    combined[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)

    combined[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0).fillna(0)
    combined[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0).fillna(0)
    combined[f"{col}_dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0).fillna(0)
    combined[f"{col}_dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0).fillna(0)

    combined.drop(columns=[col], inplace=True)

# =========================================================================
# 3. Text Feature Extraction & SVD Semantic Embedding
# =========================================================================
text_cols = []
for col in combined.columns:
    if col in ["__is_train__", id_col]:
        continue
    if combined[col].dtype == "object" or str(combined[col].dtype).startswith("string"):
        avg_len = combined[col].dropna().astype(str).str.len().mean()
        nunique = combined[col].nunique()
        if avg_len > 30 or nunique > 500:
            text_cols.append(col)

for col in text_cols:
    str_series = combined[col].fillna("").astype(str)
    combined[f"{col}_char_len"] = str_series.str.len()
    combined[f"{col}_word_count"] = str_series.str.split().str.len()
    combined[f"{col}_digit_count"] = str_series.apply(
        lambda s: sum(c.isdigit() for c in s)
    )
    combined[f"{col}_upper_count"] = str_series.apply(
        lambda s: sum(c.isupper() for c in s)
    )
    combined[f"{col}_comma_count"] = str_series.str.count(",")
    combined[f"{col}_semicolon_count"] = str_series.str.count(";")
    combined[f"{col}_slash_count"] = str_series.str.count("/")
    combined[f"{col}_plus_count"] = str_series.str.count(r"\+")
    combined[f"{col}_and_count"] = str_series.str.count(r"(?i)\band\b")

    from scipy.sparse import hstack

    train_text = str_series[combined["__is_train__"] == 1]
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b",
    )
    tfidf_char = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char_wb",
        max_features=2500,
    )

    tfidf_word.fit(train_text)
    tfidf_char.fit(train_text)

    full_word = tfidf_word.transform(str_series)
    full_char = tfidf_char.transform(str_series)
    full_tfidf = hstack([full_word, full_char])

    n_components = min(25, full_tfidf.shape[1] - 1)
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        train_mask = (combined["__is_train__"] == 1).values
        svd.fit(full_tfidf.tocsr()[train_mask])
        svd_features = svd.transform(full_tfidf)
        for comp_idx in range(n_components):
            combined[f"{col}_svd_{comp_idx}"] = svd_features[:, comp_idx]

    combined.drop(columns=[col], inplace=True)

# =========================================================================
# 4. Categorical & Interaction Features
# =========================================================================
categorical_cols = []
for col in combined.columns:
    if col in ["__is_train__", id_col]:
        continue
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        categorical_cols.append(col)
    elif combined[col].nunique() < 50 and not np.issubdtype(
        combined[col].dtype, np.floating
    ):
        categorical_cols.append(col)

for col in categorical_cols:
    freq_map = (
        combined.loc[combined["__is_train__"] == 1, col]
        .value_counts(normalize=True)
        .to_dict()
    )
    combined[f"{col}_freq"] = combined[col].map(freq_map).fillna(0)

top_cats = sorted(categorical_cols, key=lambda c: combined[c].nunique())[:6]
inter_cols = []
for i in range(len(top_cats)):
    for j in range(i + 1, len(top_cats)):
        col_a, col_b = top_cats[i], top_cats[j]
        inter_name = f"{col_a}_x_{col_b}"
        inter_series = combined[col_a].astype(str) + "__" + combined[col_b].astype(str)
        combined[inter_name] = inter_series
        inter_freq = (
            inter_series[combined["__is_train__"] == 1]
            .value_counts(normalize=True)
            .to_dict()
        )
        combined[f"{inter_name}_freq"] = inter_series.map(inter_freq).fillna(0)
        inter_cols.append(inter_name)

all_cat_target_enc_cols = categorical_cols + inter_cols

for col in categorical_cols + inter_cols:
    if combined[col].dtype == "object" or combined[col].dtype.name == "category":
        combined[col] = pd.factorize(combined[col].astype(str))[0]

# =========================================================================
# 5. Numeric Imputation & Target Encoding
# =========================================================================
numeric_cols = [c for c in combined.columns if c not in ["__is_train__", id_col]]
for c in numeric_cols:
    if combined[c].isna().sum() > 0:
        combined[f"{c}_isna"] = combined[c].isna().astype(int)
        fill_val = combined.loc[combined["__is_train__"] == 1, c].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        combined[c] = combined[c].fillna(fill_val)

train_processed = (
    combined[combined["__is_train__"] == 1]
    .drop(columns=["__is_train__"])
    .reset_index(drop=True)
)
test_processed = (
    combined[combined["__is_train__"] == 0]
    .drop(columns=["__is_train__"])
    .reset_index(drop=True)
)

train_processed[target_col] = y_train

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_processed["fold"] = -1

for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(train_processed, y_train)):
    train_processed.loc[val_idx, "fold"] = fold_idx

for col in all_cat_target_enc_cols:
    te_col_name = f"{col}_target_enc"
    train_processed[te_col_name] = 0.0
    test_processed[te_col_name] = 0.0

    global_mean = y_train.mean()

    for fold_idx in range(5):
        trn_mask = train_processed["fold"] != fold_idx
        val_mask = train_processed["fold"] == fold_idx

        fold_means = (
            train_processed.loc[trn_mask]
            .groupby(col)[target_col]
            .agg(["count", "mean"])
        )
        smoothing = 10
        smoothed_means = (
            fold_means["count"] * fold_means["mean"] + smoothing * global_mean
        ) / (fold_means["count"] + smoothing)

        train_processed.loc[val_mask, te_col_name] = (
            train_processed.loc[val_mask, col].map(smoothed_means).fillna(global_mean)
        )

    full_means = train_processed.groupby(col)[target_col].agg(["count", "mean"])
    smoothed_full = (full_means["count"] * full_means["mean"] + 10 * global_mean) / (
        full_means["count"] + 10
    )
    test_processed[te_col_name] = (
        test_processed[col].map(smoothed_full).fillna(global_mean)
    )


# =========================================================================
# 6. Model Architectures & Configurations
# =========================================================================
class TabularResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class TabularDeepModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(
                TabularResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout_rate)
            )
        self.res_blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        logits = self.head(x)
        return logits.squeeze(-1)


def build_nn_components(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = TabularDeepModel(num_features=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def get_lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "min_child_samples": 20,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "min_child_weight": 3,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


# =========================================================================
# 7. Stratified 5-Fold Cross-Validation Training & Evaluation
# =========================================================================
ignore_cols = [target_col, "fold", "__is_train__"]
if id_col is not None:
    ignore_cols.append(id_col)

feature_cols = [c for c in train_processed.columns if c not in ignore_cols]
n_features = len(feature_cols)

y_all = train_processed[target_col].values.astype(np.float32)
X_test_raw = test_processed[feature_cols].values.astype(np.float32)

oof_preds = np.zeros(len(train_processed), dtype=np.float64)
test_preds = np.zeros(len(test_processed), dtype=np.float64)

n_splits = 5

for fold_idx in range(n_splits):
    trn_mask = train_processed["fold"] != fold_idx
    val_mask = train_processed["fold"] == fold_idx

    X_trn = train_processed.loc[trn_mask, feature_cols].values.astype(np.float32)
    y_trn = y_all[trn_mask]
    X_val = train_processed.loc[val_mask, feature_cols].values.astype(np.float32)
    y_val = y_all[val_mask]

    # --- LightGBM ---
    lgb_train_data = lgb.Dataset(X_trn, label=y_trn)
    lgb_val_data = lgb.Dataset(X_val, label=y_val, reference=lgb_train_data)
    lgb_params = get_lgb_params(seed=42 + fold_idx)
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        num_boost_round=1200,
        valid_sets=[lgb_val_data],
        callbacks=lgb_callbacks,
    )

    lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    lgb_test_pred = lgb_model.predict(
        X_test_raw, num_iteration=lgb_model.best_iteration
    )

    # --- XGBoost ---
    dtrain = xgb.DMatrix(X_trn, label=y_trn)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test_raw)
    xgb_params = get_xgb_params(seed=42 + fold_idx)

    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1200,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    xgb_val_pred = xgb_model.predict(dval)
    xgb_test_pred = xgb_model.predict(dtest)

    # --- PyTorch Tabular Deep Model ---
    scaler = StandardScaler()
    X_trn_scaled = np.nan_to_num(scaler.fit_transform(X_trn), 0.0)
    X_val_scaled = np.nan_to_num(scaler.transform(X_val), 0.0)
    X_test_scaled = np.nan_to_num(scaler.transform(X_test_raw), 0.0)

    trn_dataset = TensorDataset(
        torch.tensor(X_trn_scaled, dtype=torch.float32),
        torch.tensor(y_trn, dtype=torch.float32),
    )
    trn_loader = DataLoader(trn_dataset, batch_size=256, shuffle=True, drop_last=False)

    nn_model, criterion, optimizer, scheduler = build_nn_components(
        input_dim=n_features, lr=1e-3, weight_decay=1e-4, device=str(device)
    )

    best_nn_val_auc = -1.0
    best_nn_val_pred = None
    best_nn_test_pred = None

    epochs = 25
    for epoch in range(epochs):
        nn_model.train()
        for batch_x, batch_y in trn_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            val_probs = torch.sigmoid(nn_model(val_t)).cpu().numpy()
            curr_val_auc = roc_auc_score(y_val, val_probs)

            if curr_val_auc > best_nn_val_auc:
                best_nn_val_auc = curr_val_auc
                best_nn_val_pred = val_probs
                test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
                best_nn_test_pred = torch.sigmoid(nn_model(test_t)).cpu().numpy()

    # --- Blending ---
    fold_val_blend = 0.45 * lgb_val_pred + 0.45 * xgb_val_pred + 0.10 * best_nn_val_pred
    fold_test_blend = (
        0.45 * lgb_test_pred + 0.45 * xgb_test_pred + 0.10 * best_nn_test_pred
    )

    oof_preds[val_mask] = fold_val_blend
    test_preds += fold_test_blend / n_splits

    fold_lgb_auc = roc_auc_score(y_val, lgb_val_pred)
    fold_xgb_auc = roc_auc_score(y_val, xgb_val_pred)
    fold_blend_auc = roc_auc_score(y_val, fold_val_blend)

    print(
        f"Fold {fold_idx + 1}/{n_splits} - LGB AUC: {fold_lgb_auc:.5f} | XGB AUC: {fold_xgb_auc:.5f} | NN AUC: {best_nn_val_auc:.5f} | Blend AUC: {fold_blend_auc:.5f}"
    )

# =========================================================================
# 8. Evaluation & Submission Generation
# =========================================================================
final_score = roc_auc_score(y_all, oof_preds)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_processed)), target_col: test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")