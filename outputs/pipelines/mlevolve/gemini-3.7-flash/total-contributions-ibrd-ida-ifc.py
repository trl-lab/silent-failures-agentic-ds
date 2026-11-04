import os
import re
import copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =========================================================================
# 1. Directories & Data Ingestion
# =========================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Total Contribution (USD)"
if target_col not in train_df.columns:
    candidates = [
        c
        for c in train_df.columns
        if "contribution" in c.lower() or "total" in c.lower()
    ]
    target_col = candidates[0] if candidates else train_df.columns[-1]

print(f"Target column: {target_col}")
print(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

y_train = train_df[target_col].copy()
train_features = train_df.drop(columns=[target_col]).copy()
test_features = test_df.copy()
if target_col in test_features.columns:
    test_features = test_features.drop(columns=[target_col])

combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# =========================================================================
# 2. Feature Type Detection
# =========================================================================
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in combined.columns:
    if np.issubdtype(combined[col].dtype, np.datetime64):
        date_cols.append(col)
    elif combined[col].dtype == "object":
        sample_non_null = combined[col].dropna().astype(str).head(50)
        is_date = False
        if len(sample_non_null) > 0:
            try:
                pd.to_datetime(sample_non_null, errors="raise")
                is_date = True
            except (ValueError, TypeError, OverflowError):
                is_date = False

        if is_date:
            date_cols.append(col)
        else:
            avg_word_count = (
                sample_non_null.apply(lambda x: len(x.split())).mean()
                if len(sample_non_null) > 0
                else 0
            )
            unique_ratio = combined[col].nunique() / max(1, len(combined[col].dropna()))
            if avg_word_count > 3 or (
                unique_ratio > 0.6 and combined[col].nunique() > 100
            ):
                text_cols.append(col)
            else:
                cat_cols.append(col)
    else:
        num_cols.append(col)

print(
    f"Detected columns - Numeric: {num_cols}, Date: {date_cols}, Categorical: {cat_cols}, Text: {text_cols}"
)


# =========================================================================
# 3. Feature Extraction Functions
# =========================================================================
def extract_date_features(df, date_columns):
    df_out = pd.DataFrame(index=df.index)
    for col in date_columns:
        dt_series = pd.to_datetime(df[col], errors="coerce")
        clean_col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
        df_out[f"{clean_col}_year"] = dt_series.dt.year.fillna(-1).astype(np.float32)
        df_out[f"{clean_col}_month"] = dt_series.dt.month.fillna(-1).astype(np.float32)
        df_out[f"{clean_col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(
            np.float32
        )
        df_out[f"{clean_col}_day"] = dt_series.dt.day.fillna(-1).astype(np.float32)
        df_out[f"{clean_col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
            np.float32
        )
        df_out[f"{clean_col}_is_month_end"] = dt_series.dt.is_month_end.fillna(
            False
        ).astype(np.float32)

        valid_mask = dt_series.notna()
        month_rad = 2 * np.pi * dt_series.dt.month / 12.0
        df_out[f"{clean_col}_sin_month"] = np.where(
            valid_mask, np.sin(month_rad), 0.0
        ).astype(np.float32)
        df_out[f"{clean_col}_cos_month"] = np.where(
            valid_mask, np.cos(month_rad), 0.0
        ).astype(np.float32)

        min_dt = dt_series.min()
        if pd.notna(min_dt):
            df_out[f"{clean_col}_elapsed_days"] = (
                ((dt_series - min_dt).dt.total_seconds() / (24 * 3600))
                .fillna(-1)
                .astype(np.float32)
            )
        else:
            df_out[f"{clean_col}_elapsed_days"] = -1.0
    return df_out


def extract_numeric_features(df, num_columns):
    df_out = pd.DataFrame(index=df.index)
    for col in num_columns:
        clean_col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
        vals = pd.to_numeric(df[col], errors="coerce")
        df_out[f"{clean_col}_raw"] = vals.astype(np.float32)
        df_out[f"{clean_col}_isnan"] = vals.isna().astype(np.float32)
        pos_vals = np.maximum(0, vals.fillna(0).values)
        df_out[f"{clean_col}_log1p"] = np.log1p(pos_vals).astype(np.float32)

    for i in range(len(num_columns)):
        for j in range(i + 1, len(num_columns)):
            col1, col2 = num_columns[i], num_columns[j]
            c1_clean = re.sub(r"[^a-zA-Z0-9_]", "_", col1)
            c2_clean = re.sub(r"[^a-zA-Z0-9_]", "_", col2)
            v1 = pd.to_numeric(df[col1], errors="coerce").fillna(0).values
            v2 = pd.to_numeric(df[col2], errors="coerce").fillna(0).values
            df_out[f"{c1_clean}_add_{c2_clean}"] = (v1 + v2).astype(np.float32)
            df_out[f"{c1_clean}_diff_{c2_clean}"] = (v1 - v2).astype(np.float32)
            df_out[f"{c1_clean}_ratio_{c2_clean}"] = np.where(
                np.abs(v2) > 1e-5, v1 / (v2 + 1e-5), 0.0
            ).astype(np.float32)
    return df_out


def extract_categorical_features(df, cat_columns, freq_maps=None):
    df_out = pd.DataFrame(index=df.index)
    freq_dict = {} if freq_maps is None else freq_maps

    for col in cat_columns:
        clean_col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
        series_str = df[col].astype(str).str.strip().str.lower().fillna("missing")

        if freq_maps is None:
            freq_map = series_str.value_counts(normalize=True).to_dict()
            freq_dict[col] = freq_map
        else:
            freq_map = freq_dict.get(col, {})

        df_out[f"{clean_col}_freq"] = (
            series_str.map(freq_map).fillna(0.0).astype(np.float32)
        )
        df_out[f"{clean_col}_cat_code"] = series_str.astype(
            "category"
        ).cat.codes.astype(np.int32)

    return df_out, freq_dict


# =========================================================================
# 4. Extract Base Features
# =========================================================================
date_feats_train = extract_date_features(train_features, date_cols)
date_feats_test = extract_date_features(test_features, date_cols)

num_feats_train = extract_numeric_features(train_features, num_cols)
num_feats_test = extract_numeric_features(test_features, num_cols)

cat_feats_train, fitted_freq_maps = extract_categorical_features(
    train_features, cat_cols
)
cat_feats_test, _ = extract_categorical_features(
    test_features, cat_cols, freq_maps=fitted_freq_maps
)

# Text TF-IDF + SVD
text_feats_train = pd.DataFrame(index=train_features.index)
text_feats_test = pd.DataFrame(index=test_features.index)

for col in text_cols:
    clean_col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
    train_text = train_features[col].fillna("").astype(str).values
    test_text = test_features[col].fillna("").astype(str).values

    tfidf = TfidfVectorizer(
        max_features=500, stop_words="english", token_pattern=r"(?u)\b\w+\b"
    )
    svd = TruncatedSVD(
        n_components=min(8, max(2, len(train_text) - 1)), random_state=42
    )

    tfidf_train = tfidf.fit_transform(train_text)
    tfidf_test = tfidf.transform(test_text)

    svd_train = svd.fit_transform(tfidf_train)
    svd_test = svd.transform(tfidf_test)

    for comp_idx in range(svd_train.shape[1]):
        text_feats_train[f"{clean_col}_svd_{comp_idx}"] = svd_train[:, comp_idx].astype(
            np.float32
        )
        text_feats_test[f"{clean_col}_svd_{comp_idx}"] = svd_test[:, comp_idx].astype(
            np.float32
        )

# Cross-Validation Setup & Leak-Free Target Encodings
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_te_train = pd.DataFrame(index=train_features.index)
te_test = pd.DataFrame(index=test_features.index)

smoothing_weight = 10.0
global_target_mean = float(y_train.mean())

for col in cat_cols:
    clean_col = re.sub(r"[^a-zA-Z0-9_]", "_", col)
    oof_series = pd.Series(index=train_features.index, dtype=np.float32)
    train_cats = (
        train_features[col].astype(str).str.strip().str.lower().fillna("missing")
    )
    test_cats = test_features[col].astype(str).str.strip().str.lower().fillna("missing")
    test_accum = np.zeros(len(test_features), dtype=np.float64)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(train_features, y_train)):
        trn_c, val_c = train_cats.iloc[trn_idx], train_cats.iloc[val_idx]
        trn_y = y_train.iloc[trn_idx]

        fold_stats = trn_y.groupby(trn_c).agg(["count", "mean"])
        fold_counts = fold_stats["count"]
        fold_means = fold_stats["mean"]

        smoothed = (
            fold_counts * fold_means + smoothing_weight * global_target_mean
        ) / (fold_counts + smoothing_weight)
        oof_series.iloc[val_idx] = (
            val_c.map(smoothed).fillna(global_target_mean).astype(np.float32)
        )

        test_fold_te = test_cats.map(smoothed).fillna(global_target_mean).values
        test_accum += test_fold_te / n_splits

    oof_te_train[f"{clean_col}_te_mean"] = oof_series
    te_test[f"{clean_col}_te_mean"] = test_accum.astype(np.float32)

# Full matrix assembly
X_train_full = pd.concat(
    [
        num_feats_train,
        date_feats_train,
        cat_feats_train,
        text_feats_train,
        oof_te_train,
    ],
    axis=1,
)
X_test_full = pd.concat(
    [num_feats_test, date_feats_test, cat_feats_test, text_feats_test, te_test], axis=1
)

common_cols = [c for c in X_train_full.columns if c in X_test_full.columns]
X_train_full = X_train_full[common_cols].copy()
X_test_full = X_test_full[common_cols].copy()

train_medians = X_train_full.median()
X_train_full = X_train_full.fillna(train_medians).fillna(0)
X_test_full = X_test_full.fillna(train_medians).fillna(0)
X_train_full = X_train_full.replace([np.inf, -np.inf], 0)
X_test_full = X_test_full.replace([np.inf, -np.inf], 0)

folds = []
for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_full, y_train)):
    folds.append(
        {"fold": fold, "trn_idx": trn_idx.tolist(), "val_idx": val_idx.tolist()}
    )


# =========================================================================
# 5. Model Architectures & Configurations
# =========================================================================
class ResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return residual + out


class TabularResNetRegressor(nn.Module):

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
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def get_lgb_config(seed: int = 42):
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_config(seed: int = 42):
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 2000,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def get_catboost_config(seed: int = 42):
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": False,
        "thread_count": -1,
    }


# =========================================================================
# 6. Model Training & Cross-Validation
# =========================================================================
feature_cols = [c for c in X_train_full.columns if c != target_col]
X_train_df = X_train_full[feature_cols].copy()
X_test_df = X_test_full[feature_cols].copy()
y_train_arr = y_train.values.astype(np.float64)

n_samples = len(X_train_df)
n_test_samples = len(X_test_df)
input_dim = len(feature_cols)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

oof_lgb = np.zeros(n_samples, dtype=np.float64)
oof_xgb = np.zeros(n_samples, dtype=np.float64)
oof_cb = np.zeros(n_samples, dtype=np.float64)
oof_nn = np.zeros(n_samples, dtype=np.float64)

test_preds_lgb = np.zeros(n_test_samples, dtype=np.float64)
test_preds_xgb = np.zeros(n_test_samples, dtype=np.float64)
test_preds_cb = np.zeros(n_test_samples, dtype=np.float64)
test_preds_nn = np.zeros(n_test_samples, dtype=np.float64)

n_folds = len(folds)

for fold_info in folds:
    fold = fold_info["fold"]
    trn_idx = np.array(fold_info["trn_idx"])
    val_idx = np.array(fold_info["val_idx"])

    X_tr_fold = X_train_df.iloc[trn_idx]
    y_tr_fold = y_train_arr[trn_idx]
    X_val_fold = X_train_df.iloc[val_idx]
    y_val_fold = y_train_arr[val_idx]

    # 1. LightGBM
    lgb_model = lgb.LGBMRegressor(**get_lgb_config(seed=42 + fold))
    lgb_model.fit(
        X_tr_fold,
        y_tr_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_pred_lgb = lgb_model.predict(X_val_fold)
    oof_lgb[val_idx] = val_pred_lgb
    test_preds_lgb += lgb_model.predict(X_test_df) / n_folds

    # 2. XGBoost
    xgb_config = get_xgb_config(seed=42 + fold)
    xgb_model = xgb.XGBRegressor(**xgb_config, early_stopping_rounds=50)
    xgb_model.fit(
        X_tr_fold,
        y_tr_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False,
    )
    val_pred_xgb = xgb_model.predict(X_val_fold)
    oof_xgb[val_idx] = val_pred_xgb
    test_preds_xgb += xgb_model.predict(X_test_df) / n_folds

    # 3. CatBoost
    cb_model = CatBoostRegressor(**get_catboost_config(seed=42 + fold))
    cb_model.fit(
        X_tr_fold,
        y_tr_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_pred_cb = cb_model.predict(X_val_fold)
    oof_cb[val_idx] = val_pred_cb
    test_preds_cb += cb_model.predict(X_test_df) / n_folds

    # 4. PyTorch Tabular ResNet
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_fold).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_fold).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_df).astype(np.float32)

    t_X_tr = torch.tensor(X_tr_scaled, dtype=torch.float32)
    t_y_tr = torch.tensor(y_tr_fold, dtype=torch.float32)
    t_X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    t_y_val = torch.tensor(y_val_fold, dtype=torch.float32).to(device)
    t_X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(t_X_tr, t_y_tr)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    nn_model = TabularResNetRegressor(
        input_dim=input_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.15
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        nn_model.parameters(), lr=2e-3, weight_decay=1e-4, eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )

    best_val_rmse = float("inf")
    best_weights = None
    patience = 20
    patience_counter = 0
    epochs = 120

    for epoch in range(epochs):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_out = nn_model(t_X_val)
            val_rmse = torch.sqrt(torch.mean((val_out - t_y_val) ** 2) + 1e-6).item()

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights = copy.deepcopy(nn_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict(best_weights)
    nn_model.eval()
    with torch.no_grad():
        val_pred_nn = nn_model(t_X_val).cpu().numpy()
        test_pred_nn_fold = nn_model(t_X_test).cpu().numpy()

    oof_nn[val_idx] = val_pred_nn
    test_preds_nn += test_pred_nn_fold / n_folds

# Non-Negative Floor
floor_val = max(0.0, float(np.min(y_train_arr)))

oof_lgb = np.clip(oof_lgb, floor_val, None)
oof_xgb = np.clip(oof_xgb, floor_val, None)
oof_cb = np.clip(oof_cb, floor_val, None)
oof_nn = np.clip(oof_nn, floor_val, None)

test_preds_lgb = np.clip(test_preds_lgb, floor_val, None)
test_preds_xgb = np.clip(test_preds_xgb, floor_val, None)
test_preds_cb = np.clip(test_preds_cb, floor_val, None)
test_preds_nn = np.clip(test_preds_nn, floor_val, None)

# =========================================================================
# 7. Optimal Constrained Ensemble Blending
# =========================================================================
model_oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_nn])
model_test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_nn]
)


def blending_objective(weights: np.ndarray) -> float:
    w = weights / np.sum(weights)
    blend = np.dot(model_oof_matrix, w)
    return np.sqrt(np.mean((blend - y_train_arr) ** 2))


init_weights = np.array([0.35, 0.35, 0.20, 0.10], dtype=np.float64)
bounds = [(0.0, 1.0) for _ in range(4)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

opt_result = minimize(
    blending_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_result.x / np.sum(opt_result.x)

blended_oof = np.clip(np.dot(model_oof_matrix, best_weights), floor_val, None)
blended_test_preds = np.clip(np.dot(model_test_matrix, best_weights), floor_val, None)

# =========================================================================
# 8. Verification & Submission Generation
# =========================================================================
sub_df = pd.DataFrame(
    {"id": np.arange(len(blended_test_preds)), target_col: blended_test_preds}
)

submission_path = "./submission/submission.csv"
sub_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created!"
assert (
    len(sub_df) == n_test_samples
), f"Submission length mismatch: {len(sub_df)} vs {n_test_samples}"
assert list(sub_df.columns) == [
    "id",
    target_col,
], f"Invalid submission columns: {list(sub_df.columns)}"
assert sub_df[target_col].isna().sum() == 0, "Submission contains NaN values!"

final_score = np.sqrt(np.mean((blended_oof - y_train_arr) ** 2))
print(f"Final Validation Score: {final_score}")
