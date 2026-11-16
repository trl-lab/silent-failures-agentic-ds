import copy
import os
import re
from typing import Any, Dict, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ==============================================================================
# 0. Setup Directories and Seeds
# ==============================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ==============================================================================
# 1. Load Data and Split First to Prevent Leakage
# ==============================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "ACA FUL"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' not found in train data."

y_all = train_df[target_col].copy()
X_all = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

if "id" in X_test_raw.columns:
    test_ids = X_test_raw["id"].values
else:
    test_ids = np.arange(len(X_test_raw))

train_idx, val_idx = train_test_split(
    np.arange(len(X_all)), test_size=0.2, random_state=42, shuffle=True
)

X_tr = X_all.iloc[train_idx].copy().reset_index(drop=True)
y_tr = y_all.iloc[train_idx].copy().reset_index(drop=True)
X_va = X_all.iloc[val_idx].copy().reset_index(drop=True)
y_va = y_all.iloc[val_idx].copy().reset_index(drop=True)
X_te = X_test_raw.copy().reset_index(drop=True)


# ==============================================================================
# 2. Feature Engineering & Preprocessing
# ==============================================================================
def extract_regex_features(series: pd.Series) -> pd.DataFrame:
    """Extract numeric dosage values, packaging counts, and string properties."""
    s = series.astype(str)
    dosage_val = s.str.extract(
        r"(\d+\.?\d*)\s*(?:mg|ml|mcg|ug|g|iu|unit|meq|%|tab|cap)", flags=re.IGNORECASE
    )[0]
    dosage_val = pd.to_numeric(dosage_val, errors="coerce").fillna(0.0)

    pkg_val = s.str.extract(
        r"(\d+)\s*(?:count|ct|pack|pk|capsules|tabs|tablets|bottles|vials)",
        flags=re.IGNORECASE,
    )[0]
    pkg_val = pd.to_numeric(pkg_val, errors="coerce").fillna(0.0)

    char_len = s.str.len().astype(np.float32)
    word_cnt = s.str.split().apply(len).astype(np.float32)
    digit_cnt = s.apply(lambda x: sum(c.isdigit() for c in x)).astype(np.float32)

    return pd.DataFrame(
        {
            f"{series.name}_dosage_val": dosage_val.astype(np.float32),
            f"{series.name}_pkg_val": pkg_val.astype(np.float32),
            f"{series.name}_char_len": char_len,
            f"{series.name}_word_cnt": word_cnt,
            f"{series.name}_digit_cnt": digit_cnt,
        },
        index=series.index,
    )


processed_tr = pd.DataFrame(index=X_tr.index)
processed_va = pd.DataFrame(index=X_va.index)
processed_te = pd.DataFrame(index=X_te.index)

numeric_cols = []
categorical_cols = []
text_cols = []
datetime_cols = []

for col in X_tr.columns:
    if col.lower() == "id":
        continue
    if pd.api.types.is_datetime64_any_dtype(X_tr[col]):
        datetime_cols.append(col)
    elif pd.api.types.is_numeric_dtype(X_tr[col]):
        numeric_cols.append(col)
    else:
        avg_len = X_tr[col].astype(str).str.len().mean()
        unique_ratio = X_tr[col].nunique() / max(len(X_tr), 1)
        if avg_len > 30 or unique_ratio > 0.5:
            text_cols.append(col)
        else:
            categorical_cols.append(col)

# 2.1 Process Datetime Columns
for col in datetime_cols:
    for df, target_df in zip(
        [X_tr, X_va, X_te], [processed_tr, processed_va, processed_te]
    ):
        dt_s = pd.to_datetime(df[col], errors="coerce")
        target_df[f"{col}_year"] = dt_s.dt.year.fillna(-1).astype(np.float32)
        target_df[f"{col}_month"] = dt_s.dt.month.fillna(-1).astype(np.float32)
        target_df[f"{col}_day"] = dt_s.dt.day.fillna(-1).astype(np.float32)
        target_df[f"{col}_dayofweek"] = dt_s.dt.dayofweek.fillna(-1).astype(np.float32)
        target_df[f"{col}_quarter"] = dt_s.dt.quarter.fillna(-1).astype(np.float32)
        target_df[f"{col}_month_sin"] = (
            np.sin(2 * np.pi * dt_s.dt.month / 12.0).fillna(0.0).astype(np.float32)
        )
        target_df[f"{col}_month_cos"] = (
            np.cos(2 * np.pi * dt_s.dt.month / 12.0).fillna(0.0).astype(np.float32)
        )

# 2.2 Process Text Columns
for col in text_cols:
    tr_regex = extract_regex_features(X_tr[col])
    va_regex = extract_regex_features(X_va[col])
    te_regex = extract_regex_features(X_te[col])

    for c in tr_regex.columns:
        processed_tr[c] = tr_regex[c].values
        processed_va[c] = va_regex[c].values
        processed_te[c] = te_regex[c].values

    tfidf = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"
    )
    tr_text_vec = tfidf.fit_transform(X_tr[col].fillna("").astype(str))
    va_text_vec = tfidf.transform(X_va[col].fillna("").astype(str))
    te_text_vec = tfidf.transform(X_te[col].fillna("").astype(str))

    if tr_text_vec.shape[1] > 1:
        n_comp = min(15, tr_text_vec.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        tr_svd = svd.fit_transform(tr_text_vec)
        va_svd = svd.transform(va_text_vec)
        te_svd = svd.transform(te_text_vec)
        for i in range(n_comp):
            processed_tr[f"{col}_svd_{i}"] = tr_svd[:, i].astype(np.float32)
            processed_va[f"{col}_svd_{i}"] = va_svd[:, i].astype(np.float32)
            processed_te[f"{col}_svd_{i}"] = te_svd[:, i].astype(np.float32)

# 2.3 Process Categorical Columns
for col in categorical_cols:
    s_tr = X_tr[col].fillna("__MISSING__").astype(str)
    s_va = X_va[col].fillna("__MISSING__").astype(str)
    s_te = X_te[col].fillna("__MISSING__").astype(str)

    freq_map = s_tr.value_counts(normalize=True).to_dict()
    processed_tr[f"{col}_freq"] = s_tr.map(freq_map).fillna(0.0).astype(np.float32)
    processed_va[f"{col}_freq"] = s_va.map(freq_map).fillna(0.0).astype(np.float32)
    processed_te[f"{col}_freq"] = s_te.map(freq_map).fillna(0.0).astype(np.float32)

    global_mean = float(y_tr.mean())
    oof_target_enc = np.full(len(X_tr), global_mean, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_f_idx, va_f_idx in kf.split(X_tr):
        s_tr_fold = s_tr.iloc[tr_f_idx]
        y_tr_fold = y_tr.iloc[tr_f_idx]
        fold_means = y_tr_fold.groupby(s_tr_fold).mean()
        fold_counts = y_tr_fold.groupby(s_tr_fold).count()
        smooth_weights = 10.0
        smoothed = (fold_means * fold_counts + global_mean * smooth_weights) / (
            fold_counts + smooth_weights
        )
        oof_target_enc[va_f_idx] = (
            s_tr.iloc[va_f_idx].map(smoothed).fillna(global_mean).values
        )

    processed_tr[f"{col}_target_enc"] = oof_target_enc

    train_means = y_tr.groupby(s_tr).mean()
    train_counts = y_tr.groupby(s_tr).count()
    train_smoothed = (train_means * train_counts + global_mean * 10.0) / (
        train_counts + 10.0
    )

    processed_va[f"{col}_target_enc"] = (
        s_va.map(train_smoothed).fillna(global_mean).astype(np.float32)
    )
    processed_te[f"{col}_target_enc"] = (
        s_te.map(train_smoothed).fillna(global_mean).astype(np.float32)
    )

    regex_tr = extract_regex_features(s_tr)
    regex_va = extract_regex_features(s_va)
    regex_te = extract_regex_features(s_te)
    for c in regex_tr.columns:
        processed_tr[f"{col}_{c}"] = regex_tr[c].values
        processed_va[f"{col}_{c}"] = regex_va[c].values
        processed_te[f"{col}_{c}"] = regex_te[c].values

# 2.4 Process Numeric Columns
for col in numeric_cols:
    median_val = X_tr[col].median()
    if pd.isna(median_val):
        median_val = 0.0

    for df, target_df in zip(
        [X_tr, X_va, X_te], [processed_tr, processed_va, processed_te]
    ):
        target_df[f"{col}_isna"] = df[col].isna().astype(np.float32)
        val = df[col].fillna(median_val).astype(np.float32)
        target_df[col] = val
        if (X_tr[col].dropna() >= 0).all():
            target_df[f"{col}_log1p"] = np.log1p(np.maximum(val, 0.0))

# 2.5 Group Aggregations
if len(categorical_cols) > 0 and len(numeric_cols) > 0:
    top_cat = categorical_cols[0]
    for num_c in numeric_cols[:3]:
        grouped_means = X_tr.groupby(X_tr[top_cat].fillna("__MISSING__"))[num_c].mean()
        grouped_stds = (
            X_tr.groupby(X_tr[top_cat].fillna("__MISSING__"))[num_c].std().fillna(0.0)
        )

        for df, target_df in zip(
            [X_tr, X_va, X_te], [processed_tr, processed_va, processed_te]
        ):
            cat_s = df[top_cat].fillna("__MISSING__")
            target_df[f"{top_cat}_{num_c}_grp_mean"] = (
                cat_s.map(grouped_means).fillna(X_tr[num_c].mean()).astype(np.float32)
            )
            target_df[f"{top_cat}_{num_c}_grp_std"] = (
                cat_s.map(grouped_stds).fillna(0.0).astype(np.float32)
            )

processed_tr = (
    processed_tr.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
)
processed_va = (
    processed_va.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
)
processed_te = (
    processed_te.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
)

y_tr_arr = y_tr.values.astype(np.float32)
y_va_arr = y_va.values.astype(np.float32)
input_dim = processed_tr.shape[1]


# ==============================================================================
# 3. Model Architecture Specifications
# ==============================================================================
class TabularResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularMLPRegressor(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
        )
        blocks = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            if in_dim == out_dim:
                blocks.append(TabularResidualBlock(in_dim, dropout_rate))
            else:
                blocks.append(
                    nn.Sequential(
                        nn.LayerNorm(in_dim),
                        nn.Linear(in_dim, out_dim),
                        nn.LeakyReLU(negative_slope=0.01),
                        nn.Dropout(dropout_rate),
                    )
                )
        self.hidden_layers = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        out = self.head(x)
        return out.squeeze(-1)


def build_model(model_type: str, input_dim: int = None, random_state: int = 42) -> Any:
    if model_type == "lgbm":
        return lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=7,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            early_stopping_rounds=50,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "ridge":
        return Ridge(alpha=10.0, random_state=random_state)
    elif model_type == "tabular_mlp":
        assert input_dim is not None, "input_dim must be provided for tabular_mlp"
        return TabularMLPRegressor(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ==============================================================================
# 4. Training & Out-of-Fold / Validation Predictions
# ==============================================================================
# 4.1 LightGBM Regressor
lgb_model = build_model("lgbm", random_state=42)
lgb_model.fit(
    processed_tr,
    y_tr_arr,
    eval_set=[(processed_va, y_va_arr)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
lgb_va_pred = np.maximum(0.0, lgb_model.predict(processed_va))
lgb_te_pred = np.maximum(0.0, lgb_model.predict(processed_te))
lgb_rmse = np.sqrt(mean_squared_error(y_va_arr, lgb_va_pred))

# 4.2 XGBoost Regressor
xgb_model = build_model("xgboost", random_state=42)
xgb_model.fit(
    processed_tr,
    y_tr_arr,
    eval_set=[(processed_va, y_va_arr)],
    verbose=False,
)
xgb_va_pred = np.maximum(0.0, xgb_model.predict(processed_va))
xgb_te_pred = np.maximum(0.0, xgb_model.predict(processed_te))
xgb_rmse = np.sqrt(mean_squared_error(y_va_arr, xgb_va_pred))

# 4.3 Tabular Residual MLP Regressor
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(processed_tr.values)
X_va_scaled = scaler.transform(processed_va.values)
X_te_scaled = scaler.transform(processed_te.values)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = build_model("tabular_mlp", input_dim=input_dim).to(device)

train_dataset = TensorDataset(
    torch.from_numpy(X_tr_scaled).float(), torch.from_numpy(y_tr_arr).float()
)
val_dataset = TensorDataset(
    torch.from_numpy(X_va_scaled).float(), torch.from_numpy(y_va_arr).float()
)

drop_last = len(train_dataset) % 64 == 1
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, drop_last=drop_last
)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=60, eta_min=1e-6
)

best_mlp_rmse = float("inf")
best_mlp_state = None
patience = 15
patience_counter = 0

for epoch in range(1, 61):
    mlp_model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = mlp_model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=2.0)
        optimizer.step()
    scheduler.step()

    mlp_model.eval()
    val_preds_list = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            preds = mlp_model(batch_x)
            val_preds_list.append(preds.cpu().numpy())

    val_preds_arr = np.maximum(0.0, np.concatenate(val_preds_list))
    cur_rmse = np.sqrt(mean_squared_error(y_va_arr, val_preds_arr))

    if cur_rmse < best_mlp_rmse:
        best_mlp_rmse = cur_rmse
        best_mlp_state = copy.deepcopy(mlp_model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break

mlp_model.load_state_dict(best_mlp_state)
mlp_model.eval()

with torch.no_grad():
    mlp_va_pred = np.maximum(
        0.0, mlp_model(torch.from_numpy(X_va_scaled).float().to(device)).cpu().numpy()
    )
    mlp_te_pred = np.maximum(
        0.0, mlp_model(torch.from_numpy(X_te_scaled).float().to(device)).cpu().numpy()
    )

mlp_rmse = np.sqrt(mean_squared_error(y_va_arr, mlp_va_pred))

# 4.4 Ridge Baseline Regressor
ridge_model = build_model("ridge", random_state=42)
ridge_model.fit(X_tr_scaled, y_tr_arr)
ridge_va_pred = np.maximum(0.0, ridge_model.predict(X_va_scaled))
ridge_te_pred = np.maximum(0.0, ridge_model.predict(X_te_scaled))
ridge_rmse = np.sqrt(mean_squared_error(y_va_arr, ridge_va_pred))

# ==============================================================================
# 5. Constrained Optimal Blending & Ensembling
# ==============================================================================
va_preds_matrix = np.column_stack(
    [lgb_va_pred, xgb_va_pred, mlp_va_pred, ridge_va_pred]
)
te_preds_matrix = np.column_stack(
    [lgb_te_pred, xgb_te_pred, mlp_te_pred, ridge_te_pred]
)


def blend_loss(weights):
    w = np.array(weights)
    pred = va_preds_matrix @ w
    return np.sqrt(mean_squared_error(y_va_arr, pred))


init_weights = np.ones(4) / 4.0
bounds = [(0.0, 1.0)] * 4
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    blend_loss,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_weights = opt_res.x if opt_res.success else init_weights

# ==============================================================================
# 6. Validation Score and Submission Generation
# ==============================================================================
final_va_pred = np.maximum(0.0, va_preds_matrix @ opt_weights)
final_val_rmse = np.sqrt(mean_squared_error(y_va_arr, final_va_pred))

final_te_pred = np.maximum(0.0, te_preds_matrix @ opt_weights)

submission_df = pd.DataFrame({"id": test_ids, target_col: final_te_pred})
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created."
assert len(submission_df) == len(
    X_te
), f"Row count {len(submission_df)} does not match test size {len(X_te)}."
assert list(submission_df.columns) == [
    "id",
    target_col,
], "Submission column names mismatch."

print(f"Final Validation Score: {final_val_rmse}")
