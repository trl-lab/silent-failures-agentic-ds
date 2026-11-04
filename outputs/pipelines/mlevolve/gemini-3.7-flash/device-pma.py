import os
import re
from typing import Any, Dict, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ==============================================================================
# 0. Directory and Environment Configuration
# ==============================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ==============================================================================
# 1. Data Loading and Automated Feature Engineering
# ==============================================================================
print("Loading train and test datasets from ./input...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

target_col = "decision_code"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' not found in train data."

# Encode Target if categorical/string
y_raw = train_df[target_col]
if (
    y_raw.dtype == "object"
    or y_raw.dtype.name == "category"
    or not np.issubdtype(y_raw.dtype, np.number)
):
    target_encoder = LabelEncoder()
    train_df[target_col] = target_encoder.fit_transform(y_raw.astype(str))
    print(
        f"Target classes mapped: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}"
    )
else:
    train_df[target_col] = y_raw.astype(int)

train_labels = train_df[target_col].copy()
features_df = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], axis=0
).reset_index(drop=True)
n_train = len(train_df)
n_test = len(test_df)

# Automated Column Type Identification
date_cols = []
text_cols = []
cat_cols = []
num_cols = []
date_patterns = re.compile(r"date|time|year|created|received|decision", re.IGNORECASE)

for col in features_df.columns:
    if col.lower() in ["id", "index", "row_id"]:
        continue
    col_data = features_df[col]
    if pd.api.types.is_datetime64_any_dtype(col_data):
        date_cols.append(col)
    elif col_data.dtype == "object" or col_data.dtype.name == "category":
        if bool(date_patterns.search(col)):
            try:
                pd.to_datetime(col_data.dropna().head(100))
                date_cols.append(col)
                continue
            except Exception:
                pass

        sample_str = col_data.dropna().astype(str).head(200)
        avg_len = sample_str.apply(len).mean() if len(sample_str) > 0 else 0
        avg_spaces = (
            sample_str.apply(lambda s: s.count(" ")).mean()
            if len(sample_str) > 0
            else 0
        )
        if avg_len > 35 or avg_spaces > 3:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    elif np.issubdtype(col_data.dtype, np.number):
        num_cols.append(col)

print(
    f"Identified features -> Dates: {len(date_cols)}, Text: {len(text_cols)}, Categorical: {len(cat_cols)}, Numeric: {len(num_cols)}"
)

# Date Feature Extraction
engineered_features = pd.DataFrame(index=features_df.index)
parsed_dates = {}

for col in date_cols:
    dt_series = pd.to_datetime(features_df[col], errors="coerce")
    parsed_dates[col] = dt_series

    engineered_features[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(int)
    engineered_features[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
    engineered_features[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(int)
    engineered_features[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
        int
    )
    engineered_features[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(-1).astype(
        int
    )
    engineered_features[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(int)
    engineered_features[f"{col}_is_na"] = dt_series.isna().astype(int)

    month_val = dt_series.dt.month.fillna(1)
    doy_val = dt_series.dt.dayofyear.fillna(1)
    engineered_features[f"{col}_sin_month"] = np.sin(2 * np.pi * month_val / 12.0)
    engineered_features[f"{col}_cos_month"] = np.cos(2 * np.pi * month_val / 12.0)
    engineered_features[f"{col}_sin_doy"] = np.sin(2 * np.pi * doy_val / 365.25)
    engineered_features[f"{col}_cos_doy"] = np.cos(2 * np.pi * doy_val / 365.25)

# Pairwise Date Differences
date_keys = list(parsed_dates.keys())
for i in range(len(date_keys)):
    for j in range(i + 1, len(date_keys)):
        d1, d2 = date_keys[i], date_keys[j]
        diff_days = (parsed_dates[d1] - parsed_dates[d2]).dt.total_seconds() / (
            24.0 * 3600.0
        )
        engineered_features[f"diff_days_{d1}_minus_{d2}"] = diff_days.fillna(-999.0)

# Categorical Feature Processing
for col in cat_cols:
    str_series = features_df[col].fillna("MISSING").astype(str)
    freq_map = str_series.value_counts(normalize=True).to_dict()
    engineered_features[f"{col}_freq"] = (
        str_series.map(freq_map).fillna(0.0).astype(float)
    )
    engineered_features[f"{col}_is_na"] = features_df[col].isna().astype(int)

    le = LabelEncoder()
    engineered_features[f"{col}_le"] = le.fit_transform(str_series)

# Text Feature Processing
for col in text_cols:
    str_series = features_df[col].fillna("").astype(str)
    engineered_features[f"{col}_char_len"] = str_series.apply(len)
    engineered_features[f"{col}_word_count"] = str_series.apply(
        lambda s: len(s.split())
    )
    engineered_features[f"{col}_upper_ratio"] = str_series.apply(
        lambda s: sum(1 for c in s if c.isupper()) / (len(s) + 1e-6)
    )
    engineered_features[f"{col}_digit_ratio"] = str_series.apply(
        lambda s: sum(1 for c in s if c.isdigit()) / (len(s) + 1e-6)
    )
    engineered_features[f"{col}_is_na"] = features_df[col].isna().astype(int)

    train_text = str_series.iloc[:n_train]
    test_text = str_series.iloc[n_train:]

    tfidf = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(train_text)
    test_tfidf = tfidf.transform(test_text)

    n_features_tfidf = train_tfidf.shape[1]
    if n_features_tfidf > 1:
        n_components = min(16, max(1, min(train_tfidf.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        train_svd = svd.fit_transform(train_tfidf)
        test_svd = svd.transform(test_tfidf)

        for comp in range(n_components):
            engineered_features.loc[: n_train - 1, f"{col}_svd_{comp}"] = train_svd[
                :, comp
            ]
            engineered_features.loc[n_train:, f"{col}_svd_{comp}"] = test_svd[:, comp]

# Numerical Feature Processing
for col in num_cols:
    num_series = features_df[col].copy()
    engineered_features[f"{col}_is_na"] = num_series.isna().astype(int)
    median_val = num_series.iloc[:n_train].median()
    engineered_features[col] = num_series.fillna(
        median_val if not pd.isna(median_val) else 0.0
    )

# Aggregation features across prominent categoricals
primary_cats = cat_cols[:3] if len(cat_cols) >= 1 else []
numeric_targets = [c for c in num_cols if engineered_features[c].std() > 1e-4]
if not numeric_targets and any("diff_days" in c for c in engineered_features.columns):
    numeric_targets = [c for c in engineered_features.columns if "diff_days" in c][:2]

for cat in primary_cats:
    for num_col in numeric_targets[:2]:
        group_stats = (
            engineered_features.iloc[:n_train]
            .groupby(f"{cat}_le")[num_col]
            .agg(["mean", "std"])
            .reset_index()
        )
        group_stats.columns = [
            f"{cat}_le",
            f"{cat}_{num_col}_mean",
            f"{cat}_{num_col}_std",
        ]

        merged = pd.merge(
            engineered_features[[f"{cat}_le"]], group_stats, on=f"{cat}_le", how="left"
        ).fillna(0.0)
        engineered_features[f"{cat}_{num_col}_mean"] = merged[
            f"{cat}_{num_col}_mean"
        ].values
        engineered_features[f"{cat}_{num_col}_std"] = merged[
            f"{cat}_{num_col}_std"
        ].values
        engineered_features[f"{cat}_{num_col}_diff_mean"] = (
            engineered_features[num_col] - engineered_features[f"{cat}_{num_col}_mean"]
        )

# Impute Infs and residual NaNs
engineered_features = engineered_features.replace([np.inf, -np.inf], np.nan)
for c in engineered_features.columns:
    if engineered_features[c].isna().any():
        fill_val = engineered_features.iloc[:n_train][c].median()
        engineered_features[c] = engineered_features[c].fillna(
            fill_val if not pd.isna(fill_val) else 0.0
        )

# Stratified K-Fold Out-of-Fold Target Encodings
train_processed = engineered_features.iloc[:n_train].copy().reset_index(drop=True)
test_processed = engineered_features.iloc[n_train:].copy().reset_index(drop=True)
train_processed[target_col] = train_labels.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_processed["fold"] = -1

for fold_idx, (train_idx, val_idx) in enumerate(
    skf.split(train_processed, train_processed[target_col])
):
    train_processed.loc[val_idx, "fold"] = fold_idx

for cat in primary_cats:
    col_name = f"{cat}_le"
    oof_te_train = np.zeros(n_train)
    global_mean = train_processed[target_col].mean()
    smoothing = 10.0

    for fold_idx in range(5):
        tr_mask = train_processed["fold"] != fold_idx
        va_mask = train_processed["fold"] == fold_idx

        tr_fold = train_processed[tr_mask]
        stats = tr_fold.groupby(col_name)[target_col].agg(["count", "mean"])
        smooth_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        val_cats = train_processed.loc[va_mask, col_name]
        oof_te_train[va_mask] = val_cats.map(smooth_te).fillna(global_mean).values

    train_processed[f"{cat}_target_enc"] = oof_te_train

    full_stats = train_processed.groupby(col_name)[target_col].agg(["count", "mean"])
    full_smooth_te = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    test_processed[f"{cat}_target_enc"] = (
        test_processed[col_name].map(full_smooth_te).fillna(global_mean).values
    )

train_processed.to_parquet("./working/train_features.parquet", index=False)
test_processed.to_parquet("./working/test_features.parquet", index=False)

# ==============================================================================
# 2. Model Architectures & Definitions
# ==============================================================================


class TabularResBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate),
        )

        blocks = []
        for i in range(len(hidden_dims) - 1):
            in_d = hidden_dims[i]
            out_d = hidden_dims[i + 1]
            blocks.append(TabularResBlock(in_d, dropout_rate=dropout_rate))
            if in_d != out_d:
                blocks.append(
                    nn.Sequential(
                        nn.Linear(in_d, out_d),
                        nn.LayerNorm(out_d),
                        nn.LeakyReLU(negative_slope=0.01),
                        nn.Dropout(p=dropout_rate),
                    )
                )

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_bn(x)
        h = self.input_proj(x_norm)
        h = self.backbone(h)
        logits = self.head(h)
        return logits.squeeze(-1)


def get_pytorch_model_components(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.25,
) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        dropout_rate=dropout_rate,
        output_dim=1,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    return model, criterion, optimizer


def get_lightgbm_model(custom_params: Dict[str, Any] = None) -> lgb.LGBMClassifier:
    default_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return lgb.LGBMClassifier(**default_params)


def get_xgboost_model(custom_params: Dict[str, Any] = None) -> xgb.XGBClassifier:
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }
    if custom_params:
        default_params.update(custom_params)
    return xgb.XGBClassifier(**default_params)


# ==============================================================================
# 3. Model Training, Ensemble Optimization, and Evaluation
# ==============================================================================
fold_col = "fold"
feature_cols = [c for c in train_processed.columns if c not in [target_col, fold_col]]

X_train_full = train_processed[feature_cols].values.astype(np.float32)
y_train_full = train_processed[target_col].values.astype(np.float32)
folds = train_processed[fold_col].values
X_test_full = test_processed[feature_cols].values.astype(np.float32)

n_samples = len(train_processed)
n_features = len(feature_cols)
n_test_samples = len(test_processed)

oof_preds_lgb = np.zeros(n_samples, dtype=np.float32)
oof_preds_xgb = np.zeros(n_samples, dtype=np.float32)
oof_preds_nn = np.zeros(n_samples, dtype=np.float32)

test_preds_lgb = np.zeros(n_test_samples, dtype=np.float32)
test_preds_xgb = np.zeros(n_test_samples, dtype=np.float32)
test_preds_nn = np.zeros(n_test_samples, dtype=np.float32)

n_folds = 5
for fold_idx in range(n_folds):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_tr, y_tr = X_train_full[train_mask], y_train_full[train_mask]
    X_va, y_va = X_train_full[val_mask], y_train_full[val_mask]

    # --- A. LightGBM ---
    lgb_clf = get_lightgbm_model({"random_state": 42 + fold_idx})
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=lgb_callbacks,
    )

    val_pred_lgb = lgb_clf.predict_proba(X_va)[:, 1]
    oof_preds_lgb[val_mask] = val_pred_lgb
    test_preds_lgb += lgb_clf.predict_proba(X_test_full)[:, 1] / n_folds

    # --- B. XGBoost ---
    xgb_clf = get_xgboost_model({"random_state": 42 + fold_idx})
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_pred_xgb = xgb_clf.predict_proba(X_va)[:, 1]
    oof_preds_xgb[val_mask] = val_pred_xgb
    test_preds_xgb += xgb_clf.predict_proba(X_test_full)[:, 1] / n_folds

    # --- C. TabularResNet ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test_full)

    train_tensor_x = torch.tensor(X_tr_scaled, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_tr, dtype=torch.float32)
    val_tensor_x = torch.tensor(X_va_scaled, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_va, dtype=torch.float32)
    test_tensor_x = torch.tensor(X_te_scaled, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor_x, train_tensor_y),
        batch_size=256,
        shuffle=True,
        drop_last=False,
    )

    nn_model, criterion, optimizer = get_pytorch_model_components(
        input_dim=n_features, lr=1e-3, weight_decay=1e-4, dropout_rate=0.25
    )
    nn_model = nn_model.to(device)

    best_val_auc = -1.0
    best_nn_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}
    epochs = 20

    for epoch in range(epochs):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        with torch.no_grad():
            val_logits = nn_model(val_tensor_x.to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            epoch_auc = roc_auc_score(y_va, val_probs)

        if epoch_auc > best_val_auc:
            best_val_auc = epoch_auc
            best_nn_weights = {k: v.cpu() for k, v in nn_model.state_dict().items()}

    nn_model.load_state_dict({k: v.to(device) for k, v in best_nn_weights.items()})
    nn_model.eval()
    with torch.no_grad():
        val_pred_nn = torch.sigmoid(nn_model(val_tensor_x.to(device))).cpu().numpy()
        test_pred_nn_fold = (
            torch.sigmoid(nn_model(test_tensor_x.to(device))).cpu().numpy()
        )

    oof_preds_nn[val_mask] = val_pred_nn
    test_preds_nn += test_pred_nn_fold / n_folds

    fold_ensemble_val = 0.45 * val_pred_lgb + 0.40 * val_pred_xgb + 0.15 * val_pred_nn
    fold_auc = roc_auc_score(y_va, fold_ensemble_val)
    print(
        f"Fold {fold_idx + 1}/{n_folds} - LGB: {roc_auc_score(y_va, val_pred_lgb):.5f}, XGB: {roc_auc_score(y_va, val_pred_xgb):.5f}, NN: {roc_auc_score(y_va, val_pred_nn):.5f} | Blend: {fold_auc:.5f}"
    )

# Optimize Ensemble Blending Weights on OOF
oof_matrix = np.column_stack([oof_preds_lgb, oof_preds_xgb, oof_preds_nn])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_nn])


def objective(weights):
    weights = np.clip(weights, 0, 1)
    norm_w = weights / (np.sum(weights) + 1e-8)
    blend_preds = oof_matrix @ norm_w
    return -roc_auc_score(y_train_full, blend_preds)


init_weights = np.array([0.45, 0.40, 0.15])
res = minimize(
    objective,
    init_weights,
    method="Nelder-Mead",
    options={"maxiter": 200, "disp": False},
)
opt_weights = np.clip(res.x, 0, 1)
opt_weights = opt_weights / np.sum(opt_weights)

oof_blend_preds = oof_matrix @ opt_weights
final_test_preds = test_matrix @ opt_weights
oof_auc = roc_auc_score(y_train_full, oof_blend_preds)

print(
    f"Individual OOF AUC -> LGB: {roc_auc_score(y_train_full, oof_preds_lgb):.5f}, XGB: {roc_auc_score(y_train_full, oof_preds_xgb):.5f}, TabularResNet: {roc_auc_score(y_train_full, oof_preds_nn):.5f}"
)
print(
    f"Optimal Ensemble Weights: LGB={opt_weights[0]:.3f}, XGB={opt_weights[1]:.3f}, NN={opt_weights[2]:.3f}"
)

# Generate and Validate Submission
submission_df = pd.DataFrame(
    {"id": np.arange(n_test_samples), "decision_code": final_test_preds}
)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(
    submission_path
), f"Submission file not found at {submission_path}"
assert len(submission_df) == n_test_samples, (
    f"Submission row count mismatch: expected {n_test_samples}, got"
    f" {len(submission_df)}"
)
assert list(submission_df.columns) == [
    "id",
    "decision_code",
], f"Incorrect submission columns: {list(submission_df.columns)}"
assert not submission_df.isna().any().any(), "Submission contains NaN values"

print(
    f"Submission successfully generated: {submission_path} with {len(submission_df)} rows."
)
print(f"Final Validation Score: {oof_auc}")
