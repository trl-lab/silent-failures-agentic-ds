import json
import math
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ==============================================================================
# Configuration & Reproducibility
# ==============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET_COL = "Total IFC investment as approved by Board(Million - USD)"

# ==============================================================================
# 1. Data Loading & Feature Engineering
# ==============================================================================
print(f"[INFO] Loading datasets from {INPUT_DIR}...")
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

assert (
    TARGET_COL in train_df.columns
), f"Target column '{TARGET_COL}' not found in train data!"

target_series = train_df[TARGET_COL].copy()
features_train = train_df.drop(columns=[TARGET_COL]).copy()
features_test = test_df.copy()

candidate_cols = list(features_train.columns)
date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in candidate_cols:
    train_vals = features_train[col].dropna()
    if len(train_vals) == 0:
        cat_cols.append(col)
        continue

    if pd.api.types.is_datetime64_any_dtype(features_train[col]):
        date_cols.append(col)
    elif any(k in col.lower() for k in ["date", "approval", "disclosed", "closing"]):
        try:
            pd.to_datetime(train_vals.head(50), errors="raise")
            date_cols.append(col)
        except Exception:
            if pd.api.types.is_numeric_dtype(features_train[col]):
                num_cols.append(col)
            else:
                cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(features_train[col]):
        num_cols.append(col)
    else:
        avg_char_len = train_vals.astype(str).str.len().mean()
        nunique = train_vals.nunique()
        if avg_char_len > 60 or nunique > 0.6 * len(train_vals):
            text_cols.append(col)
        else:
            cat_cols.append(col)

# 1.1 Temporal Feature Engineering
for dcol in date_cols:
    train_dt = pd.to_datetime(features_train[dcol], errors="coerce")
    test_dt = pd.to_datetime(features_test[dcol], errors="coerce")
    min_date = train_dt.min()

    for prefix, dt_s, df_target in [
        ("train", train_dt, features_train),
        ("test", test_dt, features_test),
    ]:
        df_target[f"{dcol}_year"] = dt_s.dt.year.fillna(-1).astype(int)
        df_target[f"{dcol}_month"] = dt_s.dt.month.fillna(-1).astype(int)
        df_target[f"{dcol}_day"] = dt_s.dt.day.fillna(-1).astype(int)
        df_target[f"{dcol}_quarter"] = dt_s.dt.quarter.fillna(-1).astype(int)
        df_target[f"{dcol}_dayofweek"] = dt_s.dt.dayofweek.fillna(-1).astype(int)
        df_target[f"{dcol}_is_month_end"] = dt_s.dt.is_month_end.fillna(False).astype(
            int
        )

        if pd.notnull(min_date):
            df_target[f"{dcol}_days_since_min"] = (
                (dt_s - min_date).dt.days.fillna(-999).astype(float)
            )
        else:
            df_target[f"{dcol}_days_since_min"] = -999.0

    features_train.drop(columns=[dcol], inplace=True)
    features_test.drop(columns=[dcol], inplace=True)

# 1.2 Text Feature Engineering (Length + TF-IDF SVD)
for tcol in text_cols:
    for df_target in [features_train, features_test]:
        str_series = df_target[tcol].fillna("").astype(str)
        df_target[f"{tcol}_char_len"] = str_series.str.len()
        df_target[f"{tcol}_word_count"] = str_series.apply(lambda x: len(x.split()))
        df_target[f"{tcol}_num_digits"] = str_series.apply(
            lambda x: sum(c.isdigit() for c in x)
        )

    tfidf = TfidfVectorizer(
        max_features=5000, stop_words="english", token_pattern=r"(?u)\b\w+\b"
    )
    train_text = features_train[tcol].fillna("").astype(str)
    test_text = features_test[tcol].fillna("").astype(str)

    train_tfidf_sparse = tfidf.fit_transform(train_text)
    test_tfidf_sparse = tfidf.transform(test_text)

    n_components = (
        min(10, train_tfidf_sparse.shape[1] - 1)
        if train_tfidf_sparse.shape[1] > 2
        else 0
    )
    if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        train_svd = svd.fit_transform(train_tfidf_sparse)
        test_svd = svd.transform(test_tfidf_sparse)

        for comp_i in range(n_components):
            features_train[f"{tcol}_svd_{comp_i}"] = train_svd[:, comp_i]
            features_test[f"{tcol}_svd_{comp_i}"] = test_svd[:, comp_i]

    features_train.drop(columns=[tcol], inplace=True)
    features_test.drop(columns=[tcol], inplace=True)

# 1.3 Categorical Feature Engineering
for ccol in cat_cols:
    features_train[ccol] = (
        features_train[ccol].fillna("Missing").astype(str).str.strip()
    )
    features_test[ccol] = features_test[ccol].fillna("Missing").astype(str).str.strip()

    freq_map = features_train[ccol].value_counts(normalize=True).to_dict()
    features_train[f"{ccol}_freq"] = features_train[ccol].map(freq_map).fillna(0.0)
    features_test[f"{ccol}_freq"] = features_test[ccol].map(freq_map).fillna(0.0)

    le = LabelEncoder()
    le.fit(list(features_train[ccol].unique()) + ["__UNKNOWN__"])
    mapping = {val: idx for idx, val in enumerate(le.classes_)}
    unk_idx = mapping["__UNKNOWN__"]

    features_train[f"{ccol}_encoded"] = (
        features_train[ccol].map(mapping).fillna(unk_idx).astype(int)
    )
    features_test[f"{ccol}_encoded"] = (
        features_test[ccol].map(mapping).fillna(unk_idx).astype(int)
    )

    features_train.drop(columns=[ccol], inplace=True)
    features_test.drop(columns=[ccol], inplace=True)

# 1.4 Numerical Transformations & Ratios
for ncol in num_cols:
    features_train[f"{ncol}_isna"] = features_train[ncol].isna().astype(int)
    features_test[f"{ncol}_isna"] = features_test[ncol].isna().astype(int)

    if (features_train[ncol].dropna() >= 0).all():
        features_train[f"{ncol}_log1p"] = np.log1p(np.maximum(0, features_train[ncol]))
        features_test[f"{ncol}_log1p"] = np.log1p(np.maximum(0, features_test[ncol]))

num_col_list = [c for c in num_cols if c in features_train.columns]
for i in range(len(num_col_list)):
    for j in range(i + 1, min(i + 4, len(num_col_list))):
        c1 = num_col_list[i]
        c2 = num_col_list[j]
        features_train[f"{c1}_ratio_{c2}"] = features_train[c1] / (
            features_train[c2].abs() + 1e-5
        )
        features_test[f"{c1}_ratio_{c2}"] = features_test[c1] / (
            features_test[c2].abs() + 1e-5
        )

# Group-level aggregations
top_group_cols = [c for c in features_train.columns if c.endswith("_encoded")][:3]
for gcol in top_group_cols:
    for ncol in num_col_list[:3]:
        group_stats = (
            features_train.groupby(gcol)[ncol]
            .agg(["mean", "std", "median"])
            .rename(
                columns={
                    "mean": f"{ncol}_mean_by_{gcol}",
                    "std": f"{ncol}_std_by_{gcol}",
                    "median": f"{ncol}_median_by_{gcol}",
                }
            )
        )
        features_train = features_train.merge(group_stats, on=gcol, how="left")
        features_test = features_test.merge(group_stats, on=gcol, how="left")

# 1.5 Stratified K-Fold Splits & Out-Of-Fold Smoothed Target Encoding
N_SPLITS = 5
features_train["fold"] = -1

n_bins = min(10, max(2, len(target_series) // 50))
target_bins = pd.qcut(target_series.rank(method="first"), q=n_bins, labels=False)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features_train, target_bins)):
    features_train.loc[val_idx, "fold"] = fold_idx

for gcol in top_group_cols:
    features_train[f"{gcol}_oof_te"] = np.nan
    features_test[f"{gcol}_oof_te"] = np.nan

    global_mean = target_series.mean()
    smoothing_weight = 10.0

    for fold_idx in range(N_SPLITS):
        tr_mask = features_train["fold"] != fold_idx
        va_mask = features_train["fold"] == fold_idx

        tr_cat = features_train.loc[tr_mask, gcol]
        tr_target = target_series[tr_mask]

        stats = tr_target.groupby(tr_cat).agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smoothing_weight * global_mean) / (
            stats["count"] + smoothing_weight
        )

        features_train.loc[va_mask, f"{gcol}_oof_te"] = (
            features_train.loc[va_mask, gcol].map(smoothed).fillna(global_mean)
        )

    full_stats = target_series.groupby(features_train[gcol]).agg(["count", "mean"])
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + smoothing_weight * global_mean
    ) / (full_stats["count"] + smoothing_weight)
    features_test[f"{gcol}_oof_te"] = (
        features_test[gcol].map(full_smoothed).fillna(global_mean)
    )

features_train[TARGET_COL] = target_series.values

feature_cols = [c for c in features_train.columns if c not in [TARGET_COL, "fold"]]
print(
    f"[INFO] Processed train shape: {features_train.shape}, test shape: {features_test.shape}, features: {len(feature_cols)}"
)


# ==============================================================================
# 2. Neural Model Architecture Definition
# ==============================================================================
class ResBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TabularResNetModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [
                ResBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)


# ==============================================================================
# 3. Model Training & Out-of-Fold Cross-Validation
# ==============================================================================
X_full = features_train[feature_cols].copy()
y_full = features_train[TARGET_COL].values
folds = features_train["fold"].values
X_test = features_test[feature_cols].copy()

X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

oof_preds_lgb = np.zeros(len(features_train), dtype=np.float64)
oof_preds_xgb = np.zeros(len(features_train), dtype=np.float64)
oof_preds_nn = np.zeros(len(features_train), dtype=np.float64)

test_preds_lgb = np.zeros(len(features_test), dtype=np.float64)
test_preds_xgb = np.zeros(len(features_test), dtype=np.float64)
test_preds_nn = np.zeros(len(features_test), dtype=np.float64)

print("\n[INFO] Starting 5-Fold Cross-Validation Training...")

for fold in range(N_SPLITS):
    tr_mask = folds != fold
    va_mask = folds == fold

    X_tr_df, y_tr = X_full.iloc[tr_mask], y_full[tr_mask]
    X_va_df, y_va = X_full.iloc[va_mask], y_full[va_mask]
    X_te_df = X_test.copy()

    # LightGBM
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED + fold,
        "verbose": -1,
        "n_estimators": 2000,
        "n_jobs": -1,
    }
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr_df,
        y_tr,
        eval_set=[(X_va_df, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    va_pred_lgb = model_lgb.predict(X_va_df)
    oof_preds_lgb[va_mask] = va_pred_lgb
    test_preds_lgb += model_lgb.predict(X_te_df) / N_SPLITS

    # XGBoost
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED + fold,
        "n_estimators": 2000,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "n_jobs": -1,
    }
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_tr_df, y_tr, eval_set=[(X_va_df, y_va)], verbose=False)

    va_pred_xgb = model_xgb.predict(X_va_df)
    oof_preds_xgb[va_mask] = va_pred_xgb
    test_preds_xgb += model_xgb.predict(X_te_df) / N_SPLITS

    # Tabular Residual Neural Network
    impute_vals = X_tr_df.median()
    X_tr_imp = X_tr_df.fillna(impute_vals).fillna(0.0).values
    X_va_imp = X_va_df.fillna(impute_vals).fillna(0.0).values
    X_te_imp = X_te_df.fillna(impute_vals).fillna(0.0).values

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_va_scaled = scaler.transform(X_va_imp)
    X_te_scaled = scaler.transform(X_te_imp)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    batch_size = min(128, max(32, len(X_tr_scaled) // 16))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_nn = TabularResNetModel(
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.15,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model_nn.parameters(), lr=3e-3, weight_decay=1e-4, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=80, eta_min=1e-5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_nn_weights = None
    patience = 20
    patience_counter = 0

    for epoch in range(80):
        model_nn.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model_nn(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_nn.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model_nn.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                preds = model_nn(batch_x)
                val_loss = criterion(preds, batch_y).item()
                val_losses.append(val_loss)

        avg_val_loss = np.mean(val_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in model_nn.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model_nn.load_state_dict(best_nn_weights)
    model_nn.eval()
    with torch.no_grad():
        va_pred_nn = (
            model_nn(torch.tensor(X_va_scaled, dtype=torch.float32).to(DEVICE))
            .cpu()
            .numpy()
        )
        te_pred_nn = (
            model_nn(torch.tensor(X_te_scaled, dtype=torch.float32).to(DEVICE))
            .cpu()
            .numpy()
        )

    oof_preds_nn[va_mask] = va_pred_nn
    test_preds_nn += te_pred_nn / N_SPLITS

    fold_lgb_rmse = root_mean_squared_error(y_va, va_pred_lgb)
    fold_xgb_rmse = root_mean_squared_error(y_va, va_pred_xgb)
    fold_nn_rmse = root_mean_squared_error(y_va, va_pred_nn)
    print(
        f"[Fold {fold+1}/{N_SPLITS}] LGB RMSE: {fold_lgb_rmse:.4f} | XGB RMSE: {fold_xgb_rmse:.4f} | NN RMSE: {fold_nn_rmse:.4f}"
    )

# ==============================================================================
# 4. Ensemble Optimization & Validation Evaluation
# ==============================================================================
total_lgb_rmse = root_mean_squared_error(y_full, oof_preds_lgb)
total_xgb_rmse = root_mean_squared_error(y_full, oof_preds_xgb)
total_nn_rmse = root_mean_squared_error(y_full, oof_preds_nn)

print(
    f"\n[INFO] Overall OOF - LGB: {total_lgb_rmse:.4f}, XGB: {total_xgb_rmse:.4f}, NN: {total_nn_rmse:.4f}"
)

oof_matrix = np.column_stack([oof_preds_lgb, oof_preds_xgb, oof_preds_nn])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_nn])


def loss_func(weights):
    weights = weights / np.sum(weights)
    blend_oof = np.dot(oof_matrix, weights)
    return root_mean_squared_error(y_full, blend_oof)


init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}
opt_res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

opt_weights = (
    opt_res.x / np.sum(opt_res.x) if opt_res.success else np.array([0.45, 0.45, 0.10])
)
print(
    f"[INFO] Optimal ensemble weights (LGB, XGB, NN): [{opt_weights[0]:.3f}, {opt_weights[1]:.3f}, {opt_weights[2]:.3f}]"
)

final_oof_preds = np.maximum(0.0, np.dot(oof_matrix, opt_weights))
final_validation_rmse = root_mean_squared_error(y_full, final_oof_preds)

# ==============================================================================
# 5. Generate Submission File
# ==============================================================================
final_test_preds = np.maximum(0.0, np.dot(test_matrix, opt_weights))

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), TARGET_COL: final_test_preds}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"[INFO] Submission saved to {submission_path} with {len(submission_df)} rows.")

print(f"Final Validation Score: {final_validation_rmse}")
