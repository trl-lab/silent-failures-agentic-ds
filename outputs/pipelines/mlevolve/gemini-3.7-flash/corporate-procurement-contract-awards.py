import os
import re
from typing import Any, Dict, Tuple
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ==============================================================================
# 0. Global Setup and Directory Initialization
# ==============================================================================
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Load Raw Data and Fold Partitioning
# ==============================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Contract Award Amount (USD)"
assert target_col in train_df.columns, f"Target {target_col} not found in train data!"

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold_idx, (_, val_idx) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, "fold"] = fold_idx

# ==============================================================================
# 2. Dynamic Column Identification & Leak-Free Feature Engineering
# ==============================================================================
feature_cols = [c for c in train_df.columns if c not in [target_col, "fold"]]

date_cols = []
text_cols = []
cat_cols = []
num_cols = []

for col in feature_cols:
    if (
        pd.api.types.is_datetime64_any_dtype(train_df[col])
        or "date" in col.lower()
        or "year" in col.lower()
    ):
        try:
            _ = pd.to_datetime(train_df[col].dropna().iloc[:50], format="mixed", errors="coerce")
            date_cols.append(col)
            continue
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(train_df[col]):
        num_cols.append(col)
    elif pd.api.types.is_string_dtype(train_df[col]) or pd.api.types.is_object_dtype(
        train_df[col]
    ):
        avg_len = (
            train_df[col].dropna().astype(str).str.len().mean()
            if len(train_df[col].dropna()) > 0
            else 0
        )
        if avg_len > 40:
            text_cols.append(col)
        else:
            cat_cols.append(col)
    else:
        cat_cols.append(col)


def extract_date_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Extract granular date components, cyclic encodings, and elapsed intervals."""
    date_features = pd.DataFrame(index=df.index)
    parsed_dates = {}
    for col in cols:
        s = pd.to_datetime(df[col], format="mixed", errors="coerce")
        parsed_dates[col] = s
        prefix = f"date_{col}"
        month = s.dt.month.fillna(1).astype(np.float32)
        dow = s.dt.dayofweek.fillna(0).astype(np.float32)
        doy = s.dt.dayofyear.fillna(1).astype(np.float32)

        date_features[f"{prefix}_year"] = s.dt.year.fillna(-1).astype(np.int32)
        date_features[f"{prefix}_month"] = s.dt.month.fillna(-1).astype(np.int32)
        date_features[f"{prefix}_day"] = s.dt.day.fillna(-1).astype(np.int32)
        date_features[f"{prefix}_dayofweek"] = s.dt.dayofweek.fillna(-1).astype(np.int32)
        date_features[f"{prefix}_quarter"] = s.dt.quarter.fillna(-1).astype(np.int32)
        date_features[f"{prefix}_dayofyear"] = s.dt.dayofyear.fillna(-1).astype(np.int32)
        date_features[f"{prefix}_is_month_end"] = s.dt.is_month_end.astype(float).fillna(0)
        date_features[f"{prefix}_days_epoch"] = (
            (s - pd.Timestamp("2000-01-01")).dt.days.fillna(-999).astype(np.float32)
        )

        # Cyclical representations
        date_features[f"{prefix}_month_sin"] = np.sin(2.0 * np.pi * month / 12.0).astype(np.float32)
        date_features[f"{prefix}_month_cos"] = np.cos(2.0 * np.pi * month / 12.0).astype(np.float32)
        date_features[f"{prefix}_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
        date_features[f"{prefix}_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)
        date_features[f"{prefix}_doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
        date_features[f"{prefix}_doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)

    if len(cols) >= 2:
        col_list = list(cols)
        for i in range(len(col_list)):
            for j in range(i + 1, len(col_list)):
                c1, c2 = col_list[i], col_list[j]
                diff_days = (
                    (parsed_dates[c1] - parsed_dates[c2])
                    .dt.days.fillna(-999)
                    .astype(np.float32)
                )
                date_features[f"diff_days_{c1}_vs_{c2}"] = diff_days

    return date_features


def extract_text_meta_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Extract statistical properties from text columns."""
    text_meta = pd.DataFrame(index=df.index)
    for col in cols:
        s = df[col].fillna("").astype(str)
        prefix = f"text_{col}"
        text_meta[f"{prefix}_char_len"] = s.str.len().astype(np.float32)
        text_meta[f"{prefix}_word_count"] = s.str.split().str.len().astype(np.float32)
        text_meta[f"{prefix}_digit_count"] = s.apply(
            lambda x: sum(c.isdigit() for c in x)
        ).astype(np.float32)
        text_meta[f"{prefix}_upper_count"] = s.apply(
            lambda x: sum(c.isupper() for c in x)
        ).astype(np.float32)
        text_meta[f"{prefix}_avg_word_len"] = (
            text_meta[f"{prefix}_char_len"] / (text_meta[f"{prefix}_word_count"] + 1e-5)
        ).astype(np.float32)
    return text_meta


# Date and numeric feature extraction
train_date_feat = extract_date_features(train_df, date_cols)
test_date_feat = extract_date_features(test_df, date_cols)

train_text_meta = extract_text_meta_features(train_df, text_cols)
test_text_meta = extract_text_meta_features(test_df, text_cols)

train_num_feat = pd.DataFrame(index=train_df.index)
test_num_feat = pd.DataFrame(index=test_df.index)
for col in num_cols:
    train_num_feat[f"num_{col}"] = train_df[col].astype(np.float32)
    train_num_feat[f"num_{col}_isna"] = train_df[col].isna().astype(np.float32)
    test_num_feat[f"num_{col}"] = test_df[col].astype(np.float32)
    test_num_feat[f"num_{col}_isna"] = test_df[col].isna().astype(np.float32)

# TF-IDF + SVD text representations under strict fold isolation
train_svd_dict = {
    col: np.zeros((len(train_df), 16), dtype=np.float32) for col in text_cols
}
test_svd_dict = {
    col: np.zeros((len(test_df), 16), dtype=np.float32) for col in text_cols
}

for col in text_cols:
    n_components = min(16, max(2, len(train_df) // 100))
    tfidf_full = TfidfVectorizer(
        max_features=2500, ngram_range=(1, 2), stop_words="english"
    )
    svd_full = TruncatedSVD(n_components=n_components, random_state=42)

    full_train_text = train_df[col].fillna("").astype(str)
    tfidf_mat_full = tfidf_full.fit_transform(full_train_text)
    _ = svd_full.fit_transform(tfidf_mat_full)

    full_test_text = test_df[col].fillna("").astype(str)
    test_tfidf_mat = tfidf_full.transform(full_test_text)
    test_svd_dict[col] = svd_full.transform(test_tfidf_mat).astype(np.float32)

    for f in range(N_SPLITS):
        tr_idx = train_df["fold"] != f
        va_idx = train_df["fold"] == f

        tfidf_fold = TfidfVectorizer(
            max_features=2500, ngram_range=(1, 2), stop_words="english"
        )
        svd_fold = TruncatedSVD(n_components=n_components, random_state=42)

        tr_vec = tfidf_fold.fit_transform(
            train_df.loc[tr_idx, col].fillna("").astype(str)
        )
        svd_fold.fit(tr_vec)

        va_vec = tfidf_fold.transform(train_df.loc[va_idx, col].fillna("").astype(str))
        train_svd_dict[col][va_idx] = svd_fold.transform(va_vec).astype(np.float32)

train_svd_df = pd.DataFrame(index=train_df.index)
test_svd_df = pd.DataFrame(index=test_df.index)
for col in text_cols:
    for i in range(train_svd_dict[col].shape[1]):
        train_svd_df[f"svd_{col}_{i}"] = train_svd_dict[col][:, i]
        test_svd_df[f"svd_{col}_{i}"] = test_svd_dict[col][:, i]

# Categorical Frequency and Out-of-Fold Target Encoding
target_log = np.log1p(np.maximum(0.0, train_df[target_col].values))
train_cat_feat = pd.DataFrame(index=train_df.index)
test_cat_feat = pd.DataFrame(index=test_df.index)

cat_interaction_pairs = []
if len(cat_cols) >= 2:
    for i in range(min(4, len(cat_cols))):
        for j in range(i + 1, min(4, len(cat_cols))):
            cat_interaction_pairs.append((cat_cols[i], cat_cols[j]))

all_eval_cats = list(cat_cols)
for c1, c2 in cat_interaction_pairs:
    inter_name = f"{c1}__x__{c2}"
    train_df[inter_name] = train_df[c1].astype(str) + "___" + train_df[c2].astype(str)
    test_df[inter_name] = test_df[c1].astype(str) + "___" + test_df[c2].astype(str)
    all_eval_cats.append(inter_name)

for col in all_eval_cats:
    freq_map = train_df[col].value_counts(dropna=False).to_dict()
    train_cat_feat[f"freq_{col}"] = (
        train_df[col].map(freq_map).fillna(0).astype(np.float32)
    )
    test_cat_feat[f"freq_{col}"] = (
        test_df[col].map(freq_map).fillna(0).astype(np.float32)
    )

SMOOTHING = 10.0
global_target_mean = float(np.mean(target_log))
global_target_std = float(np.std(target_log))

for col in all_eval_cats:
    train_te_mean = np.zeros(len(train_df), dtype=np.float32)
    train_te_std = np.zeros(len(train_df), dtype=np.float32)

    for f in range(N_SPLITS):
        tr_mask = train_df["fold"] != f
        va_mask = train_df["fold"] == f

        fold_tr_df = train_df[tr_mask]
        fold_y = target_log[tr_mask]

        stats = (
            pd.DataFrame({"cat": fold_tr_df[col], "target": fold_y})
            .groupby("cat")["target"]
            .agg(["count", "mean", "std"])
        )

        smoothed_mean = (
            stats["count"] * stats["mean"] + SMOOTHING * global_target_mean
        ) / (stats["count"] + SMOOTHING)
        smoothed_std = stats["std"].fillna(global_target_std)

        val_cats = train_df.loc[va_mask, col]
        train_te_mean[va_mask] = (
            val_cats.map(smoothed_mean).fillna(global_target_mean).values
        )
        train_te_std[va_mask] = (
            val_cats.map(smoothed_std).fillna(global_target_std).values
        )

    train_cat_feat[f"te_mean_{col}"] = train_te_mean
    train_cat_feat[f"te_std_{col}"] = train_te_std

    full_stats = (
        pd.DataFrame({"cat": train_df[col], "target": target_log})
        .groupby("cat")["target"]
        .agg(["count", "mean", "std"])
    )
    full_smoothed_mean = (
        full_stats["count"] * full_stats["mean"] + SMOOTHING * global_target_mean
    ) / (full_stats["count"] + SMOOTHING)
    full_smoothed_std = full_stats["std"].fillna(global_target_std)

    test_cat_feat[f"te_mean_{col}"] = (
        test_df[col]
        .map(full_smoothed_mean)
        .fillna(global_target_mean)
        .astype(np.float32)
        .values
    )
    test_cat_feat[f"te_std_{col}"] = (
        test_df[col]
        .map(full_smoothed_std)
        .fillna(global_target_std)
        .astype(np.float32)
        .values
    )

# Fold-safe group aggregations of numeric features grouped by categorical features
train_grp_feat = pd.DataFrame(index=train_df.index)
test_grp_feat = pd.DataFrame(index=test_df.index)

if len(num_cols) > 0 and len(cat_cols) > 0:
    for c_col in cat_cols[:4]:
        for n_col in num_cols[:4]:
            grp_mean_train = np.zeros(len(train_df), dtype=np.float32)
            grp_std_train = np.zeros(len(train_df), dtype=np.float32)

            for f in range(N_SPLITS):
                tr_mask = train_df["fold"] != f
                va_mask = train_df["fold"] == f

                grp_stats = (
                    train_df.loc[tr_mask]
                    .groupby(c_col)[n_col]
                    .agg(["mean", "std"])
                )
                global_m = float(train_df.loc[tr_mask, n_col].mean())
                global_s = float(train_df.loc[tr_mask, n_col].std())
                if np.isnan(global_s):
                    global_s = 0.0

                grp_mean_train[va_mask] = (
                    train_df.loc[va_mask, c_col]
                    .map(grp_stats["mean"])
                    .fillna(global_m)
                    .values
                )
                grp_std_train[va_mask] = (
                    train_df.loc[va_mask, c_col]
                    .map(grp_stats["std"])
                    .fillna(global_s)
                    .values
                )

            train_grp_feat[f"grp_{c_col}_{n_col}_mean"] = grp_mean_train
            train_grp_feat[f"grp_{c_col}_{n_col}_std"] = grp_std_train

            full_grp = train_df.groupby(c_col)[n_col].agg(["mean", "std"])
            full_m = float(train_df[n_col].mean())
            full_s = float(train_df[n_col].std())
            if np.isnan(full_s):
                full_s = 0.0

            test_grp_feat[f"grp_{c_col}_{n_col}_mean"] = (
                test_df[c_col]
                .map(full_grp["mean"])
                .fillna(full_m)
                .astype(np.float32)
                .values
            )
            test_grp_feat[f"grp_{c_col}_{n_col}_std"] = (
                test_df[c_col]
                .map(full_grp["std"])
                .fillna(full_s)
                .astype(np.float32)
                .values
            )

# Combine engineered features
X_train_df = pd.concat(
    [train_num_feat, train_date_feat, train_text_meta, train_svd_df, train_cat_feat, train_grp_feat],
    axis=1,
)
X_test_df = pd.concat(
    [test_num_feat, test_date_feat, test_text_meta, test_svd_df, test_cat_feat, test_grp_feat], axis=1
)

clean_cols = [re.sub(r"[^\w]+", "_", col) for col in X_train_df.columns]
X_train_df.columns = clean_cols
X_test_df.columns = clean_cols

test_ids = (
    test_df["id"].values
    if "id" in test_df.columns
    else np.arange(len(test_df), dtype=np.int64)
)

X_train_arr = X_train_df.fillna(0.0).values.astype(np.float32)
X_test_arr = X_test_df.fillna(0.0).values.astype(np.float32)
y_raw = train_df[target_col].values.astype(np.float64)
y_log = target_log.astype(np.float32)
folds = train_df["fold"].values.astype(np.int32)


# ==============================================================================
# 3. Model Definitions
# ==============================================================================
def create_lgb_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def create_xgb_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        booster="gbtree",
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def create_cb_model() -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        early_stopping_rounds=50,
        verbose=False,
    )


class ResidualBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.2,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dims[0], dropout_rate=dropout_rate) for _ in range(2)]
        )
        layers = []
        for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_d, out_d),
                    nn.LayerNorm(out_d),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
        self.transition_layers = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        h = self.transition_layers(h)
        return self.head(h).squeeze(-1)


# ==============================================================================
# 4. 5-Fold Cross-Validation Training & Out-Of-Fold Ensembling
# ==============================================================================
n_train = len(train_df)
n_test = len(test_df)

oof_lgb = np.zeros(n_train, dtype=np.float64)
oof_xgb = np.zeros(n_train, dtype=np.float64)
oof_cb = np.zeros(n_train, dtype=np.float64)
oof_mlp = np.zeros(n_train, dtype=np.float64)

test_lgb = np.zeros(n_test, dtype=np.float64)
test_xgb = np.zeros(n_test, dtype=np.float64)
test_cb = np.zeros(n_test, dtype=np.float64)
test_mlp = np.zeros(n_test, dtype=np.float64)

for fold in range(N_SPLITS):
    tr_idx = folds != fold
    va_idx = folds == fold

    X_tr, y_tr_log = X_train_arr[tr_idx], y_log[tr_idx]
    X_va, y_val_log = X_train_arr[va_idx], y_log[va_idx]
    y_val_raw = y_raw[va_idx]

    # LightGBM
    lgb_model = create_lgb_model()
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_val_log)],
        callbacks=lgb_callbacks,
    )
    oof_lgb[va_idx] = np.maximum(0.0, np.expm1(lgb_model.predict(X_va)))
    test_lgb += np.maximum(0.0, np.expm1(lgb_model.predict(X_test_arr))) / N_SPLITS

    # XGBoost
    xgb_model = create_xgb_model()
    xgb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_val_log)],
        verbose=False,
    )
    oof_xgb[va_idx] = np.maximum(0.0, np.expm1(xgb_model.predict(X_va)))
    test_xgb += np.maximum(0.0, np.expm1(xgb_model.predict(X_test_arr))) / N_SPLITS

    # CatBoost
    cb_model = create_cb_model()
    cb_model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_val_log)],
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[va_idx] = np.maximum(0.0, np.expm1(cb_model.predict(X_va)))
    test_cb += np.maximum(0.0, np.expm1(cb_model.predict(X_test_arr))) / N_SPLITS

    # Tabular Residual MLP
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_test_scaled = scaler.transform(X_test_arr)

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr_log, dtype=torch.float32),
    )
    batch_size = min(256, max(32, len(X_tr) // 10))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    mlp_model = TabularResMLP(
        input_dim=X_tr.shape[1], hidden_dims=(256, 128, 64), dropout_rate=0.2
    ).to(device)
    criterion = nn.MSELoss()
    EPOCHS = 35
    optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6
    )

    best_val_loss = float("inf")
    best_weights = None

    vx = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
    vy = torch.tensor(y_val_log, dtype=torch.float32).to(device)

    for epoch in range(EPOCHS):
        mlp_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = mlp_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            scheduler.step()

        mlp_model.eval()
        with torch.no_grad():
            v_out = mlp_model(vx)
            v_loss = criterion(v_out, vy).item()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_weights = {
                k: v.cpu().clone() for k, v in mlp_model.state_dict().items()
            }

    mlp_model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    mlp_model.eval()
    with torch.no_grad():
        val_mlp_tensor = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
        pred_val_mlp_log = mlp_model(val_mlp_tensor).cpu().numpy()
        oof_mlp[va_idx] = np.maximum(0.0, np.expm1(pred_val_mlp_log))

        test_mlp_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        pred_test_mlp_log = mlp_model(test_mlp_tensor).cpu().numpy()
        test_mlp += np.maximum(0.0, np.expm1(pred_test_mlp_log)) / N_SPLITS


# ==============================================================================
# 5. Optimal Metric-Aligned Ensemble Optimization
# ==============================================================================
def blend_objective(weights):
    w1, w2, w3, w4 = weights
    pred = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb + w4 * oof_mlp
    return np.sqrt(mean_squared_error(y_raw, pred))


init_weights = [0.35, 0.35, 0.20, 0.10]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

opt_res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
opt_w = opt_res.x if opt_res.success else np.array([0.35, 0.35, 0.20, 0.10])

oof_final_pred = (
    opt_w[0] * oof_lgb
    + opt_w[1] * oof_xgb
    + opt_w[2] * oof_cb
    + opt_w[3] * oof_mlp
)
final_val_score = float(np.sqrt(mean_squared_error(y_raw, oof_final_pred)))

# ==============================================================================
# 6. Test Predictions & Final Submission Generation
# ==============================================================================
test_final_pred = (
    opt_w[0] * test_lgb
    + opt_w[1] * test_xgb
    + opt_w[2] * test_cb
    + opt_w[3] * test_mlp
)
test_final_pred = np.maximum(0.0, test_final_pred)

submission_df = pd.DataFrame(
    {"id": test_ids, "Contract Award Amount (USD)": test_final_pred}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_val_score}")
