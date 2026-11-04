import os
import sys
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. Global Setup & Seed Fixation
# ------------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# 2. Data Loading & Dynamic Column Discovery
# ------------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Size of Purchase (MW)"
if target_col not in train_df.columns:
    for c in train_df.columns:
        if any(term in c.lower() for term in ["purchase", "mw", "size"]):
            target_col = c
            break

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))
    test_df["id"] = test_ids

if "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df))

# Clean Target Column
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
train_df = train_df[train_df[target_col] >= 0].reset_index(drop=True)
y = train_df[target_col].values

num_bins = min(10, len(np.unique(y)))
y_binned = (
    pd.qcut(y, q=num_bins, labels=False, duplicates="drop")
    if num_bins > 1
    else np.zeros(len(y), dtype=int)
)

feature_cols_raw = [c for c in train_df.columns if c not in [target_col, "id"]]

cat_cols = []
num_cols = []
text_cols = []
date_cols = []

for col in feature_cols_raw:
    if pd.api.types.is_datetime64_any_dtype(train_df[col]):
        date_cols.append(col)
        continue

    if train_df[col].dtype == "object":
        sample_non_null = train_df[col].dropna().astype(str).head(50)
        is_date = False
        if len(sample_non_null) > 0:
            try:
                parsed = pd.to_datetime(sample_non_null, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    train_df[col] = pd.to_datetime(train_df[col], errors="coerce")
                    test_df[col] = pd.to_datetime(test_df[col], errors="coerce")
                    date_cols.append(col)
                    is_date = True
            except Exception:
                pass
        if is_date:
            continue

    if pd.api.types.is_numeric_dtype(train_df[col]):
        num_cols.append(col)
    else:
        avg_len = train_df[col].astype(str).str.len().mean()
        n_unique = train_df[col].nunique()
        if avg_len > 50 or (n_unique > 100 and avg_len > 25):
            text_cols.append(col)
        else:
            cat_cols.append(col)

# ------------------------------------------------------------------------------
# 3. Feature Engineering Pipeline
# ------------------------------------------------------------------------------
train_feat = pd.DataFrame({"id": train_df["id"]})
test_feat = pd.DataFrame({"id": test_df["id"]})

# Date Feature Extraction
for col in date_cols:
    for df, feat_df in [(train_df, train_feat), (test_df, test_feat)]:
        feat_df[f"{col}_year"] = df[col].dt.year
        feat_df[f"{col}_month"] = df[col].dt.month
        feat_df[f"{col}_quarter"] = df[col].dt.quarter
        feat_df[f"{col}_dayofweek"] = df[col].dt.dayofweek

# Categorical Frequency & Factorization
for col in cat_cols:
    train_str = train_df[col].astype(str).str.strip().str.lower().fillna("missing")
    test_str = test_df[col].astype(str).str.strip().str.lower().fillna("missing")

    freq_map = pd.concat([train_str, test_str]).value_counts().to_dict()
    train_feat[f"{col}_freq"] = train_str.map(freq_map).fillna(0)
    test_feat[f"{col}_freq"] = test_str.map(freq_map).fillna(0)

    cats, _ = pd.factorize(pd.concat([train_str, test_str]))
    train_feat[f"{col}_cat"] = cats[: len(train_df)]
    test_feat[f"{col}_cat"] = cats[len(train_df) :]

# Pairwise Categorical Interaction Features
if len(cat_cols) >= 2:
    for i in range(min(3, len(cat_cols))):
        for j in range(i + 1, min(4, len(cat_cols))):
            col1, col2 = cat_cols[i], cat_cols[j]
            train_comb = train_df[col1].astype(str) + "__" + train_df[col2].astype(str)
            test_comb = test_df[col1].astype(str) + "__" + test_df[col2].astype(str)
            cats, _ = pd.factorize(pd.concat([train_comb, test_comb]))
            train_feat[f"{col1}_{col2}_comb"] = cats[: len(train_df)]
            test_feat[f"{col1}_{col2}_comb"] = cats[len(train_df) :]

# Text TF-IDF Vectorization + SVD Decomposition & Ridge OOF
skf_text = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for col in text_cols:
    train_text = train_df[col].fillna("").astype(str)
    test_text = test_df[col].fillna("").astype(str)

    train_feat[f"{col}_len"] = train_text.str.len()
    test_feat[f"{col}_len"] = test_text.str.len()
    train_feat[f"{col}_word_count"] = train_text.str.split().str.len()
    test_feat[f"{col}_word_count"] = test_text.str.split().str.len()

    tfidf_word = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    tfidf_char = TfidfVectorizer(max_features=5000, analyzer="char", ngram_range=(2, 4))

    tfidf_word.fit(train_text)
    tfidf_char.fit(train_text)

    train_word = tfidf_word.transform(train_text)
    train_char = tfidf_char.transform(train_text)
    train_comb = hstack([train_word, train_char])

    test_word = tfidf_word.transform(test_text)
    test_char = tfidf_char.transform(test_text)
    test_comb = hstack([test_word, test_char])

    oof_ridge = np.zeros(len(train_df))
    test_ridge = np.zeros(len(test_df))

    for tr_i, va_i in skf_text.split(train_feat, y_binned):
        ridge = Ridge(alpha=1.0)
        ridge.fit(train_comb.tocsr()[tr_i], y[tr_i])
        oof_ridge[va_i] = ridge.predict(train_comb.tocsr()[va_i])
        test_ridge += ridge.predict(test_comb.tocsr()) / skf_text.n_splits

    train_feat[f"{col}_ridge_oof"] = oof_ridge
    test_feat[f"{col}_ridge_oof"] = test_ridge

    n_comp = min(16, train_comb.shape[1] - 1)
    if n_comp > 0:
        svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
        train_svd = svd.fit_transform(train_comb)
        test_svd = svd.transform(test_comb)

        for k in range(train_svd.shape[1]):
            train_feat[f"{col}_svd_{k}"] = train_svd[:, k]
            test_feat[f"{col}_svd_{k}"] = test_svd[:, k]

# Numeric Feature Extraction & Group Aggregations
for col in num_cols:
    train_feat[col] = train_df[col].values
    test_feat[col] = test_df[col].values

    train_feat[f"{col}_isna"] = train_df[col].isna().astype(int)
    test_feat[f"{col}_isna"] = test_df[col].isna().astype(int)

    med = train_df[col].median()
    med = 0.0 if pd.isna(med) else med
    train_feat[col] = train_feat[col].fillna(med)
    test_feat[col] = test_feat[col].fillna(med)

if len(num_cols) > 0 and len(cat_cols) > 0:
    primary_cat = cat_cols[0]
    for num_c in num_cols[:3]:
        grp = train_df.groupby(primary_cat)[num_c].agg(["mean", "std"]).reset_index()
        grp.columns = [
            primary_cat,
            f"{num_c}_by_{primary_cat}_mean",
            f"{num_c}_by_{primary_cat}_std",
        ]

        train_merged = train_df[[primary_cat]].merge(grp, on=primary_cat, how="left")
        test_merged = test_df[[primary_cat]].merge(grp, on=primary_cat, how="left")

        train_feat[f"{num_c}_by_{primary_cat}_mean"] = (
            train_merged[f"{num_c}_by_{primary_cat}_mean"].fillna(0).values
        )
        train_feat[f"{num_c}_by_{primary_cat}_std"] = (
            train_merged[f"{num_c}_by_{primary_cat}_std"].fillna(0).values
        )
        test_feat[f"{num_c}_by_{primary_cat}_mean"] = (
            test_merged[f"{num_c}_by_{primary_cat}_mean"].fillna(0).values
        )
        test_feat[f"{num_c}_by_{primary_cat}_std"] = (
            test_merged[f"{num_c}_by_{primary_cat}_std"].fillna(0).values
        )

# Strict Out-Of-Fold Target Encoding
skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

stats = ["mean", "std", "min", "max"]

for cat_c in cat_cols[:3]:
    for stat in stats:
        train_feat[f"{cat_c}_target_enc_{stat}"] = np.nan
        test_feat[f"{cat_c}_target_enc_{stat}"] = np.nan

        cat_series_full = train_df[cat_c].astype(str)
        grp_full = pd.Series(y).groupby(cat_series_full).agg(stat)
        te_map_full = grp_full.to_dict()
        global_stat_full = getattr(pd.Series(y), stat)()
        if pd.isna(global_stat_full):
            global_stat_full = 0.0

        test_feat[f"{cat_c}_target_enc_{stat}"] = (
            test_df[cat_c].astype(str).map(te_map_full).fillna(global_stat_full).fillna(0.0)
        )

for fold, (train_idx, val_idx) in enumerate(skf_te.split(train_feat, y_binned)):
    y_tr_raw = y[train_idx]
    for cat_c in cat_cols[:3]:
        cat_series = train_df.iloc[train_idx][cat_c].astype(str)
        grp = pd.Series(y_tr_raw).groupby(cat_series)
        for stat in stats:
            te_map = grp.agg(stat).to_dict()
            global_stat = getattr(pd.Series(y_tr_raw), stat)()
            if pd.isna(global_stat):
                global_stat = 0.0

            col_idx = train_feat.columns.get_loc(f"{cat_c}_target_enc_{stat}")
            mapped_vals = (
                train_df.iloc[val_idx][cat_c]
                .astype(str)
                .map(te_map)
                .fillna(global_stat)
                .fillna(0.0)
                .values
            )
            train_feat.iloc[val_idx, col_idx] = mapped_vals

# Format Final Feature Matrices
feature_cols = [c for c in train_feat.columns if c != "id"]
X_train = train_feat[feature_cols].copy()
X_test = test_feat[feature_cols].copy()

for col in feature_cols:
    med = X_train[col].median()
    med = 0.0 if pd.isna(med) else med
    X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan).fillna(med).values
    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan).fillna(med).values


# ------------------------------------------------------------------------------
# 4. PyTorch Tabular ResNet Specification
# ------------------------------------------------------------------------------
class TabularResBlock(nn.Module):

    def __init__(self, features: int, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, features),
            nn.BatchNorm1d(features),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, hidden_dim * 2, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)


# ------------------------------------------------------------------------------
# 5. Cross-Validation & Model Training Strategy
# ------------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cb = np.zeros(len(X_train))
oof_nn = np.zeros(len(X_train))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cb = np.zeros(len(X_test))
test_nn = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_binned)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    X_te = X_test.copy()

    # --- 1. LightGBM ---
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED + fold,
        objective="regression",
        verbosity=-1,
        n_jobs=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = np.clip(model_lgb.predict(X_va), 0, None)
    test_lgb += np.clip(model_lgb.predict(X_te), 0, None) / skf.n_splits

    # --- 2. XGBoost ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=SEED + fold,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = np.clip(model_xgb.predict(X_va), 0, None)
    test_xgb += np.clip(model_xgb.predict(X_te), 0, None) / skf.n_splits

    # --- 3. CatBoost ---
    model_cb = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=SEED + fold,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=0,
        thread_count=-1,
    )
    model_cb.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[val_idx] = np.clip(model_cb.predict(X_va), 0, None)
    test_cb += np.clip(model_cb.predict(X_te), 0, None) / skf.n_splits

    # --- 4. PyTorch Tabular ResNet ---
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    y_scaler = StandardScaler()
    y_tr_sc = y_scaler.fit_transform(y_tr.reshape(-1, 1)).astype(np.float32)
    y_va_sc = y_scaler.transform(y_va.reshape(-1, 1)).astype(np.float32)

    train_ds = TensorDataset(
        torch.tensor(X_tr_sc, dtype=torch.float32),
        torch.tensor(y_tr_sc, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_sc, dtype=torch.float32),
        torch.tensor(y_va_sc, dtype=torch.float32),
    )

    bs = min(128, max(32, len(X_tr) // 10))
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=True if len(X_tr) > bs else False,
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    nn_model = TabularResNet(num_features=X_tr.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience, patience_counter = 12, 0
    best_weights = None

    for epoch in range(60):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        nn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * len(bx)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = nn_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_weights is not None:
        nn_model.load_state_dict(best_weights)

    nn_model.eval()
    with torch.no_grad():
        val_pred_sc = (
            nn_model(torch.tensor(X_va_sc, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
        te_pred_sc = (
            nn_model(torch.tensor(X_te_sc, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )

    val_pred_raw = y_scaler.inverse_transform(val_pred_sc).squeeze()
    te_pred_raw = y_scaler.inverse_transform(te_pred_sc).squeeze()

    oof_nn[val_idx] = np.clip(val_pred_raw, 0, None)
    test_nn += np.clip(te_pred_raw, 0, None) / skf.n_splits


# ------------------------------------------------------------------------------
# 6. Ensemble Blending Optimization
# ------------------------------------------------------------------------------
def ensemble_loss(weights):
    w = np.maximum(0, weights)
    if np.sum(w) > 0:
        w = w / np.sum(w)
    blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cb + w[3] * oof_nn
    return root_mean_squared_error(y, blend)


res = minimize(
    ensemble_loss,
    x0=[0.25, 0.25, 0.25, 0.25],
    method="SLSQP",
    bounds=[(0, 1)] * 4,
    constraints={"type": "eq", "fun": lambda w: 1.0 - np.sum(w)},
)

weights = np.maximum(0, res.x)
weights /= np.sum(weights)

oof_ensemble = (
    weights[0] * oof_lgb
    + weights[1] * oof_xgb
    + weights[2] * oof_cb
    + weights[3] * oof_nn
)
test_ensemble = (
    weights[0] * test_lgb
    + weights[1] * test_xgb
    + weights[2] * test_cb
    + weights[3] * test_nn
)

# Save Processed Datasets
os.makedirs("./working", exist_ok=True)
train_feat[target_col] = y
train_feat.to_parquet("./working/processed_train.parquet", index=False)
test_feat.to_parquet("./working/processed_test.parquet", index=False)

# ------------------------------------------------------------------------------
# 7. Generate Submission & Output Validation Score
# ------------------------------------------------------------------------------
os.makedirs("./submission", exist_ok=True)
submission_df = pd.DataFrame(
    {"id": test_ids.astype(int), "Size of Purchase (MW)": test_ensemble}
)
submission_df.to_csv("./submission/submission.csv", index=False)

final_rmse = root_mean_squared_error(y, oof_ensemble)
print(f"Final Validation Score: {final_rmse}")