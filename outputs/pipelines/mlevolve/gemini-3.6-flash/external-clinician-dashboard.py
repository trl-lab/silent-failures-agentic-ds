import copy
import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# Ensure seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Ensure working and submission directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ==========================================
# STEP 1: DATA PROCESSING & FEATURE ENGINEERING
# ==========================================
print("--- Loading Raw Parquet Datasets ---")
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Score"
assert (
    target_col in train_df.columns
), f"Target column '{target_col}' missing from train data."

y_train = train_df[target_col].values.astype(np.float32)

# ID handling
id_col = (
    "id" if "id" in test_df.columns else ("ID" if "ID" in test_df.columns else None)
)
if id_col is not None:
    test_ids = test_df[id_col].values
else:
    test_ids = np.arange(len(test_df))

train_features = train_df.drop(columns=[target_col], errors="ignore").copy()
test_features = test_df.copy()

n_train = len(train_features)
combined_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# Drop index/ID columns
cols_to_drop = [
    c
    for c in [id_col, "index", "Unnamed: 0"]
    if c is not None and c in combined_df.columns
]
if cols_to_drop:
    combined_df.drop(columns=cols_to_drop, inplace=True)

# Feature categorizations
datetime_cols = []
for col in combined_df.columns:
    if pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        datetime_cols.append(col)
    elif combined_df[col].dtype == "object":
        try:
            parsed = pd.to_datetime(combined_df[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(combined_df):
                combined_df[col] = parsed
                datetime_cols.append(col)
        except Exception:
            pass

categorical_cols = [
    col for col in combined_df.select_dtypes(include=["object", "category"]).columns
]
numerical_cols = [col for col in combined_df.select_dtypes(include=["number"]).columns]

# Datetime Feature Extraction
for col in datetime_cols:
    combined_df[f"{col}_year"] = combined_df[col].dt.year
    combined_df[f"{col}_month"] = combined_df[col].dt.month
    combined_df[f"{col}_day"] = combined_df[col].dt.day
    combined_df[f"{col}_dayofweek"] = combined_df[col].dt.dayofweek
    combined_df[f"{col}_quarter"] = combined_df[col].dt.quarter
    combined_df.drop(columns=[col], inplace=True)

numerical_cols = [col for col in combined_df.select_dtypes(include=["number"]).columns]

# Missing Value Indicators
for col in numerical_cols:
    null_count = combined_df[col].isnull().sum()
    if null_count > 0:
        combined_df[f"{col}_isna"] = combined_df[col].isnull().astype(np.int8)

# Frequency Encoding for Categorical Variables (fit on train only)
for col in categorical_cols:
    train_cat = combined_df.iloc[:n_train][col]
    freq = train_cat.value_counts(normalize=True).to_dict()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq).fillna(0).astype(np.float32)

# Row-level Summary Metrics
combined_df["row_mean"] = combined_df[numerical_cols].mean(axis=1).astype(np.float32)
combined_df["row_std"] = combined_df[numerical_cols].std(axis=1).fillna(0).astype(np.float32)
combined_df["row_min"] = combined_df[numerical_cols].min(axis=1).astype(np.float32)
combined_df["row_max"] = combined_df[numerical_cols].max(axis=1).astype(np.float32)
combined_df["row_missing_count"] = combined_df[numerical_cols].isnull().sum(axis=1).astype(np.float32)

# Quantile Rank Transformations (fit on train only)
for col in numerical_cols:
    train_vals = combined_df.iloc[:n_train][col].dropna()
    if len(train_vals) > 0:
        combined_df[f"{col}_rank"] = combined_df[col].apply(lambda x: (train_vals < x).mean() if pd.notna(x) else np.nan).astype(np.float32)

# Non-linear Skewness-Reducing Transformations
for col in numerical_cols:
    skew_val = combined_df[col].skew()
    if pd.notna(skew_val) and abs(skew_val) > 1.0:
        min_v = combined_df[col].min()
        if min_v >= 0:
            combined_df[f"{col}_log1p"] = np.log1p(combined_df[col]).astype(np.float32)
        else:
            combined_df[f"{col}_log1p"] = (np.sign(combined_df[col]) * np.log1p(np.abs(combined_df[col]))).astype(np.float32)

# Correlation-guided Non-linear Numerical Interaction Features
corrs = {}
train_num = combined_df.iloc[:n_train][numerical_cols]
for col in numerical_cols:
    valid_mask = train_num[col].notna()
    if valid_mask.sum() > 0:
        corr_val = np.corrcoef(train_num[col].fillna(0)[valid_mask], y_train[valid_mask])[0, 1]
        corrs[col] = abs(corr_val) if pd.notna(corr_val) else 0.0
    else:
        corrs[col] = 0.0

sorted_num_cols = sorted(corrs.keys(), key=lambda x: corrs[x], reverse=True)
top_num_cols = sorted_num_cols[:5] if len(sorted_num_cols) >= 5 else sorted_num_cols

for i in range(len(top_num_cols)):
    for j in range(i + 1, len(top_num_cols)):
        c1, c2 = top_num_cols[i], top_num_cols[j]
        combined_df[f"{c1}_prod_{c2}"] = (combined_df[c1] * combined_df[c2]).astype(np.float32)
        combined_df[f"{c1}_ratio_{c2}"] = (
            combined_df[c1] / (combined_df[c2] + 1e-5)
        ).astype(np.float32)
        combined_df[f"{c1}_diff_{c2}"] = (combined_df[c1] - combined_df[c2]).astype(
            np.float32
        )

# PCA Latent Components & KMeans Cluster Distance Features
num_cols_for_latent = [c for c in numerical_cols if c in combined_df.columns]
imputer_lat = SimpleImputer(strategy="median")
scaler_lat = StandardScaler()

lat_train_raw = imputer_lat.fit_transform(combined_df.iloc[:n_train][num_cols_for_latent])
lat_all_raw = imputer_lat.transform(combined_df[num_cols_for_latent])

lat_train_scaled = scaler_lat.fit_transform(lat_train_raw)
lat_all_scaled = scaler_lat.transform(lat_all_raw)

pca_model = PCA(n_components=5, random_state=42)
pca_feats = pca_model.fit_transform(lat_train_scaled)
pca_all_feats = pca_model.transform(lat_all_scaled)

for p_i in range(5):
    combined_df[f"pca_{p_i}"] = pca_all_feats[:, p_i].astype(np.float32)

kmeans_model = KMeans(n_clusters=8, random_state=42, n_init=10)
kmeans_model.fit(lat_train_scaled)
kmeans_all_dists = kmeans_model.transform(lat_all_scaled)

for k_i in range(8):
    combined_df[f"kmeans_dist_{k_i}"] = kmeans_all_dists[:, k_i].astype(np.float32)

# Group Aggregation Features across Categorical Variables (computed on train only)
if len(categorical_cols) > 0 and len(numerical_cols) > 0:
    train_sub = combined_df.iloc[:n_train]
    for cat_c in categorical_cols:
        for num_c in numerical_cols[:3]:
            grp_stats = train_sub.groupby(cat_c, observed=False)[num_c].agg(["mean", "std", "min", "max"])
            combined_df[f"{num_c}_by_{cat_c}_mean"] = combined_df[cat_c].map(grp_stats["mean"]).astype(np.float32)
            combined_df[f"{num_c}_by_{cat_c}_std"] = combined_df[cat_c].map(grp_stats["std"]).fillna(0).astype(np.float32)
            combined_df[f"{num_c}_by_{cat_c}_min"] = combined_df[cat_c].map(grp_stats["min"]).astype(np.float32)
            combined_df[f"{num_c}_by_{cat_c}_max"] = combined_df[cat_c].map(grp_stats["max"]).astype(np.float32)

# Unify categorical types
for cat_c in categorical_cols:
    combined_df[cat_c] = combined_df[cat_c].astype("category")

# Split back to train/test
X_train_df = combined_df.iloc[:n_train].copy().reset_index(drop=True)
X_test_df = combined_df.iloc[n_train:].copy().reset_index(drop=True)

# Out-Of-Fold Target Encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(kf.split(X_train_df, y_train))

for cat_c in categorical_cols:
    X_train_df[f"{cat_c}_te"] = np.nan
    global_mean = float(y_train.mean())

    for train_idx, val_idx in cv_splits:
        fold_train_cat = X_train_df.iloc[train_idx][cat_c]
        fold_train_y = y_train[train_idx]
        cat_df = pd.DataFrame({"cat": fold_train_cat, "target": fold_train_y})
        te_map = cat_df.groupby("cat", observed=True)["target"].mean().to_dict()
        X_train_df.loc[val_idx, f"{cat_c}_te"] = (
            X_train_df.iloc[val_idx][cat_c]
            .map(te_map)
            .astype(float)
            .fillna(global_mean)
            .values
        )

    full_cat_df = pd.DataFrame({"cat": X_train_df[cat_c], "target": y_train})
    full_te_map = full_cat_df.groupby("cat", observed=True)["target"].mean().to_dict()
    X_test_df[f"{cat_c}_te"] = (
        X_test_df[cat_c].map(full_te_map).astype(float).fillna(global_mean)
    )

# Preserve native missing values (NaNs) in X_train_df and X_test_df for GBDT models.


# ==========================================
# STEP 2: MODEL DESIGN ARCHITECTURES
# ==========================================
class TabularResidualBlock(nn.Module):
    """Residual Block for Deep Tabular Neural Network."""

    def __init__(
        self, in_features: int, hidden_features: int, dropout_rate: float = 0.2
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    """Deep Tabular ResNet model for continuous score regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dim, hidden_dim * 2, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


lgb_params = {
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

xgb_params = {
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
    "tree_method": "hist",
}

cat_params = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 2500,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": 0,
}

# ==========================================
# STEP 3: CROSS-VALIDATION TRAINING & ENSEMBLE
# ==========================================
# Format categorical data for CatBoost and PyTorch
X_train_cat = X_train_df.copy()
X_test_cat = X_test_df.copy()
for col in categorical_cols:
    X_train_cat[col] = (
        X_train_cat[col].astype(str).replace("nan", "missing").fillna("missing")
    )
    X_test_cat[col] = (
        X_test_cat[col].astype(str).replace("nan", "missing").fillna("missing")
    )

X_train_nn = X_train_df.copy()
X_test_nn = X_test_df.copy()
for col in categorical_cols:
    X_train_nn[col] = X_train_nn[col].cat.codes.astype(np.float32)
    X_test_nn[col] = X_test_nn[col].cat.codes.astype(np.float32)

X_train_nn = X_train_nn.fillna(0.0).values.astype(np.float32)
X_test_nn = X_test_nn.fillna(0.0).values.astype(np.float32)

n_test = len(X_test_df)
oof_lgb = np.zeros(n_train, dtype=np.float32)
oof_xgb = np.zeros(n_train, dtype=np.float32)
oof_cat = np.zeros(n_train, dtype=np.float32)
oof_nn = np.zeros(n_train, dtype=np.float32)
oof_hgb = np.zeros(n_train, dtype=np.float32)
oof_et = np.zeros(n_train, dtype=np.float32)
oof_ridge = np.zeros(n_train, dtype=np.float32)

test_preds_lgb = np.zeros(n_test, dtype=np.float32)
test_preds_xgb = np.zeros(n_test, dtype=np.float32)
test_preds_cat = np.zeros(n_test, dtype=np.float32)
test_preds_nn = np.zeros(n_test, dtype=np.float32)
test_preds_hgb = np.zeros(n_test, dtype=np.float32)
test_preds_et = np.zeros(n_test, dtype=np.float32)
test_preds_ridge = np.zeros(n_test, dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

for fold, (train_idx, val_idx) in enumerate(cv_splits):
    X_tr, y_tr = X_train_df.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_df.iloc[val_idx], y_train[val_idx]

    # 1. LightGBM Model
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_preds_lgb += lgb_model.predict(X_test_df) / len(cv_splits)

    # 2. XGBoost Model
    xgb_model = xgb.XGBRegressor(
        **xgb_params, enable_categorical=True, early_stopping_rounds=50
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_preds_xgb += xgb_model.predict(X_test_df) / len(cv_splits)

    # 3. CatBoost Model
    X_tr_cat, X_va_cat = X_train_cat.iloc[train_idx], X_train_cat.iloc[val_idx]
    cat_model = CatBoostRegressor(**cat_params, early_stopping_rounds=50)
    cat_model.fit(
        X_tr_cat,
        y_tr,
        eval_set=(X_va_cat, y_va),
        cat_features=categorical_cols if len(categorical_cols) > 0 else None,
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict(X_va_cat)
    test_preds_cat += cat_model.predict(X_test_cat) / len(cv_splits)

    # 4. PyTorch TabularResNet
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_nn[train_idx])
    X_va_scaled = scaler.transform(X_train_nn[val_idx])
    X_te_scaled = scaler.transform(X_test_nn)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        drop_last=(len(train_ds) > 256),
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    nn_model = TabularResNet(input_dim=X_train_nn.shape[1]).to(device)
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

    best_val_loss = float("inf")
    best_weights = None
    patience, patience_cnt = 8, 0

    for epoch in range(40):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_loss, val_cnt = 0.0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = nn_model(bx)
                val_loss += criterion(out, by).item() * bx.size(0)
                val_cnt += bx.size(0)

        epoch_rmse = np.sqrt(val_loss / val_cnt)
        if epoch_rmse < best_val_loss:
            best_val_loss = epoch_rmse
            best_weights = copy.deepcopy(nn_model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    nn_model.load_state_dict(best_weights)
    nn_model.eval()

    with torch.no_grad():
        oof_nn[val_idx] = (
            nn_model(torch.tensor(X_va_scaled, dtype=torch.float32).to(device))
            .cpu()
            .numpy()
        )
        test_preds_nn += nn_model(
            torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        ).cpu().numpy() / len(cv_splits)

    # 5. HistGradientBoostingRegressor
    hgb_model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_depth=8,
        l2_regularization=1.0,
        random_state=42,
    )
    hgb_model.fit(X_train_nn[train_idx], y_tr)
    oof_hgb[val_idx] = hgb_model.predict(X_train_nn[val_idx])
    test_preds_hgb += hgb_model.predict(X_test_nn) / len(cv_splits)

    # 6. ExtraTreesRegressor
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    et_model.fit(X_train_nn[train_idx], y_tr)
    oof_et[val_idx] = et_model.predict(X_train_nn[val_idx])
    test_preds_et += et_model.predict(X_test_nn) / len(cv_splits)

    # 7. Ridge Regressor
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_tr_scaled, y_tr)
    oof_ridge[val_idx] = ridge_model.predict(X_va_scaled)
    test_preds_ridge += ridge_model.predict(X_te_scaled) / len(cv_splits)

# Level-2 Out-Of-Fold Stacking Meta-Learner Ensemble
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_nn, oof_hgb, oof_et, oof_ridge])
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_nn, test_preds_hgb, test_preds_et, test_preds_ridge]
)

# Supplement prediction matrix with summary statistics (std, min, max, mean)
oof_meta = np.column_stack([
    oof_matrix,
    np.mean(oof_matrix, axis=1),
    np.std(oof_matrix, axis=1),
    np.min(oof_matrix, axis=1),
    np.max(oof_matrix, axis=1)
])

test_meta = np.column_stack([
    test_matrix,
    np.mean(test_matrix, axis=1),
    np.std(test_matrix, axis=1),
    np.min(test_matrix, axis=1),
    np.max(test_matrix, axis=1)
])

meta_kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_meta_preds = np.zeros(n_train, dtype=np.float32)
test_meta_preds = np.zeros(n_test, dtype=np.float32)

for meta_tr_idx, meta_va_idx in meta_kf.split(oof_meta, y_train):
    m_X_tr, m_y_tr = oof_meta[meta_tr_idx], y_train[meta_tr_idx]
    m_X_va, m_y_va = oof_meta[meta_va_idx], y_train[meta_va_idx]

    meta_ridge = Ridge(alpha=1.0, random_state=42)
    meta_ridge.fit(m_X_tr, m_y_tr)

    meta_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    meta_gb.fit(m_X_tr, m_y_tr)

    val_pred_ridge = meta_ridge.predict(m_X_va)
    val_pred_gb = meta_gb.predict(m_X_va)
    oof_meta_preds[meta_va_idx] = 0.5 * val_pred_ridge + 0.5 * val_pred_gb

    test_pred_ridge = meta_ridge.predict(test_meta)
    test_pred_gb = meta_gb.predict(test_meta)
    test_meta_preds += (0.5 * test_pred_ridge + 0.5 * test_pred_gb) / len(cv_splits)

final_validation_score = np.sqrt(mean_squared_error(y_train, oof_meta_preds))

# Output Predictions and Submission Generation
final_test_preds = np.clip(
    test_meta_preds, a_min=y_train.min(), a_max=y_train.max()
)

submission_path = "./submission/submission.csv"
sub_df = pd.DataFrame({"id": test_ids, "Score": final_test_preds})
sub_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {final_validation_score}")