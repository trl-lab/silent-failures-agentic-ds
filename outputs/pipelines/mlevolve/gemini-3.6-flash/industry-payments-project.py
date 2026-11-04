import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# STEP 1: DATA PROCESSING & FEATURE ENGINEERING
# ==========================================

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "projectPayment"

y_train = train_df[target_col].copy() if target_col in train_df.columns else None

df_train = train_df.drop(columns=[target_col], errors="ignore").copy()
df_test = test_df.drop(columns=[target_col], errors="ignore").copy()

df_train["_is_train"] = 1
df_test["_is_train"] = 0

combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# Datetime Processing
datetime_cols = []
for col in combined.columns:
    if col in ["_is_train", "id"]:
        continue
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        datetime_cols.append(col)
    elif combined[col].dtype == "object":
        try:
            converted = pd.to_datetime(combined[col], format="mixed", errors="coerce")
            if converted.notna().sum() / len(converted) > 0.5:
                combined[col] = converted
                datetime_cols.append(col)
        except Exception:
            pass

for col in datetime_cols:
    combined[f"{col}_year"] = combined[col].dt.year
    combined[f"{col}_month"] = combined[col].dt.month
    combined[f"{col}_day"] = combined[col].dt.day
    combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
    combined[f"{col}_dayofyear"] = combined[col].dt.dayofyear
    combined[f"{col}_quarter"] = combined[col].dt.quarter
    combined[f"{col}_sin_month"] = np.sin(2 * np.pi * combined[col].dt.month / 12.0)
    combined[f"{col}_cos_month"] = np.cos(2 * np.pi * combined[col].dt.month / 12.0)
    combined.drop(columns=[col], inplace=True)

# Split back to train and test
train_proc = (
    combined[combined["_is_train"] == 1]
    .drop(columns=["_is_train"])
    .reset_index(drop=True)
)
test_proc = (
    combined[combined["_is_train"] == 0]
    .drop(columns=["_is_train"])
    .reset_index(drop=True)
)

if y_train is not None:
    train_proc[target_col] = y_train.values

# Categorical & Missing Value Processing
cat_cols = [
    c
    for c in train_proc.columns
    if c not in [target_col, "id"]
    and (train_proc[c].dtype == "object" or train_proc[c].dtype.name == "category")
]
num_cols = [c for c in train_proc.columns if c not in [target_col, "id"] + cat_cols]

# Fill missing categorical values as string missing indicator
for c in cat_cols:
    train_proc[c] = train_proc[c].fillna("__MISSING__").astype(str)
    test_proc[c] = test_proc[c].fillna("__MISSING__").astype(str)

# Category-Group Aggregations for Key Continuous Numerical Features
continuous_num_cols = [c for c in num_cols if train_proc[c].nunique() > 10][:5]

for cat_c in cat_cols:
    for num_c in continuous_num_cols:
        stats = train_proc.groupby(cat_c)[num_c].agg(["mean", "std", "min", "max"])
        stats["std"] = stats["std"].fillna(0)
        mean_map = stats["mean"].to_dict()
        std_map = stats["std"].to_dict()
        min_map = stats["min"].to_dict()
        max_map = stats["max"].to_dict()
        global_mean = train_proc[num_c].mean()

        for df in [train_proc, test_proc]:
            g_mean = df[cat_c].map(mean_map).fillna(global_mean)
            g_std = df[cat_c].map(std_map).fillna(0)
            g_min = df[cat_c].map(min_map).fillna(train_proc[num_c].min())
            g_max = df[cat_c].map(max_map).fillna(train_proc[num_c].max())
            df[f"{num_c}_grp_{cat_c}_mean"] = g_mean
            df[f"{num_c}_grp_{cat_c}_std"] = g_std
            df[f"{num_c}_grp_{cat_c}_min"] = g_min
            df[f"{num_c}_grp_{cat_c}_max"] = g_max
            df[f"{num_c}_diff_{cat_c}_mean"] = df[num_c] - g_mean
            df[f"{num_c}_ratio_{cat_c}_mean"] = df[num_c] / (g_mean.abs() + 1e-5)

# Frequency Encoding
for c in cat_cols:
    freq_map = train_proc[c].value_counts(normalize=True).to_dict()
    train_proc[f"{c}_freq"] = train_proc[c].map(freq_map).fillna(0)
    test_proc[f"{c}_freq"] = test_proc[c].map(freq_map).fillna(0)

# Create 5-Fold Split
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_proc["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(train_proc)):
    train_proc.loc[val_idx, "fold"] = fold

# Multi-smoothing Bayesian Smoothed OOF Target Encoding (Leak-free)
if y_train is not None:
    global_mean = train_proc[target_col].mean()
    for m_smooth in [5.0, 20.0]:
        suffix = f"_te_s{int(m_smooth)}"
        for c in cat_cols:
            oof_target_enc = pd.Series(index=train_proc.index, dtype=float)
            for fold, (train_idx, val_idx) in enumerate(kf.split(train_proc)):
                tr = train_proc.iloc[train_idx]
                fold_global_mean = tr[target_col].mean()
                stats = tr.groupby(c)[target_col].agg(["count", "mean"])
                smooth_te = (
                    stats["count"] * stats["mean"] + m_smooth * fold_global_mean
                ) / (stats["count"] + m_smooth)
                cat_smooth_dict = smooth_te.to_dict()
                oof_target_enc.iloc[val_idx] = (
                    train_proc.iloc[val_idx][c].map(cat_smooth_dict).fillna(fold_global_mean)
                )

            train_proc[f"{c}{suffix}"] = oof_target_enc
            full_stats = train_proc.groupby(c)[target_col].agg(["count", "mean"])
            full_smooth_te = (
                full_stats["count"] * full_stats["mean"] + m_smooth * global_mean
            ) / (full_stats["count"] + m_smooth)
            test_proc[f"{c}{suffix}"] = (
                test_proc[c].map(full_smooth_te.to_dict()).fillna(global_mean)
            )

# Convert Categorical Columns to Integer Category Codes
for c in cat_cols:
    cat_categories = sorted(list(set(train_proc[c].unique()).union(set(test_proc[c].unique()))))
    cat_type = pd.CategoricalDtype(categories=cat_categories)
    train_proc[c] = train_proc[c].astype(cat_type).cat.codes
    test_proc[c] = test_proc[c].astype(cat_type).cat.codes

# Numerical Imputation & Transformations
for c in num_cols:
    if train_proc[c].isnull().sum() > 0 or test_proc[c].isnull().sum() > 0:
        train_proc[f"{c}_isna"] = train_proc[c].isnull().astype(int)
        test_proc[f"{c}_isna"] = test_proc[c].isnull().astype(int)

    med = train_proc[c].median()
    train_proc[c] = train_proc[c].fillna(med)
    test_proc[c] = test_proc[c].fillna(med)

    skewness = train_proc[c].skew()
    if skewness > 1.0 and (train_proc[c] >= 0).all():
        train_proc[f"{c}_log1p"] = np.log1p(train_proc[c])
        test_proc[f"{c}_log1p"] = np.log1p(test_proc[c])

feature_num_cols = [c for c in num_cols if c in train_proc.columns]
if len(feature_num_cols) >= 2:
    stds = train_proc[feature_num_cols].std().sort_values(ascending=False)
    top_num = stds.head(5).index.tolist()
    for i in range(len(top_num)):
        for j in range(i + 1, len(top_num)):
            c1, c2 = top_num[i], top_num[j]
            train_proc[f"{c1}_ratio_{c2}"] = train_proc[c1] / (
                train_proc[c2].abs() + 1e-5
            )
            test_proc[f"{c1}_ratio_{c2}"] = test_proc[c1] / (test_proc[c2].abs() + 1e-5)

train_proc.to_parquet("./working/train_processed.parquet", index=False)
test_proc.to_parquet("./working/test_processed.parquet", index=False)


# ==========================================
# STEP 2: MODEL ARCHITECTURES & BUILDERS
# ==========================================


class GatedLinearUnit(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc1(x)) * torch.sigmoid(self.fc2(x))


class TabularResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.glu = GatedLinearUnit(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.glu(x)
        out = self.dropout(out)
        return self.norm(out + residual)


class TabularGatedResNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, dropout=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def get_pytorch_model_and_optimizer(
    num_features: int,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    model = TabularGatedResNet(
        num_features=num_features, hidden_dim=hidden_dim, num_blocks=num_blocks
    )
    criterion = RMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-5
    )
    return model, criterion, optimizer, scheduler


def get_lgb_model(params: dict = None) -> lgb.LGBMRegressor:
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)
    return lgb.LGBMRegressor(**default_params)


def get_xgb_model(params: dict = None) -> xgb.XGBRegressor:
    default_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "enable_categorical": False,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)
    return xgb.XGBRegressor(**default_params)


def get_catboost_model(params: dict = None) -> cb.CatBoostRegressor:
    default_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }
    if params:
        default_params.update(params)
    return cb.CatBoostRegressor(**default_params)


def get_extratrees_model(params: dict = None) -> ExtraTreesRegressor:
    default_params = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)
    return ExtraTreesRegressor(**default_params)


def get_ridge_model(alpha: float = 10.0) -> Ridge:
    return Ridge(alpha=alpha, random_state=42)


# ==========================================
# STEP 3: TRAINING & EVALUATION PIPELINE
# ==========================================

y_true = train_proc[target_col].values.astype(np.float32)

if "id" in test_proc.columns:
    test_ids = test_proc["id"].values.astype(int)
else:
    test_ids = np.arange(len(test_proc), dtype=int)

drop_cols = [target_col, "fold", "id"]
features = [c for c in train_proc.columns if c not in drop_cols]

X_full = train_proc[features].values.astype(np.float32)
X_test_full = test_proc[features].values.astype(np.float32)

X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
X_test_full = np.nan_to_num(X_test_full, nan=0.0, posinf=0.0, neginf=0.0)

num_features = len(features)
num_folds = 5
folds = train_proc["fold"].values

oof_lgb = np.zeros(len(train_proc), dtype=np.float32)
oof_xgb = np.zeros(len(train_proc), dtype=np.float32)
oof_cb = np.zeros(len(train_proc), dtype=np.float32)
oof_ridge = np.zeros(len(train_proc), dtype=np.float32)
oof_nn = np.zeros(len(train_proc), dtype=np.float32)

test_preds_lgb = np.zeros(len(test_proc), dtype=np.float32)
test_preds_xgb = np.zeros(len(test_proc), dtype=np.float32)
test_preds_cb = np.zeros(len(test_proc), dtype=np.float32)
test_preds_ridge = np.zeros(len(test_proc), dtype=np.float32)
test_preds_nn = np.zeros(len(test_proc), dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in range(num_folds):
    train_idx = folds != fold
    val_idx = folds == fold

    X_tr_raw, y_tr = X_full[train_idx], y_true[train_idx]
    X_val_raw, y_val = X_full[val_idx], y_true[val_idx]
    X_ts_raw = X_test_full.copy()

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_ts_scaled = scaler.transform(X_ts_raw)

    # LightGBM
    lgb_model = get_lgb_model()
    lgb_model.fit(
        X_tr_raw,
        y_tr,
        eval_set=[(X_val_raw, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val_raw)
    test_preds_lgb += lgb_model.predict(X_ts_raw) / num_folds

    # XGBoost
    xgb_model = get_xgb_model({"early_stopping_rounds": 50})
    xgb_model.fit(X_tr_raw, y_tr, eval_set=[(X_val_raw, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val_raw)
    test_preds_xgb += xgb_model.predict(X_ts_raw) / num_folds

    # CatBoost
    cb_model = get_catboost_model({"early_stopping_rounds": 50})
    cb_model.fit(X_tr_raw, y_tr, eval_set=[(X_val_raw, y_val)], verbose=False)
    oof_cb[val_idx] = cb_model.predict(X_val_raw)
    test_preds_cb += cb_model.predict(X_ts_raw) / num_folds

    # Ridge Regression
    ridge_model = get_ridge_model()
    ridge_model.fit(X_tr_scaled, y_tr)
    oof_ridge[val_idx] = ridge_model.predict(X_val_scaled)
    test_preds_ridge += ridge_model.predict(X_ts_scaled) / num_folds

    # PyTorch Neural Net
    nn_model, criterion, optimizer, scheduler = get_pytorch_model_and_optimizer(
        num_features=num_features,
        hidden_dim=256,
        num_blocks=3,
        lr=1e-3,
        weight_decay=1e-4,
    )
    nn_model = nn_model.to(device)

    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(torch.tensor(X_ts_scaled, dtype=torch.float32))

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=True, drop_last=False, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    best_val_loss = float("inf")
    best_nn_weights = None

    for epoch in range(40):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = nn_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                val_preds_list.append(nn_model(batch_x).cpu().numpy())
        val_preds_arr = np.concatenate(val_preds_list)
        val_loss = np.sqrt(mean_squared_error(y_val, val_preds_arr))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_weights = {
                k: v.cpu().clone() for k, v in nn_model.state_dict().items()
            }

    if best_nn_weights is not None:
        nn_model.load_state_dict(best_nn_weights)

    nn_model.eval()
    val_preds_list = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            val_preds_list.append(nn_model(batch_x).cpu().numpy())
    oof_nn[val_idx] = np.concatenate(val_preds_list)

    test_preds_list = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            test_preds_list.append(nn_model(batch_x).cpu().numpy())
    test_preds_nn += np.concatenate(test_preds_list) / num_folds

# Stacking via SciPy SLSQP Optimization
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_ridge, oof_nn])
test_matrix = np.column_stack(
    [test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_ridge, test_preds_nn]
)


def blend_objective(weights):
    weights = np.array(weights)
    blend_oof = oof_matrix @ weights
    return np.sqrt(mean_squared_error(y_true, blend_oof))


init_weights = np.ones(5) / 5.0
bounds = [(0.0, 1.0)] * 5
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

res = minimize(
    blend_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = res.x

final_oof_preds = oof_matrix @ best_weights
final_test_preds = test_matrix @ best_weights

# Target domain constraint
min_target_val = y_true.min()
if min_target_val >= 0:
    final_oof_preds = np.maximum(min_target_val, final_oof_preds)
    final_test_preds = np.maximum(min_target_val, final_test_preds)

final_score = np.sqrt(mean_squared_error(y_true, final_oof_preds))

# Export predictions
submission_df = pd.DataFrame({"id": test_ids, "projectPayment": final_test_preds})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")