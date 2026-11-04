import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------
# 1. Data Loading and Feature Engineering
# ---------------------------------------------------------
print("Loading raw datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Total Contribution (USD)"
id_col = "id" if "id" in test_df.columns else None

if id_col is None or id_col not in test_df.columns:
    test_ids = np.arange(len(test_df))
else:
    test_ids = test_df[id_col].values

# Target extraction and log-1p transformation for scale stabilization
y_train_raw = train_df[target_col].values.astype(np.float64)
y_train_log = np.log1p(np.maximum(0, y_train_raw))

train_features = [c for c in train_df.columns if c != target_col and c != id_col]
test_features = [c for c in test_df.columns if c != id_col]
common_cols = [c for c in train_features if c in test_features]

df_all = pd.concat(
    [train_df[common_cols], test_df[common_cols]], axis=0, ignore_index=True
)
n_train = len(train_df)

# Date processing
date_cols = []
for col in list(df_all.columns):
    if df_all[col].dtype == "object" or pd.api.types.is_datetime64_any_dtype(
        df_all[col]
    ):
        try:
            converted = pd.to_datetime(df_all[col], errors="coerce")
            if converted.notnull().sum() > 0.5 * len(df_all):
                df_all[col + "_year"] = converted.dt.year
                df_all[col + "_month"] = converted.dt.month
                df_all[col + "_quarter"] = converted.dt.quarter
                df_all[col + "_dayofweek"] = converted.dt.dayofweek
                date_cols.append(col)
        except Exception:
            pass

num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c not in date_cols]

# Separate train/test slice for leak-free feature engineering
df_train_temp = df_all.iloc[:n_train]

# Frequency encoding for categoricals using train split only
for c in cat_cols:
    freq_map = df_train_temp[c].value_counts(normalize=True).to_dict()
    df_all[f"{c}_freq"] = df_all[c].map(freq_map).fillna(0).astype(np.float32)

# Missingness indicators
for c in num_cols:
    if df_all[c].isnull().sum() > 0:
        df_all[f"{c}_isnan"] = df_all[c].isnull().astype(np.float32)

# Categorical aggregations over key numericals computed strictly on train split
if len(cat_cols) > 0 and len(num_cols) > 0:
    primary_cat = cat_cols[0]
    primary_num = num_cols[0]
    agg_df = (
        df_train_temp.groupby(primary_cat)[primary_num]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    agg_df.columns = [primary_cat] + [
        f"{primary_cat}_{primary_num}_{stat}" for stat in ["mean", "std", "min", "max"]
    ]
    df_all = df_all.merge(agg_df, on=primary_cat, how="left")

for c in cat_cols:
    df_all[c] = df_all[c].astype("category").cat.codes

engineered_num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()

df_train = df_all.iloc[:n_train].copy()
df_test = df_all.iloc[n_train:].copy()

# Impute using train set medians to avoid data leakage
train_medians = df_train[engineered_num_cols].median().to_dict()
df_train.fillna(train_medians, inplace=True)
df_test.fillna(train_medians, inplace=True)

# Out-of-fold target encoding strictly on train split
kf = KFold(n_splits=5, shuffle=True, random_state=42)
if len(cat_cols) > 0:
    for c in cat_cols[:3]:
        target_enc_col = f"{c}_target_enc"
        df_train[target_enc_col] = np.nan
        for train_idx, val_idx in kf.split(df_train):
            fold_tr = df_train.iloc[train_idx]
            encoding_map = (
                pd.Series(y_train_log[train_idx], index=fold_tr[c])
                .groupby(fold_tr[c])
                .mean()
            )
            df_train.iloc[val_idx, df_train.columns.get_loc(target_enc_col)] = (
                df_train.iloc[val_idx][c].map(encoding_map)
            )

        global_map = (
            pd.Series(y_train_log, index=df_train[c]).groupby(df_train[c]).mean()
        )
        df_train[target_enc_col].fillna(y_train_log.mean(), inplace=True)
        df_test[target_enc_col] = df_test[c].map(global_map).fillna(y_train_log.mean())

final_feature_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train[final_feature_cols].values)
X_test_scaled = scaler.transform(df_test[final_feature_cols].values)

# Train/Validation Split (80/20)
train_idx, val_idx = next(
    KFold(n_splits=5, shuffle=True, random_state=42).split(X_train_scaled)
)

X_tr = X_train_scaled[train_idx]
y_tr_log = y_train_log[train_idx]
y_tr_raw = y_train_raw[train_idx]

X_va = X_train_scaled[val_idx]
y_va_log = y_train_log[val_idx]
y_va_raw = y_train_raw[val_idx]

X_te = X_test_scaled

X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

# ---------------------------------------------------------
# 2. Model Training
# ---------------------------------------------------------
print("Training LightGBM Model...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
)
lgb_model.fit(
    X_tr,
    y_tr_log,
    eval_set=[(X_va, y_va_log)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)
lgb_val_log = lgb_model.predict(X_va)
lgb_test_log = lgb_model.predict(X_te)

print("Training XGBoost Model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    early_stopping_rounds=50,
    n_jobs=-1,
)
xgb_model.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
xgb_val_log = xgb_model.predict(X_va)
xgb_test_log = xgb_model.predict(X_te)

print("Training ExtraTrees Model...")
et_model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
et_model.fit(X_tr, y_tr_log)
et_val_log = et_model.predict(X_va)
et_test_log = et_model.predict(X_te)


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
        )
        self.shortcut_b2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = x + self.block1(x)
        x = self.shortcut_b2(x) + self.block2(x)
        return self.head(x)


print("Training PyTorch Tabular ResNet...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr_log, dtype=torch.float32).unsqueeze(1)
X_va_t = torch.tensor(X_va, dtype=torch.float32)
X_te_t = torch.tensor(X_te, dtype=torch.float32)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)

nn_model = TabularResNet(input_dim=X_tr.shape[1]).to(device)
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
criterion = nn.MSELoss()

nn_model.train()
for epoch in range(40):
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = nn_model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
    scheduler.step()

nn_model.eval()
with torch.no_grad():
    nn_val_log = nn_model(X_va_t.to(device)).cpu().numpy().flatten()
    nn_test_log = nn_model(X_te_t.to(device)).cpu().numpy().flatten()

# ---------------------------------------------------------
# 3. Blending and Submission Generation
# ---------------------------------------------------------
lgb_val_raw = np.expm1(np.clip(lgb_val_log, 0, None))
xgb_val_raw = np.expm1(np.clip(xgb_val_log, 0, None))
et_val_raw = np.expm1(np.clip(et_val_log, 0, None))
nn_val_raw = np.expm1(np.clip(nn_val_log, 0, None))

lgb_test_raw = np.expm1(np.clip(lgb_test_log, 0, None))
xgb_test_raw = np.expm1(np.clip(xgb_test_log, 0, None))
et_test_raw = np.expm1(np.clip(et_test_log, 0, None))
nn_test_raw = np.expm1(np.clip(nn_test_log, 0, None))

P_val = np.column_stack([lgb_val_raw, xgb_val_raw, et_val_raw, nn_val_raw])
P_test = np.column_stack([lgb_test_raw, xgb_test_raw, et_test_raw, nn_test_raw])


def objective(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    pred = P_val @ w
    return np.sqrt(np.mean((pred - y_va_raw) ** 2))


init_w = [0.25, 0.25, 0.25, 0.25]
bounds = [(0, 1)] * 4
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    objective, init_w, method="SLSQP", bounds=bounds, constraints=constraints
)
opt_w = opt_res.x / np.sum(opt_res.x)

val_preds = np.maximum(0, P_val @ opt_w)
test_preds = np.maximum(0, P_test @ opt_w)

val_rmse = np.sqrt(mean_squared_error(y_va_raw, val_preds))

sub_df = pd.DataFrame({"id": test_ids, "Total Contribution (USD)": test_preds})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_rmse}")