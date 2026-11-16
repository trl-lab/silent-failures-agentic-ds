import os
import pickle
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# 1. Environment & Reproducibility Setup
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Data Loading
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "youSaveSpend"

if "id" in train_df.columns:
    train_ids = train_df["id"].values
else:
    train_df["id"] = np.arange(len(train_df))
    train_ids = train_df["id"].values

if "id" in test_df.columns:
    test_ids = test_df["id"].values.astype(int)
else:
    test_df["id"] = np.arange(len(test_df))
    test_ids = test_df["id"].values.astype(int)

y_train = train_df[target_col].values.astype(np.float32)
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

train_len = len(X_train)

# Combine for uniform categorical preprocessing (excluding stateful fit statistics)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)


# 3. Domain-Specific Physical Feature Engineering
def add_domain_features(df):
    df = df.copy()

    # Combined & dual-fuel annual and 5-year fuel cost
    f_cost08 = df["fuelCost08"] if "fuelCost08" in df.columns else 0
    f_costA08 = df["fuelCostA08"].fillna(0) if "fuelCostA08" in df.columns else 0
    df["total_fuel_cost"] = f_cost08 + f_costA08
    df["fuel_cost_5yr"] = 5.0 * df["total_fuel_cost"]

    # Fuel cost per comb08 mile ratio
    if "fuelCost08" in df.columns and "comb08" in df.columns:
        df["fuel_cost_per_comb_mile"] = df["fuelCost08"] / (
            df["comb08"].clip(lower=1) + 1e-5
        )

    # Gallons Per Mile (GPM) reciprocals
    if "comb08" in df.columns:
        df["gpm_comb"] = 1.0 / (df["comb08"].clip(lower=1) + 1e-5)
    if "city08" in df.columns:
        df["gpm_city"] = 1.0 / (df["city08"].clip(lower=1) + 1e-5)
    if "highway08" in df.columns:
        df["gpm_highway"] = 1.0 / (df["highway08"].clip(lower=1) + 1e-5)

    # City vs Highway MPG Efficiency Ratio
    if "city08" in df.columns and "highway08" in df.columns:
        df["city_hwy_ratio"] = df["city08"] / (df["highway08"].clip(lower=1) + 1e-5)

    # Displacement per Cylinder
    if "displ" in df.columns and "cylinders" in df.columns:
        df["displ_per_cyl"] = df["displ"] / (df["cylinders"].clip(lower=1) + 1e-5)

    # Cost per Barrel of Oil
    if "barrels08" in df.columns and "fuelCost08" in df.columns:
        df["cost_per_barrel"] = df["fuelCost08"] / (
            df["barrels08"].clip(lower=0.1) + 1e-5
        )

    # Total Annual Petroleum Consumption
    b08 = df["barrels08"] if "barrels08" in df.columns else 0
    bA08 = df["barrelsA08"].fillna(0) if "barrelsA08" in df.columns else 0
    df["total_barrels"] = b08 + bA08

    return df


df_all = add_domain_features(df_all)

# 4. K-Fold Out-of-Fold Target Encoding & Group Aggregations
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
df_train = df_all.iloc[:train_len].copy()
df_test = df_all.iloc[train_len:].copy()
df_train[target_col] = y_train

high_card_cols = [
    c
    for c in ["make", "model", "VClass", "trany", "drive", "fuelType1"]
    if c in df_train.columns
]

smoothing = 10
global_mean = float(y_train.mean())

for col in high_card_cols:
    te_col = f"{col}_te"
    df_train[te_col] = np.nan

    for tr_idx, val_idx in kf.split(df_train):
        fold_tr = df_train.iloc[tr_idx]
        stats = fold_tr.groupby(col)[target_col].agg(["count", "mean"])
        smooth_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        df_train.iloc[val_idx, df_train.columns.get_loc(te_col)] = df_train.iloc[
            val_idx
        ][col].map(smooth_te)

    df_train[te_col] = df_train[te_col].fillna(global_mean)

    full_stats = df_train.groupby(col)[target_col].agg(["count", "mean"])
    full_smooth_te = (
        full_stats["count"] * full_stats["mean"] + smoothing * global_mean
    ) / (full_stats["count"] + smoothing)
    df_test[te_col] = df_test[col].map(full_smooth_te).fillna(global_mean)

df_train = df_train.drop(columns=[target_col])

# Group Aggregation Features
agg_cols = [
    c
    for c in ["fuelCost08", "comb08", "displ", "co2TailpipeGpm", "total_fuel_cost", "fuel_cost_5yr"]
    if c in df_train.columns
]
group_keys = [k for k in ["VClass", "year", "fuelType1"] if k in df_train.columns]

for gk in group_keys:
    for ac in agg_cols:
        group_stats = df_train.groupby(gk)[ac].agg(["mean", "std"])
        mean_map = group_stats["mean"]
        std_map = group_stats["std"]

        df_train[f"{gk}_{ac}_mean"] = df_train[gk].map(mean_map)
        df_train[f"{gk}_{ac}_std"] = df_train[gk].map(std_map)
        df_test[f"{gk}_{ac}_mean"] = df_test[gk].map(mean_map)
        df_test[f"{gk}_{ac}_std"] = df_test[gk].map(std_map)

        df_train[f"{ac}_diff_{gk}_mean"] = df_train[ac] - df_train[f"{gk}_{ac}_mean"]
        df_test[f"{ac}_diff_{gk}_mean"] = df_test[ac] - df_test[f"{gk}_{ac}_mean"]

# Composite Multi-Column Group Aggregations
multi_group_keys = [
    ["year", "VClass"],
    ["year", "fuelType1"],
]

for g_cols in multi_group_keys:
    g_name = "_".join(g_cols)
    if all(c in df_train.columns for c in g_cols):
        for ac in agg_cols:
            mean_dict = df_train.groupby(g_cols)[ac].mean().to_dict()
            std_dict = df_train.groupby(g_cols)[ac].std().to_dict()

            tr_tuples = list(zip(*[df_train[c] for c in g_cols]))
            te_tuples = list(zip(*[df_test[c] for c in g_cols]))

            mean_tr = pd.Series(tr_tuples, index=df_train.index).map(mean_dict)
            mean_te = pd.Series(te_tuples, index=df_test.index).map(mean_dict)

            std_tr = pd.Series(tr_tuples, index=df_train.index).map(std_dict).fillna(0)
            std_te = pd.Series(te_tuples, index=df_test.index).map(std_dict).fillna(0)

            global_mean = df_train[ac].mean()
            mean_tr = mean_tr.fillna(global_mean)
            mean_te = mean_te.fillna(global_mean)

            mean_col = f"{g_name}_{ac}_mean"
            std_col = f"{g_name}_{ac}_std"

            df_train[mean_col] = mean_tr
            df_test[mean_col] = mean_te
            df_train[std_col] = std_tr
            df_test[std_col] = std_te

            df_train[f"{ac}_diff_{g_name}_mean"] = df_train[ac] - df_train[mean_col]
            df_test[f"{ac}_diff_{g_name}_mean"] = df_test[ac] - df_test[mean_col]

# 5. Missing Value Imputation, Categorical Encoding & Standard Scaling
cat_cols = df_train.select_dtypes(include=["object", "category"]).columns.tolist()

for c in cat_cols:
    df_train[c] = df_train[c].fillna("Missing").astype(str)
    df_test[c] = df_test[c].fillna("Missing").astype(str)

    freq = df_train[c].value_counts().to_dict()
    df_train[f"{c}_freq"] = df_train[c].map(freq)
    df_test[f"{c}_freq"] = df_test[c].map(freq).fillna(1)

    le = LabelEncoder()
    le.fit(pd.concat([df_train[c], df_test[c]], axis=0))
    df_train[c] = le.transform(df_train[c])
    df_test[c] = le.transform(df_test[c])

num_cols = df_train.select_dtypes(
    include=["int64", "float64", "float32", "int32"]
).columns.tolist()
if "id" in num_cols:
    num_cols.remove("id")

for c in num_cols:
    med = df_train[c].median()
    df_train[c] = df_train[c].replace([np.inf, -np.inf], np.nan).fillna(med)
    df_test[c] = df_test[c].replace([np.inf, -np.inf], np.nan).fillna(med)

scaler = StandardScaler()
X_train_proc = scaler.fit_transform(df_train[num_cols]).astype(np.float32)
X_test_proc = scaler.transform(df_test[num_cols]).astype(np.float32)

n_samples = len(X_train_proc)
n_test = len(X_test_proc)
in_features = X_train_proc.shape[1]
fold_indices = list(kf.split(X_train_proc))


# 6. PyTorch Tabular ResNet Architecture
class TabularResNet(nn.Module):

    def __init__(self, in_dim, hidden_dim=256, dropout_rate=0.15):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.act(h + self.block(h))
        out = self.head(h)
        return out


# 7. Model Ensembling Containers
oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cat = np.zeros(n_samples)
oof_et = np.zeros(n_samples)
oof_nn = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_et = np.zeros(n_test)
test_nn = np.zeros(n_test)

# 8. K-Fold Model Training Loop
for fold, (tr_idx, va_idx) in enumerate(fold_indices):
    X_tr, y_tr = X_train_proc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train_proc[va_idx], y_train[va_idx]

    # --- A. LightGBM ---
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[va_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test_proc) / n_splits

    # --- B. XGBoost ---
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.025,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test_proc) / n_splits

    # --- C. CatBoost ---
    model_cat = cb.CatBoostRegressor(
        iterations=1500,
        learning_rate=0.025,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    oof_cat[va_idx] = model_cat.predict(X_va)
    test_cat += model_cat.predict(X_test_proc) / n_splits

    # --- D. ExtraTrees Regressor ---
    model_et = ExtraTreesRegressor(
        n_estimators=350,
        max_depth=16,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_et.fit(X_tr, y_tr)
    oof_et[va_idx] = model_et.predict(X_va)
    test_et += model_et.predict(X_test_proc) / n_splits

    # --- E. PyTorch Tabular ResNet ---
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    X_te_t = torch.tensor(X_test_proc, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True
    )

    net = TabularResNet(in_dim=in_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_val_loss = float("inf")
    best_va_preds = None
    best_te_preds = None

    for epoch in range(40):
        net.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = net(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        net.eval()
        with torch.no_grad():
            val_preds_t = net(X_va_t.to(device)).cpu().numpy().flatten()
            val_loss = mean_squared_error(y_va, val_preds_t)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_va_preds = val_preds_t
                best_te_preds = net(X_te_t.to(device)).cpu().numpy().flatten()

    oof_nn[va_idx] = best_va_preds
    test_nn += best_te_preds / n_splits

# 9. 2-Tier Stacking Ensemble with Ridge Meta-Regressor
X_meta_train = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_nn])
X_meta_test = np.column_stack([test_lgb, test_xgb, test_cat, test_et, test_nn])

meta_model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
meta_model.fit(X_meta_train, y_train)

final_oof = meta_model.predict(X_meta_train)
final_test = meta_model.predict(X_meta_test)

final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))

# 10. Submission Generation
sub = pd.DataFrame({"id": test_ids, "youSaveSpend": final_test})
sub.to_csv("./submission/submission.csv", index=False)

# 11. Final Metric Reporting
print(f"Final Validation Score: {final_rmse:.5f}")
