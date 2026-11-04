import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    has_catboost = True
except ImportError:
    has_catboost = False

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Directory Setup
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_ids = train_df["id"].values
test_ids = test_df["id"].values
target = train_df["price_doc"].values
train_df_no_target = train_df.drop(columns=["price_doc"])

df = pd.concat([train_df_no_target, test_df], axis=0, ignore_index=True)

# 2. Temporal Features
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["dayofyear"] = df["timestamp"].dt.dayofyear
df["year_month"] = df["year"] * 12 + df["month"]

min_time = df["timestamp"].min()
df["time_elapsed_days"] = (df["timestamp"] - min_time).dt.days

# Binary Missingness Indicators for Structural Attributes
missing_cols = ["life_sq", "build_year", "max_floor", "kitch_sq", "state", "material"]
for col in missing_cols:
    if col in df.columns:
        df[f"{col}_isna"] = df[col].isna().astype(float)

# Frequency/Count Encodings for Categorical Variables
freq_cols = ["sub_area", "product_type", "ecosystem", "ecology"]
for col in freq_cols:
    if col in df.columns:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_freq"] = df[col].map(freq_map).astype(float)

# 3. Domain Cleaning & Anomaly Correction
df["build_year"] = df["build_year"].apply(
    lambda x: np.nan if (pd.isna(x) or x < 1600 or x > 2020) else x
)
df["building_age"] = df["year"] - df["build_year"]

if "state" in df.columns:
    df["state"] = df["state"].replace({33: 3})

df.loc[df["full_sq"] <= 0, "full_sq"] = np.nan
df.loc[df["life_sq"] > df["full_sq"], "life_sq"] = np.nan
df.loc[df["kitch_sq"] >= df["full_sq"], "kitch_sq"] = np.nan

df.loc[df["max_floor"] < df["floor"], "max_floor"] = np.nan
df.loc[df["max_floor"] == 0, "max_floor"] = np.nan

# 4. Domain Feature Engineering
df["life_sq_ratio"] = df["life_sq"] / (df["full_sq"] + 1e-5)
df["kitch_sq_ratio"] = df["kitch_sq"] / (df["full_sq"] + 1e-5)
df["extra_sq"] = df["full_sq"] - df["life_sq"].fillna(0)
df["sq_per_room"] = df["full_sq"] / (df["num_room"] + 1e-5)
df["life_sq_per_room"] = df["life_sq"] / (df["num_room"] + 1e-5)

# Log1p transforms for all area features
area_cols = ["full_sq", "life_sq", "kitch_sq", "extra_sq", "sq_per_room"]
for col in area_cols:
    df[f"{col}_log1p"] = np.log1p(df[col].clip(lower=0))

df["sq_per_floor"] = df["full_sq"] / (df["max_floor"] + 1e-5)
df["floor_fraction"] = df["floor"] / (df["max_floor"] + 1e-5)
df["room_density"] = df["num_room"] / (df["full_sq"] + 1e-5)

df["floor_ratio"] = df["floor"] / (df["max_floor"] + 1e-5)
df["floor_to_top"] = df["max_floor"] - df["floor"]
df["is_top_floor"] = (df["floor"] == df["max_floor"]).astype(float)
df["is_first_floor"] = (df["floor"] == 1).astype(float)

# Spatial (sub_area) and Temporal (year_month) Group Aggregations & Relative Ratios
if "sub_area" in df.columns:
    for stat in ["mean", "median", "std"]:
        df[f"sub_area_{stat}_full_sq"] = df.groupby("sub_area")["full_sq"].transform(stat)
        df[f"sub_area_full_sq_{stat}"] = df[f"sub_area_{stat}_full_sq"]

    df["full_sq_ratio_sub_area_median"] = df["full_sq"] / (df["sub_area_median_full_sq"] + 1e-5)
    df["full_sq_ratio_sub_area_mean"] = df["full_sq"] / (df["sub_area_mean_full_sq"] + 1e-5)
    df["sub_area_full_sq_diff"] = df["full_sq"] - df["sub_area_mean_full_sq"]

if "year_month" in df.columns:
    for stat in ["mean", "median", "std"]:
        df[f"year_month_{stat}_full_sq"] = df.groupby("year_month")["full_sq"].transform(stat)
        df[f"year_month_full_sq_{stat}"] = df[f"year_month_{stat}_full_sq"]

    df["full_sq_ratio_year_month_median"] = df["full_sq"] / (df["year_month_median_full_sq"] + 1e-5)
    df["full_sq_ratio_year_month_mean"] = df["full_sq"] / (df["year_month_mean_full_sq"] + 1e-5)

if "green_zone_km" in df.columns and "industrial_km" in df.columns:
    df["green_vs_industrial"] = df["green_zone_km"] / (df["industrial_km"] + 1e-5)

# 5. Encoding Categorical Variables
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df = df.drop(columns=["timestamp"])

# Split back into train and test sets
train_processed = df.iloc[: len(train_ids)].copy()
test_processed = df.iloc[len(train_ids) :].copy()

train_processed["price_doc"] = target
train_processed["target_log"] = np.log1p(target)

# Chronological Train / Validation Split (Hold out last 15% of transactions)
train_processed = train_processed.sort_values(
    by=["year_month", "time_elapsed_days"]
).reset_index(drop=True)

val_size = int(len(train_processed) * 0.15)
train_split = train_processed.iloc[:-val_size].reset_index(drop=True)
val_split = train_processed.iloc[-val_size:].reset_index(drop=True)

exclude_cols = ["id", "price_doc", "target_log"]
features = [c for c in train_split.columns if c not in exclude_cols]

X_train = train_split[features].astype(np.float32)
y_train = train_split["target_log"].values.astype(np.float32)

X_val = val_split[features].astype(np.float32)
y_val = val_split["target_log"].values.astype(np.float32)

X_test = test_processed[features].astype(np.float32)
test_ids = test_processed["id"].values.astype(int)

# 6. Model 1: LightGBM Regressor
print("Training LightGBM model...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=127,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

val_preds_lgb = lgb_model.predict(X_val)
test_preds_lgb = lgb_model.predict(X_test)
lgb_rmsle = float(np.sqrt(np.mean((val_preds_lgb - y_val) ** 2)))
print(f"LGBM Val RMSLE: {lgb_rmsle:.5f}")

# 7. Model 2: XGBoost Regressor
print("Training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_preds_xgb = xgb_model.predict(X_val)
test_preds_xgb = xgb_model.predict(X_test)
xgb_rmsle = float(np.sqrt(np.mean((val_preds_xgb - y_val) ** 2)))
print(f"XGB Val RMSLE: {xgb_rmsle:.5f}")

# 8. Model 3: CatBoost Regressor
if has_catboost:
    print("Training CatBoost model...")
    cb_model = cb.CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    val_preds_cb = cb_model.predict(X_val)
    test_preds_cb = cb_model.predict(X_test)
    cb_rmsle = float(np.sqrt(np.mean((val_preds_cb - y_val) ** 2)))
    print(f"CatBoost Val RMSLE: {cb_rmsle:.5f}")


# 9. Model 4: PyTorch Tabular ResNet
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=2, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_np = scaler.fit_transform(imputer.fit_transform(X_train))
X_val_np = scaler.transform(imputer.transform(X_val))
X_test_np = scaler.transform(imputer.transform(X_test))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = TensorDataset(
    torch.tensor(X_train_np, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_np, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=2)

nn_model = TabularResNet(
    input_dim=X_train_np.shape[1], hidden_dim=256, num_blocks=2, dropout=0.2
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

best_val_loss = float("inf")
best_model_path = "./working/best_nn_model.pt"
epochs = 15

print("Training PyTorch Tabular ResNet...")
for epoch in range(epochs):
    nn_model.train()
    total_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = nn_model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    scheduler.step()

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = nn_model(X_b)
            val_loss += criterion(out, y_b).item() * len(y_b)

    train_rmse = np.sqrt(total_loss / len(train_dataset))
    val_rmse = np.sqrt(val_loss / len(val_dataset))

    if val_rmse < best_val_loss:
        best_val_loss = val_rmse
        torch.save(nn_model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch+1:02d}/{epochs:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}"
    )

nn_model.load_state_dict(torch.load(best_model_path, map_location=device))
nn_model.eval()

with torch.no_grad():
    val_preds_nn = (
        nn_model(torch.tensor(X_val_np, dtype=torch.float32).to(device)).cpu().numpy()
    )
    test_preds_nn = (
        nn_model(torch.tensor(X_test_np, dtype=torch.float32).to(device)).cpu().numpy()
    )

nn_rmsle = float(np.sqrt(np.mean((val_preds_nn - y_val) ** 2)))
print(f"PyTorch NN Val RMSLE: {nn_rmsle:.5f}")

# 10. Ensemble Blending with SLSQP Optimization
if has_catboost:
    val_preds_matrix = np.column_stack([val_preds_lgb, val_preds_xgb, val_preds_cb, val_preds_nn])
    test_preds_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cb, test_preds_nn])
else:
    val_preds_matrix = np.column_stack([val_preds_lgb, val_preds_xgb, val_preds_nn])
    test_preds_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_nn])

def objective(weights):
    blend = np.dot(val_preds_matrix, weights)
    return np.sqrt(np.mean((blend - y_val) ** 2))

num_models = val_preds_matrix.shape[1]
init_weights = np.ones(num_models) / num_models
bounds = [(0.0, 1.0) for _ in range(num_models)]
constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

res = minimize(
    objective,
    init_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

opt_weights = res.x
print(f"Optimized Ensemble Weights: {opt_weights}")

val_preds_log = np.dot(val_preds_matrix, opt_weights)
test_preds_log = np.dot(test_preds_matrix, opt_weights)

final_val_rmsle = float(np.sqrt(np.mean((val_preds_log - y_val) ** 2)))

# 11. Post-Processing & Submission
test_preds_price = np.expm1(test_preds_log)
test_preds_price = np.clip(test_preds_price, 100000.0, None)

submission_df = pd.DataFrame({"id": test_ids, "price_doc": test_preds_price})
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_val_rmsle:.5f}")
