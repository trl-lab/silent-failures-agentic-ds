import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")


# ==============================================================================
# Seed Setup & Directory Creation
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)


# ==============================================================================
# Metric Definition
# ==============================================================================
def calculate_rmsle(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred_clipped)) ** 2))


# ==============================================================================
# 1. Data Loading & Feature Engineering
# ==============================================================================
print("Loading raw datasets...")
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_df["datetime"] = pd.to_datetime(train_df["datetime"])
test_df["datetime"] = pd.to_datetime(test_df["datetime"])

# Split train_df into train (days 1-14) and validation (days 15-19)
train_mask = train_df["datetime"].dt.day <= 14
val_mask = ~train_mask

tr_data = train_df[train_mask].copy().reset_index(drop=True)
va_data = train_df[val_mask].copy().reset_index(drop=True)
te_data = test_df.copy().reset_index(drop=True)


def extract_base_features(df):
    dt = df["datetime"].dt

    # Temporal core features
    df["year"] = dt.year
    df["month"] = dt.month
    df["day"] = dt.day
    df["hour"] = dt.hour
    df["dayofweek"] = dt.dayofweek
    df["quarter"] = dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Time trend (months since start of dataset)
    df["time_index"] = (df["year"] - 2011) * 12 + df["month"]

    # Cyclical representations
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # Commuter vs Leisure peak hours
    df["peak_commute"] = (
        (df["workingday"] == 1)
        & (
            ((df["hour"] >= 7) & (df["hour"] <= 9))
            | ((df["hour"] >= 17) & (df["hour"] <= 19))
        )
    ).astype(int)

    df["peak_leisure"] = (
        ((df["workingday"] == 0) | (df["holiday"] == 1))
        & (df["hour"] >= 11)
        & (df["hour"] <= 16)
    ).astype(int)

    # Weather & Comfort interactions
    df["temp_atemp_diff"] = df["atemp"] - df["temp"]
    df["humidity_temp_ratio"] = df["humidity"] / (df["temp"] + 1.0)
    df["wind_temp_ratio"] = df["windspeed"] / (df["temp"] + 1.0)
    df["windspeed_is_zero"] = (df["windspeed"] == 0).astype(int)

    # Heat Index approximation
    df["heat_index"] = (
        df["temp"] + 0.33 * df["humidity"] - 0.70 * df["windspeed"] - 4.00
    )

    return df


print("Engineering temporal and physical weather features...")
tr_data = extract_base_features(tr_data)
va_data = extract_base_features(va_data)
te_data = extract_base_features(te_data)

# Non-leaky Group Aggregations calculated strictly on Training partition
print("Computing non-leaky group statistics from training set...")
stats_cols = ["temp", "humidity", "windspeed"]

grouped_stats = (
    tr_data.groupby(["season", "hour"])[stats_cols].agg(["mean", "std"]).reset_index()
)
grouped_stats.columns = ["season", "hour"] + [
    f"{col}_season_hour_{stat}" for col in stats_cols for stat in ["mean", "std"]
]

# Merge statistics onto all partitions
tr_data = tr_data.merge(grouped_stats, on=["season", "hour"], how="left")
va_data = va_data.merge(grouped_stats, on=["season", "hour"], how="left")
te_data = te_data.merge(grouped_stats, on=["season", "hour"], how="left")

# Target logarithmic transformation
target_cols = ["count", "casual", "registered"]
for col in target_cols:
    if col in tr_data.columns:
        tr_data[f"log_{col}"] = np.log1p(tr_data[col])
    if col in va_data.columns:
        va_data[f"log_{col}"] = np.log1p(va_data[col])

non_feature_cols = [
    "datetime",
    "count",
    "casual",
    "registered",
    "log_count",
    "log_casual",
    "log_registered",
]
feature_cols = [c for c in tr_data.columns if c not in non_feature_cols]

# Impute potential NaNs from aggregation with training medians
for col in feature_cols:
    if (
        tr_data[col].isnull().any()
        or va_data[col].isnull().any()
        or te_data[col].isnull().any()
    ):
        fill_val = tr_data[col].median()
        tr_data[col] = tr_data[col].fillna(fill_val)
        va_data[col] = va_data[col].fillna(fill_val)
        te_data[col] = te_data[col].fillna(fill_val)

X_tr = tr_data[feature_cols].copy()
X_va = va_data[feature_cols].copy()
X_te = te_data[feature_cols].copy()

y_tr_casual = tr_data["log_casual"].values
y_va_casual = va_data["log_casual"].values
y_tr_reg = tr_data["log_registered"].values
y_va_reg = va_data["log_registered"].values
y_tr_count = tr_data["log_count"].values
y_va_count = va_data["log_count"].values

val_true_count = va_data["count"].values


# ==============================================================================
# 2. LightGBM Training Pipeline
# ==============================================================================
print("Training LightGBM models...")


def train_lgbm(y_tr, y_va):
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1500,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(50, verbose=False)]
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=callbacks,
    )
    va_pred = model.predict(X_va)
    te_pred = model.predict(X_te)
    return va_pred, te_pred


lgb_va_cas, lgb_te_cas = train_lgbm(y_tr_casual, y_va_casual)
lgb_va_reg, lgb_te_reg = train_lgbm(y_tr_reg, y_va_reg)
lgb_va_cnt, lgb_te_cnt = train_lgbm(y_tr_count, y_va_count)

lgb_va_dual = np.expm1(np.clip(lgb_va_cas, 0, None)) + np.expm1(
    np.clip(lgb_va_reg, 0, None)
)
lgb_te_dual = np.expm1(np.clip(lgb_te_cas, 0, None)) + np.expm1(
    np.clip(lgb_te_reg, 0, None)
)
lgb_va_dir = np.expm1(np.clip(lgb_va_cnt, 0, None))
lgb_te_dir = np.expm1(np.clip(lgb_te_cnt, 0, None))

lgb_va_final = 0.5 * lgb_va_dual + 0.5 * lgb_va_dir
lgb_te_final = 0.5 * lgb_te_dual + 0.5 * lgb_te_dir
print(f"LightGBM Validation RMSLE: {calculate_rmsle(val_true_count, lgb_va_final):.4f}")


# ==============================================================================
# 3. XGBoost Training Pipeline
# ==============================================================================
print("Training XGBoost models...")


def train_xgb(y_tr, y_va):
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=6,
        learning_rate=0.03,
        n_estimators=1500,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    va_pred = model.predict(X_va)
    te_pred = model.predict(X_te)
    return va_pred, te_pred


xgb_va_cas, xgb_te_cas = train_xgb(y_tr_casual, y_va_casual)
xgb_va_reg, xgb_te_reg = train_xgb(y_tr_reg, y_va_reg)
xgb_va_cnt, xgb_te_cnt = train_xgb(y_tr_count, y_va_count)

xgb_va_dual = np.expm1(np.clip(xgb_va_cas, 0, None)) + np.expm1(
    np.clip(xgb_va_reg, 0, None)
)
xgb_te_dual = np.expm1(np.clip(xgb_te_cas, 0, None)) + np.expm1(
    np.clip(xgb_te_reg, 0, None)
)
xgb_va_dir = np.expm1(np.clip(xgb_va_cnt, 0, None))
xgb_te_dir = np.expm1(np.clip(xgb_te_cnt, 0, None))

xgb_va_final = 0.5 * xgb_va_dual + 0.5 * xgb_va_dir
xgb_te_final = 0.5 * xgb_te_dual + 0.5 * xgb_te_dir
print(f"XGBoost Validation RMSLE: {calculate_rmsle(val_true_count, xgb_va_final):.4f}")


# ==============================================================================
# 4. CatBoost Training Pipeline (if available)
# ==============================================================================
if HAS_CATBOOST:
    print("Training CatBoost models...")

    def train_cat(y_tr, y_va):
        model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        va_pred = model.predict(X_va)
        te_pred = model.predict(X_te)
        return va_pred, te_pred

    cat_va_cas, cat_te_cas = train_cat(y_tr_casual, y_va_casual)
    cat_va_reg, cat_te_reg = train_cat(y_tr_reg, y_va_reg)
    cat_va_cnt, cat_te_cnt = train_cat(y_tr_count, y_va_count)

    cat_va_dual = np.expm1(np.clip(cat_va_cas, 0, None)) + np.expm1(
        np.clip(cat_va_reg, 0, None)
    )
    cat_te_dual = np.expm1(np.clip(cat_te_cas, 0, None)) + np.expm1(
        np.clip(cat_te_reg, 0, None)
    )
    cat_va_dir = np.expm1(np.clip(cat_va_cnt, 0, None))
    cat_te_dir = np.expm1(np.clip(cat_te_cnt, 0, None))

    cat_va_final = 0.5 * cat_va_dual + 0.5 * cat_va_dir
    cat_te_final = 0.5 * cat_te_dual + 0.5 * cat_te_dir
    print(
        f"CatBoost Validation RMSLE: {calculate_rmsle(val_true_count, cat_va_final):.4f}"
    )


# ==============================================================================
# 5. PyTorch Deep Tabular Network Training Pipeline
# ==============================================================================
print("Training PyTorch Deep Tabular MLP...")
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_va_scaled = scaler.transform(X_va)
X_te_scaled = scaler.transform(X_te)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Y_tr_dual = torch.tensor(np.column_stack([y_tr_casual, y_tr_reg]), dtype=torch.float32)
Y_va_dual = torch.tensor(np.column_stack([y_va_casual, y_va_reg]), dtype=torch.float32)

tr_dataset = TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32), Y_tr_dual)
va_dataset = TensorDataset(torch.tensor(X_va_scaled, dtype=torch.float32), Y_va_dual)

tr_loader = DataLoader(tr_dataset, batch_size=128, shuffle=True)
va_loader = DataLoader(va_dataset, batch_size=256, shuffle=False)


class TabularMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


nn_model = TabularMLP(input_dim=X_tr_scaled.shape[1], hidden_dim=128).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

best_loss = float("inf")
best_state = None

for epoch in range(40):
    nn_model.train()
    for bx, by in tr_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = nn_model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()

    scheduler.step()

    nn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx, by in va_loader:
            bx, by = bx.to(device), by.to(device)
            out = nn_model(bx)
            val_loss += criterion(out, by).item() * len(bx)
    val_loss /= len(va_dataset)

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.cpu() for k, v in nn_model.state_dict().items()}

    if (epoch + 1) % 10 == 0 or epoch == 39:
        print(f"Epoch {epoch+1:02d}/40 | Val MSE Loss: {val_loss:.4f}")

nn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
nn_model.eval()

with torch.no_grad():
    nn_va_preds = (
        nn_model(torch.tensor(X_va_scaled, dtype=torch.float32).to(device))
        .cpu()
        .numpy()
    )
    nn_te_preds = (
        nn_model(torch.tensor(X_te_scaled, dtype=torch.float32).to(device))
        .cpu()
        .numpy()
    )

nn_va_dual = np.expm1(np.clip(nn_va_preds[:, 0], 0, None)) + np.expm1(
    np.clip(nn_va_preds[:, 1], 0, None)
)
nn_te_dual = np.expm1(np.clip(nn_te_preds[:, 0], 0, None)) + np.expm1(
    np.clip(nn_te_preds[:, 1], 0, None)
)
print(f"PyTorch Neural RMSLE: {calculate_rmsle(val_true_count, nn_va_dual):.4f}")


# ==============================================================================
# 6. Model Ensembling & Validation Scoring
# ==============================================================================
print("Ensembling predictions across GBDTs and Neural Architecture...")

if HAS_CATBOOST:
    final_val_preds = (
        0.35 * lgb_va_final
        + 0.35 * xgb_va_final
        + 0.20 * cat_va_final
        + 0.10 * nn_va_dual
    )
    final_test_preds = (
        0.35 * lgb_te_final
        + 0.35 * xgb_te_final
        + 0.20 * cat_te_final
        + 0.10 * nn_te_dual
    )
else:
    final_val_preds = 0.425 * lgb_va_final + 0.425 * xgb_va_final + 0.15 * nn_va_dual
    final_test_preds = 0.425 * lgb_te_final + 0.425 * xgb_te_final + 0.15 * nn_te_dual

final_val_score = calculate_rmsle(val_true_count, final_val_preds)


# ==============================================================================
# 7. Submission File Generation
# ==============================================================================
final_test_preds = np.clip(final_test_preds, 0, None)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "count": final_test_preds}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved test predictions to {submission_path} with shape {submission_df.shape}")

# Final validation score print out
print(f"Final Validation Score: {final_val_score:.5f}")
