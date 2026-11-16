import copy
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# 1. Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Load Raw Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Ensure unique index/id preservation
if "id" not in train_df.columns:
    train_df["id"] = train_df.index
if "id" not in test_df.columns:
    test_df["id"] = test_df.index

target_col = "price"

# Clean known domain typo outlier in train (33 bedrooms in 1620 sqft house -> 3 bedrooms)
if "bedrooms" in train_df.columns and "sqft_living" in train_df.columns:
    train_df.loc[
        (train_df["bedrooms"] == 33) & (train_df["sqft_living"] < 2000), "bedrooms"
    ] = 3

# Combine datasets for consistent feature engineering
train_df["is_train"] = 1
test_df["is_train"] = 0
if target_col not in test_df.columns:
    test_df[target_col] = np.nan

combined = pd.concat([train_df, test_df], ignore_index=True)

# Convert object/categorical feature columns to numeric
for col in combined.columns:
    if col not in ["id", "date", "price", "is_train"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

# 3. Temporal Features
if "date" in combined.columns:
    dates = pd.to_datetime(
        combined["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    combined["sale_year"] = dates.dt.year.fillna(2014).astype(int)
    combined["sale_month"] = dates.dt.month.fillna(6).astype(int)
    combined["sale_day"] = dates.dt.day.fillna(15).astype(int)
    combined["sale_dayofweek"] = dates.dt.dayofweek.fillna(2).astype(int)
    combined["sale_quarter"] = dates.dt.quarter.fillna(2).astype(int)
    combined["sale_year_month"] = (
        combined["sale_year"] + (combined["sale_month"] - 1) / 12.0
    )
else:
    combined["sale_year"] = 2014
    combined["sale_month"] = 6

# Age & Renovation
if "yr_built" in combined.columns:
    combined["house_age"] = (combined["sale_year"] - combined["yr_built"]).clip(lower=0)

if "yr_renovated" in combined.columns and "yr_built" in combined.columns:
    combined["is_renovated"] = (combined["yr_renovated"] > 0).astype(int)
    effective_year = np.where(
        combined["yr_renovated"] > 0, combined["yr_renovated"], combined["yr_built"]
    )
    combined["effective_age"] = (combined["sale_year"] - effective_year).clip(lower=0)

# 4. Structural & Area Ratios
if "sqft_living" in combined.columns and "sqft_lot" in combined.columns:
    combined["sqft_total"] = combined["sqft_living"] + combined["sqft_lot"]
    combined["living_lot_ratio"] = combined["sqft_living"] / (
        combined["sqft_lot"] + 1.0
    )

if "bedrooms" in combined.columns and "bathrooms" in combined.columns:
    combined["total_rooms"] = combined["bedrooms"] + combined["bathrooms"]
    combined["bed_bath_ratio"] = combined["bedrooms"] / (combined["bathrooms"] + 0.1)

if "sqft_living" in combined.columns:
    if "total_rooms" in combined.columns:
        combined["sqft_per_room"] = combined["sqft_living"] / (
            combined["total_rooms"] + 0.1
        )
    if "bedrooms" in combined.columns:
        combined["sqft_per_bedroom"] = combined["sqft_living"] / (
            combined["bedrooms"] + 0.1
        )
    if "floors" in combined.columns:
        combined["sqft_per_floor"] = combined["sqft_living"] / (
            combined["floors"] + 0.1
        )

if "sqft_living" in combined.columns and "sqft_above" in combined.columns:
    combined["sqft_above_ratio"] = combined["sqft_above"] / (
        combined["sqft_living"] + 1.0
    )

if "sqft_basement" in combined.columns:
    combined["has_basement"] = (combined["sqft_basement"] > 0).astype(int)
    if "sqft_living" in combined.columns:
        combined["sqft_basement_ratio"] = combined["sqft_basement"] / (
            combined["sqft_living"] + 1.0
        )

if "sqft_living15" in combined.columns and "sqft_living" in combined.columns:
    combined["living_diff_15"] = combined["sqft_living"] - combined["sqft_living15"]
    combined["living_ratio_15"] = combined["sqft_living"] / (
        combined["sqft_living15"] + 1.0
    )

if "sqft_lot15" in combined.columns and "sqft_lot" in combined.columns:
    combined["lot_diff_15"] = combined["sqft_lot"] - combined["sqft_lot15"]
    combined["lot_ratio_15"] = combined["sqft_lot"] / (combined["sqft_lot15"] + 1.0)

# 5. Quality Interactions
if "grade" in combined.columns and "condition" in combined.columns:
    combined["quality_score"] = combined["grade"] * combined["condition"]

if "waterfront" in combined.columns and "view" in combined.columns:
    combined["view_waterfront"] = combined["view"] * (combined["waterfront"] + 1)

if (
    "grade" in combined.columns
    and "waterfront" in combined.columns
    and "view" in combined.columns
):
    combined["luxury_score"] = (
        combined["grade"] * (combined["waterfront"] + 1) * (combined["view"] + 1)
    )

# 6. Spatial Features & Center Distances
if "lat" in combined.columns and "long" in combined.columns:

    def haversine_np(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        return 6371.0 * 2 * np.arcsin(np.sqrt(a))

    SEATTLE_LAT, SEATTLE_LON = 47.6062, -122.3321
    BELLEVUE_LAT, BELLEVUE_LON = 47.6101, -122.2015
    REDMOND_LAT, REDMOND_LON = 47.6740, -122.1215
    SEATAC_LAT, SEATAC_LON = 47.4502, -122.3088

    combined["dist_seattle"] = haversine_np(
        combined["lat"], combined["long"], SEATTLE_LAT, SEATTLE_LON
    )
    combined["dist_bellevue"] = haversine_np(
        combined["lat"], combined["long"], BELLEVUE_LAT, BELLEVUE_LON
    )
    combined["dist_redmond"] = haversine_np(
        combined["lat"], combined["long"], REDMOND_LAT, REDMOND_LON
    )
    combined["dist_seatac"] = haversine_np(
        combined["lat"], combined["long"], SEATAC_LAT, SEATAC_LON
    )

    # Spatial Coordinate Rotations
    for angle in [30, 45, 60]:
        rad = np.radians(angle)
        combined[f"rot_lat_{angle}"] = combined["lat"] * np.cos(rad) - combined[
            "long"
        ] * np.sin(rad)
        combined[f"rot_long_{angle}"] = combined["lat"] * np.sin(rad) + combined[
            "long"
        ] * np.cos(rad)

    # K-Means Spatial Clusters fit solely on training set
    train_mask = combined["is_train"] == 1
    coords_train = combined.loc[train_mask, ["lat", "long"]].values
    coords_all = combined[["lat", "long"]].values

    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    kmeans.fit(coords_train)
    combined["geo_cluster"] = kmeans.predict(coords_all)

# Separate train and test
train_processed = combined[combined["is_train"] == 1].copy()
test_processed = combined[combined["is_train"] == 0].copy()

# 7. Zipcode Group Statistics (Strict Train-Only Fitting)
if "zipcode" in train_processed.columns:
    zip_stats = (
        train_processed.groupby("zipcode")
        .agg(
            zip_sqft_mean=("sqft_living", "mean"),
            zip_sqft_median=("sqft_living", "median"),
            zip_grade_mean=("grade", "mean"),
        )
        .reset_index()
    )

    train_processed = train_processed.merge(zip_stats, on="zipcode", how="left")
    test_processed = test_processed.merge(zip_stats, on="zipcode", how="left")

train_processed.drop(columns=["is_train"], inplace=True, errors="ignore")
test_processed.drop(columns=["is_train", target_col], inplace=True, errors="ignore")

# Log target variable transformation for log-space optimization
train_processed["log_price"] = np.log1p(train_processed["price"])

# 8. 5-Fold Stratified Split Index Assignment
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_processed["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(train_processed)):
    train_processed.loc[val_idx, "fold"] = fold

ignore_cols = ["id", "date", "price", "log_price", "fold"]
feature_cols = [c for c in train_processed.columns if c not in ignore_cols]

n_train = len(train_processed)
n_test = len(test_processed)
y_true_all = train_processed["price"].values


# 9. PyTorch TabularResNet Architecture Definitions
class TabularResBlock(nn.Module):

    def __init__(
        self, in_features: int, hidden_features: int, dropout_rate: float = 0.15
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_features, in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TabularResBlock(hidden_dim, hidden_dim * 2, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


# Hyperparameter Getter Functions
def get_lgb_params(seed: int = 42) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(seed: int = 42) -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
        "enable_categorical": True,
    }


def get_catboost_params(seed: int = 42) -> dict:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3000,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "verbose": 0,
    }


# 10. Setup OOF and Test Prediction Containers
oof_lgb = np.zeros(n_train)
oof_xgb = np.zeros(n_train)
oof_cat = np.zeros(n_train)
oof_nn = np.zeros(n_train)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cat = np.zeros(n_test)
test_nn = np.zeros(n_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 11. 5-Fold Training Loop
n_folds = 5
for fold in range(n_folds):
    train_mask = train_processed["fold"] != fold
    val_mask = train_processed["fold"] == fold

    X_train = train_processed.loc[train_mask, feature_cols].copy()
    y_train_log = train_processed.loc[train_mask, "log_price"].values

    X_val = train_processed.loc[val_mask, feature_cols].copy()
    y_val_log = train_processed.loc[val_mask, "log_price"].values
    y_val_true = train_processed.loc[val_mask, "price"].values

    X_test = test_processed[feature_cols].copy()

    # --- LightGBM ---
    model_lgb = lgb.LGBMRegressor(**get_lgb_params(seed=42 + fold))
    model_lgb.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    val_pred_lgb = np.expm1(model_lgb.predict(X_val))
    oof_lgb[val_mask] = val_pred_lgb
    test_lgb += np.expm1(model_lgb.predict(X_test)) / n_folds

    # --- XGBoost ---
    model_xgb = xgb.XGBRegressor(
        **get_xgb_params(seed=42 + fold), early_stopping_rounds=50
    )
    model_xgb.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
    val_pred_xgb = np.expm1(model_xgb.predict(X_val))
    oof_xgb[val_mask] = val_pred_xgb
    test_xgb += np.expm1(model_xgb.predict(X_test)) / n_folds

    # --- CatBoost ---
    model_cat = CatBoostRegressor(
        **get_catboost_params(seed=42 + fold), early_stopping_rounds=50
    )
    model_cat.fit(
        X_train,
        y_train_log,
        eval_set=(X_val, y_val_log),
        verbose=False,
        use_best_model=True,
    )
    val_pred_cat = np.expm1(model_cat.predict(X_val))
    oof_cat[val_mask] = val_pred_cat
    test_cat += np.expm1(model_cat.predict(X_test)) / n_folds

    # --- PyTorch TabularResNet ---
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_tr_proc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_va_proc = scaler.transform(imputer.transform(X_val))
    X_te_proc = scaler.transform(imputer.transform(X_test))

    train_dataset = TensorDataset(
        torch.tensor(X_tr_proc, dtype=torch.float32),
        torch.tensor(y_train_log, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    torch.manual_seed(42 + fold)
    nn_model = TabularResNet(
        input_dim=len(feature_cols),
        hidden_dim=256,
        num_blocks=3,
        dropout_rate=0.15,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

    best_loss = float("inf")
    best_weights = None
    patience = 12
    patience_counter = 0

    val_tensor_x = torch.tensor(X_va_proc, dtype=torch.float32).to(device)
    val_tensor_y = torch.tensor(y_val_log, dtype=torch.float32).to(device)

    for epoch in range(60):
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
        with torch.no_grad():
            v_out = nn_model(val_tensor_x)
            v_loss = criterion(v_out, val_tensor_y).item()

        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = copy.deepcopy(nn_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    nn_model.load_state_dict(best_weights)
    nn_model.eval()
    with torch.no_grad():
        val_pred_nn_log = nn_model(val_tensor_x).cpu().numpy()
        test_tensor_x = torch.tensor(X_te_proc, dtype=torch.float32).to(device)
        test_pred_nn_log = nn_model(test_tensor_x).cpu().numpy()

    val_pred_nn = np.expm1(val_pred_nn_log)
    oof_nn[val_mask] = val_pred_nn
    test_nn += np.expm1(test_pred_nn_log) / n_folds


# 12. SciPy Optimize Bounded Ensemble Blending Weights
def loss_func(weights):
    w = np.maximum(np.array(weights), 0)
    w_sum = np.sum(w)
    if w_sum == 0:
        return 1e10
    w = w / w_sum
    blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat + w[3] * oof_nn
    return np.sqrt(mean_squared_error(y_true_all, blend))


res = minimize(
    loss_func,
    x0=[0.3, 0.3, 0.3, 0.1],
    bounds=[(0, 1)] * 4,
    method="Nelder-Mead",
)
best_w = np.maximum(res.x, 0)
best_w = best_w / np.sum(best_w)

oof_ensemble = (
    best_w[0] * oof_lgb + best_w[1] * oof_xgb + best_w[2] * oof_cat + best_w[3] * oof_nn
)
final_score = np.sqrt(mean_squared_error(y_true_all, oof_ensemble))

# 13. Generate Submission File
test_ensemble = np.clip(
    best_w[0] * test_lgb
    + best_w[1] * test_xgb
    + best_w[2] * test_cat
    + best_w[3] * test_nn,
    a_min=0,
    a_max=None,
)

submission_df = pd.DataFrame(
    {"id": test_processed["id"].astype(int), "price": test_ensemble}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_score}")
