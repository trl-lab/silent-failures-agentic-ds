import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "SALE_PRC"
drop_cols = ["PARCELNO", target_col]


def engineer_features(df):
    data = df.copy()

    # Area features and ratios
    data["living_to_land_ratio"] = data["TOT_LVG_AREA"] / (data["LND_SQFOOT"] + 1.0)
    data["spec_feat_per_sqft"] = data["SPEC_FEAT_VAL"] / (data["TOT_LVG_AREA"] + 1.0)
    data["unbuilt_land_area"] = np.maximum(0, data["LND_SQFOOT"] - data["TOT_LVG_AREA"])

    # Structural interactions
    data["quality_x_area"] = data["structure_quality"] * data["TOT_LVG_AREA"]
    data["quality_x_age"] = data["structure_quality"] * data["age"]
    data["age_per_quality"] = data["age"] / (data["structure_quality"] + 0.1)

    # Proximity & Noise features
    data["min_noise_dist"] = np.minimum(data["RAIL_DIST"], data["HWY_DIST"])
    data["min_water_dist"] = np.minimum(data["OCEAN_DIST"], data["WATER_DIST"])
    data["water_to_cntr_ratio"] = data["min_water_dist"] / (data["CNTR_DIST"] + 1.0)
    data["subcntr_to_cntr_ratio"] = data["SUBCNTR_DI"] / (data["CNTR_DIST"] + 1.0)

    # Spatial coordinates transformations (Miami coast angle ~ 25-35 degrees)
    rad_30 = np.radians(30)
    data["coord_rot_x"] = data["LATITUDE"] * np.cos(rad_30) - data[
        "LONGITUDE"
    ] * np.sin(rad_30)
    data["coord_rot_y"] = data["LATITUDE"] * np.sin(rad_30) + data[
        "LONGITUDE"
    ] * np.cos(rad_30)
    data["lat_long_ratio"] = data["LATITUDE"] / (np.abs(data["LONGITUDE"]) + 1e-5)

    # Seasonality
    data["month_sin"] = np.sin(2 * np.pi * data["month_sold"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month_sold"] / 12.0)

    return data


# Engineer features
train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)

feature_cols = [c for c in train_feat.columns if c not in drop_cols]
X = train_feat[feature_cols].copy()
y = train_df[target_col].values
X_test = test_feat[feature_cols].copy()

# Setup 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

cb_params = {
    "iterations": 2500,
    "learning_rate": 0.04,
    "depth": 7,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model = CatBoostRegressor(**cb_params)
    model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, verbose=False
    )

    val_pred = model.predict(X_va)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / kf.n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.4f}")

# Ensure positive predictions and format submission
test_preds = np.clip(test_preds, a_min=0, a_max=None)
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "SALE_PRC": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv successfully.")
