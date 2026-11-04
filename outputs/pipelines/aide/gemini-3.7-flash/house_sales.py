import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Downtown Seattle coordinates
SEATTLE_LAT = 47.6062
SEATTLE_LON = -122.3321


def engineer_features(df):
    data = df.copy()

    # Parse date if present
    if "date" in data.columns:
        date_col = pd.to_datetime(data["date"], errors="coerce")
        data["sale_year"] = date_col.dt.year.fillna(2014).astype(int)
        data["sale_month"] = date_col.dt.month.fillna(5).astype(int)
        data["sale_quarter"] = date_col.dt.quarter.fillna(2).astype(int)
        data["sale_dayofweek"] = date_col.dt.dayofweek.fillna(0).astype(int)
    else:
        data["sale_year"] = 2014
        data["sale_month"] = 5
        data["sale_quarter"] = 2
        data["sale_dayofweek"] = 0

    # Age and renovation features
    data["house_age"] = data["sale_year"] - data["yr_built"]
    data["house_age"] = data["house_age"].clip(lower=0)
    data["is_renovated"] = (data["yr_renovated"] > 0).astype(int)
    data["years_since_renovated"] = np.where(
        data["yr_renovated"] > 0,
        data["sale_year"] - data["yr_renovated"],
        data["house_age"],
    ).clip(min=0)

    # Distance to Seattle center (Euclidean & Manhattan approximations)
    data["dist_lat"] = data["lat"] - SEATTLE_LAT
    data["dist_long"] = data["long"] - SEATTLE_LON
    data["dist_to_seattle"] = np.sqrt(data["dist_lat"] ** 2 + data["dist_long"] ** 2)
    data["manhattan_to_seattle"] = np.abs(data["dist_lat"]) + np.abs(data["dist_long"])

    # Area and ratio features
    data["total_sqft"] = data["sqft_living"] + data["sqft_lot"]
    data["living_to_lot_ratio"] = data["sqft_living"] / (data["sqft_lot"] + 1.0)
    data["living_diff_living15"] = data["sqft_living"] - data["sqft_living15"]
    data["living_ratio_living15"] = data["sqft_living"] / (data["sqft_living15"] + 1.0)
    data["lot_ratio_lot15"] = data["sqft_lot"] / (data["sqft_lot15"] + 1.0)
    data["basement_ratio"] = data["sqft_basement"] / (data["sqft_living"] + 1.0)
    data["above_ratio"] = data["sqft_above"] / (data["sqft_living"] + 1.0)

    # Room metrics
    data["total_rooms"] = data["bedrooms"] + data["bathrooms"]
    data["sqft_per_room"] = data["sqft_living"] / (data["total_rooms"] + 1.0)
    data["bath_per_bed"] = data["bathrooms"] / (data["bedrooms"] + 1.0)
    data["bed_per_floor"] = data["bedrooms"] / (data["floors"] + 1.0)

    # Quality score interactions
    data["grade_x_condition"] = data["grade"] * data["condition"]
    data["grade_x_living"] = data["grade"] * data["sqft_living"]

    # Categorical zip code
    data["zipcode"] = data["zipcode"].astype("category")

    # Drop non-feature columns
    drop_cols = ["id", "date", "price"]
    features_to_drop = [c for c in drop_cols if c in data.columns]
    data = data.drop(columns=features_to_drop)

    return data


# Extract features
X = engineer_features(train_df)
y = train_df["price"].values
X_test = engineer_features(test_df)

# Target encoding for zipcodes within cross-validation to prevent leakage
cat_cols = ["zipcode"]
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

# 2. 5-Fold Cross-Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 42,
    "n_estimators": 1500,
    "enable_categorical": True,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = xgb.XGBRegressor(**xgb_params)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / n_splits

    fold_rmse = root_mean_squared_error(y_val, val_pred)

# 3. Overall Evaluation
overall_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV Out-of-Fold RMSE: {overall_rmse:.4f}")

# 4. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "price": test_preds})
submission.to_csv("./working/submission.csv", index=False)
