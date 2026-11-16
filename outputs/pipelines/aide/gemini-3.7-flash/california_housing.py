import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "medianHouseValue"
y = train_df[target_col].values

# Major California city coordinates (lat, lon)
CITIES = {
    "los_angeles": (34.0522, -118.2437),
    "san_francisco": (37.7749, -122.4194),
    "san_diego": (32.7157, -117.1611),
    "san_jose": (37.3382, -121.8863),
    "sacramento": (38.5816, -121.4944),
    "fresno": (36.7468, -119.7726),
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()

    # Ratios and census block characteristics
    df_feat["rooms_per_household"] = df_feat["totalRooms"] / (
        df_feat["households"] + 1e-5
    )
    df_feat["bedrooms_per_household"] = df_feat["totalBedrooms"] / (
        df_feat["households"] + 1e-5
    )
    df_feat["bedrooms_per_room"] = df_feat["totalBedrooms"] / (
        df_feat["totalRooms"] + 1e-5
    )
    df_feat["population_per_household"] = df_feat["population"] / (
        df_feat["households"] + 1e-5
    )

    # Interaction terms
    df_feat["income_per_room"] = df_feat["medianIncome"] / (
        df_feat["rooms_per_household"] + 1e-5
    )
    df_feat["income_per_bedroom"] = df_feat["medianIncome"] / (
        df_feat["bedrooms_per_room"] + 1e-5
    )

    # Spatial features
    lat = df_feat["latitude"].values
    lon = df_feat["longitude"].values

    # Rotated coordinates
    df_feat["rot45_x"] = 0.7071 * lon + 0.7071 * lat
    df_feat["rot45_y"] = -0.7071 * lon + 0.7071 * lat
    df_feat["rot30_x"] = np.cos(np.pi / 6) * lon + np.sin(np.pi / 6) * lat
    df_feat["rot30_y"] = -np.sin(np.pi / 6) * lon + np.cos(np.pi / 6) * lat

    # Distances to major cities
    for city, (c_lat, c_lon) in CITIES.items():
        df_feat[f"dist_{city}"] = np.sqrt((lat - c_lat) ** 2 + (lon - c_lon) ** 2)

    # Min distance to any major coastal metro (LA, SF, SD, SJ)
    coastal_dists = [
        df_feat[f"dist_{c}"]
        for c in ["los_angeles", "san_francisco", "san_diego", "san_jose"]
    ]
    df_feat["min_dist_major_metro"] = np.minimum.reduce(coastal_dists)

    return df_feat


# Apply feature engineering
X_train = engineer_features(train_df.drop(columns=[target_col]))
X_test = engineer_features(test_df)

feature_cols = [col for col in X_train.columns]

# LightGBM Parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# 5-fold Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        params,
        trn_data,
        num_boost_round=3000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.4f}")

# Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "medianHouseValue": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
