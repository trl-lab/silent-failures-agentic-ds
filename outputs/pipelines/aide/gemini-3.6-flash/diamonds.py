import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

# Load dataset
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)


def preprocess_features(df):
    data = df.copy()

    # Cast Categorical dtypes to string before mapping to avoid setitem errors
    for col in ["cut", "color", "clarity"]:
        if col in data.columns:
            data[col] = data[col].astype(str)

    cut_map = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    color_map = {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7}
    clarity_map = {
        "I1": 1,
        "SI2": 2,
        "SI1": 3,
        "VS2": 4,
        "VS1": 5,
        "VVS2": 6,
        "VVS1": 7,
        "IF": 8,
    }

    if "cut" in data.columns:
        data["cut_num"] = data["cut"].map(cut_map).fillna(0).astype(float)
    if "color" in data.columns:
        data["color_num"] = data["color"].map(color_map).fillna(0).astype(float)
    if "clarity" in data.columns:
        data["clarity_num"] = data["clarity"].map(clarity_map).fillna(0).astype(float)

    # Physical dimensions handling
    x = data["x"].replace(0, np.nan)
    y = data["y"].replace(0, np.nan)
    z = data["z"].replace(0, np.nan)

    # Dimensional calculations
    data["volume"] = x * y * z
    data["volume"] = data["volume"].fillna(data["volume"].median())

    data["carat_per_volume"] = data["carat"] / (data["volume"] + 1e-5)
    data["calc_depth"] = 200.0 * z / (x + y + 1e-5)
    data["depth_diff"] = data["depth"] - data["calc_depth"]

    data["dim_ratio_xy"] = x / (y + 1e-5)
    data["dim_ratio_xz"] = x / (z + 1e-5)
    data["dim_ratio_yz"] = y / (z + 1e-5)

    # Non-linear transformations
    data["carat_cbrt"] = np.cbrt(data["carat"])
    data["carat_sq"] = data["carat"] ** 2
    data["carat_log"] = np.log1p(data["carat"])

    # Composite quality scores
    data["quality_score"] = data["cut_num"] + data["color_num"] + data["clarity_num"]
    data["quality_prod"] = data["cut_num"] * data["color_num"] * data["clarity_num"]

    data["carat_x_quality"] = data["carat"] * data["quality_score"]
    data["carat_x_cut"] = data["carat"] * data["cut_num"]
    data["carat_x_color"] = data["carat"] * data["color_num"]
    data["carat_x_clarity"] = data["carat"] * data["clarity_num"]

    # Drop original non-numeric columns
    cat_cols = ["cut", "color", "clarity"]
    data = data.drop(columns=[c for c in cat_cols if c in data.columns])

    return data


# Preprocess features
train_processed = preprocess_features(train_df)
test_processed = preprocess_features(test_df)

target_col = "price"
features = [c for c in train_processed.columns if c != target_col and c != "id"]

X = train_processed[features]
y = train_processed[target_col]
X_test = test_processed[features]

# 5-Fold Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_processed))
test_preds = np.zeros(len(test_processed))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # XGBoost Regressor
    model_xgb = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="rmse",
        n_jobs=-1,
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # LightGBM Regressor
    model_lgb = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_xgb = model_xgb.predict(X_val)
    val_lgb = model_lgb.predict(X_val)
    oof_preds[val_idx] = 0.5 * val_xgb + 0.5 * val_lgb

    test_xgb = model_xgb.predict(X_test)
    test_lgb = model_lgb.predict(X_test)
    test_preds += (0.5 * test_xgb + 0.5 * test_lgb) / kf.n_splits

val_rmse = root_mean_squared_error(y, oof_preds)
print(f"Validation RMSE: {val_rmse:.4f}")

# Save submission file
os.makedirs("./working", exist_ok=True)
test_ids = test_df["id"] if "id" in test_df.columns else np.arange(len(test_df))
submission = pd.DataFrame({"id": test_ids, "price": np.clip(test_preds, 0, None)})
submission.to_csv("./working/submission.csv", index=False)
