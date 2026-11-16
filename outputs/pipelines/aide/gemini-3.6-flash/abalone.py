import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Determine target column
target_col = (
    "Class_number_of_rings"
    if "Class_number_of_rings" in train_df.columns
    else [c for c in train_df.columns if "ring" in c.lower() or "class" in c.lower()][0]
)

# Extract target and drop from training features
y = train_df[target_col].values.astype(np.float32)
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()


# 2. Advanced Feature Engineering Function (Allometric & Demographic Aggregations)
def create_features(data_df):
    df = data_df.copy()

    # Identify key continuous columns flexibly
    cols = df.columns

    def find_col(keyword, fallback_idx):
        matches = [c for c in cols if keyword in c.lower()]
        return matches[0] if matches else cols[fallback_idx]

    len_col = find_col("length", 1)
    diam_col = find_col("diam", 2)
    height_col = find_col("height", 3)
    whole_col = find_col("whole", 4)
    shucked_col = find_col("shucked", 5)
    viscera_col = find_col("viscera", 6)
    shell_col = find_col("shell", 7)
    sex_col = [c for c in df.columns if "sex" in c.lower()][0]

    length = df[len_col]
    diameter = df[diam_col]
    height = df[height_col]
    whole_w = df[whole_col]
    shucked_w = df[shucked_col]
    viscera_w = df[viscera_col]
    shell_w = df[shell_col]

    eps = 1e-6

    # Baseline Dimensional & Weight Ratios
    df["Volume"] = length * diameter * height
    df["Density"] = whole_w / (df["Volume"] + eps)
    df["Shell_ratio"] = shell_w / (whole_w + eps)
    df["Shucked_ratio"] = shucked_w / (whole_w + eps)
    df["Viscera_ratio"] = viscera_w / (whole_w + eps)
    df["Water_weight"] = whole_w - (shucked_w + viscera_w + shell_w)
    df["Water_ratio"] = df["Water_weight"] / (whole_w + eps)
    df["Meat_Shell_ratio"] = shucked_w / (shell_w + eps)

    df["Length_Diam_ratio"] = length / (diameter + eps)
    df["Height_Length_ratio"] = height / (length + eps)

    # Allometric Growth & Biological Features
    df["Surface_Area"] = length * diameter
    df["Condition_Factor"] = whole_w / (length**3 + eps)
    df["Shell_Thickness_Proxy"] = height / (df["Surface_Area"] + eps)
    df["Shell_Mass_Density"] = shell_w / (df["Surface_Area"] + eps)
    df["Allometric_Volume_Ratio"] = df["Volume"] / (length**3 + eps)
    df["Viscera_Shell_ratio"] = viscera_w / (shell_w + eps)

    # Demographic Aggregations Grouped by Sex
    agg_cols = [len_col, height_col, whole_col, shell_col, "Volume", "Condition_Factor"]
    grouped = df.groupby(sex_col)[agg_cols].transform
    for col_name in agg_cols:
        df[f"{sex_col}_{col_name}_mean"] = grouped("mean")[col_name]
        df[f"{sex_col}_{col_name}_std"] = grouped("std")[col_name]
        df[f"{sex_col}_{col_name}_diff_mean"] = (
            df[col_name] - df[f"{sex_col}_{col_name}_mean"]
        )

    # Log Transforms
    for c in [
        "Volume",
        "Surface_Area",
        len_col,
        diam_col,
        height_col,
        whole_col,
        shucked_col,
        viscera_col,
        shell_col,
    ]:
        if c in df.columns:
            df[f"log_{c}"] = np.log1p(np.maximum(0, df[c]))

    # One-hot encode categorical feature 'Sex'
    df = pd.get_dummies(df, columns=[sex_col], drop_first=False)

    return df


# Apply feature engineering to combined set
combined_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
combined_df = create_features(combined_df)

n_train = len(X_train_raw)
X = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# Ensure strictly numeric types
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

# 3. Validation Strategy and Modeling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X), dtype=np.float32)
test_preds = np.zeros(len(X_test), dtype=np.float32)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

cb_params = {
    "iterations": 1200,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    val_pred_lgb = lgb_model.predict(X_va)
    test_pred_lgb = lgb_model.predict(X_test)

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_pred_xgb = xgb_model.predict(X_va)
    test_pred_xgb = xgb_model.predict(X_test)

    # CatBoost
    cb_model = cb.CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )
    val_pred_cb = cb_model.predict(X_va)
    test_pred_cb = cb_model.predict(X_test)

    # Ensemble predictions for fold
    val_pred_ensemble = (
        (val_pred_lgb * 0.4) + (val_pred_xgb * 0.3) + (val_pred_cb * 0.3)
    )
    test_pred_ensemble = (
        (test_pred_lgb * 0.4) + (test_pred_xgb * 0.3) + (test_pred_cb * 0.3)
    )

    oof_preds[val_idx] = val_pred_ensemble
    test_preds += test_pred_ensemble / kf.n_splits

# 4. Evaluation Score Computation
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE Score: {cv_rmse:.5f}")

# 5. Format and Save Submission File
submission = pd.DataFrame(
    {"id": np.arange(len(X_test)), "Class_number_of_rings": test_preds}
)

submission_path = "./working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved successfully to {submission_path}")
