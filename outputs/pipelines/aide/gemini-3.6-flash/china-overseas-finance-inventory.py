import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
SEED = 42

# Paths
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

# Read datasets
train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "installed_capacity"

# Separate target and clean missing values
train = train.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train[target_col].values
X_train_raw = train.drop(columns=[target_col])
X_test_raw = test.copy()

n_train = len(X_train_raw)
combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# Process basic features (datetimes, categoricals, numericals)
features_df = pd.DataFrame(index=combined.index)
cat_cols = []
num_cols = []

for col in combined.columns:
    col_data = combined[col]

    # Process potential datetime columns
    if col_data.dtype == "object" or pd.api.types.is_datetime64_any_dtype(col_data):
        try:
            date_series = pd.to_datetime(col_data, errors="coerce")
            if date_series.notna().sum() > 0.3 * len(combined):
                features_df[f"{col}_year"] = date_series.dt.year
                features_df[f"{col}_month"] = date_series.dt.month
                features_df[f"{col}_day"] = date_series.dt.day
                features_df[f"{col}_quarter"] = date_series.dt.quarter
                features_df[f"{col}_dayofweek"] = date_series.dt.dayofweek
                num_cols.append(f"{col}_year")
                continue
        except Exception:
            pass

    if pd.api.types.is_numeric_dtype(col_data):
        features_df[col] = col_data
        num_cols.append(col)
    else:
        str_series = col_data.astype(str)
        # Frequency encoding
        freq_map = str_series.value_counts(normalize=True).to_dict()
        features_df[f"{col}_freq"] = str_series.map(freq_map)
        # Factorize encoding
        codes, _ = pd.factorize(str_series)
        features_df[f"{col}_cat"] = codes
        cat_cols.append(col)

# Correct group aggregate features across key categorical variables
for cat in cat_cols:
    str_cat = combined[cat].astype(str)
    n_unique = str_cat.nunique()
    if 2 <= n_unique <= 500:
        for num in num_cols:
            if num in features_df.columns and features_df[num].notna().sum() > 0:
                grouped = features_df.groupby(str_cat)[num]
                features_df[f"{cat}_{num}_mean"] = grouped.transform("mean")
                features_df[f"{cat}_{num}_std"] = grouped.transform("std").fillna(0)
                features_df[f"{cat}_{num}_min"] = grouped.transform("min")
                features_df[f"{cat}_{num}_max"] = grouped.transform("max")

# Separate back into train and test sets
X_train = features_df.iloc[:n_train].reset_index(drop=True)
X_test = features_df.iloc[n_train:].reset_index(drop=True)

# Setup 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof_xgb = np.zeros(len(X_train))
oof_lgb = np.zeros(len(X_train))
test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # Train XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / kf.n_splits

    # Train LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / kf.n_splits

# Weighted Ensemble (0.5 XGBoost + 0.5 LightGBM)
oof_preds = 0.5 * oof_xgb + 0.5 * oof_lgb
test_preds = 0.5 * test_xgb + 0.5 * test_lgb

# Apply non-negativity constraint for capacity
oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

# Compute out-of-fold RMSE
cv_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"Validation RMSE: {cv_rmse:.5f}")

# Save test predictions to submission.csv
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "installed_capacity": test_preds}
)
submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
