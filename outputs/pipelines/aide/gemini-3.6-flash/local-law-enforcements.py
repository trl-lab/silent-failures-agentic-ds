import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_PATH = "./working/submission.csv"

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "number_of_officers"
y = train_df[target_col].values.astype(np.float64)
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

os.makedirs("./working", exist_ok=True)


# Clean column names to comply with LightGBM and CatBoost requirements
def clean_col_names(df):
    renames = {}
    for col in df.columns:
        clean = re.sub(r"[^\w]", "_", str(col))
        renames[col] = clean
    return df.rename(columns=renames)


X = clean_col_names(X)
X_test = clean_col_names(X_test)

common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

# Compute row-wise missing value counts
X["num_missing"] = X.isnull().sum(axis=1)
X_test["num_missing"] = X_test.isnull().sum(axis=1)

# Extract 3D Cartesian coordinates if geographic features are present
lat_cols = [c for c in X.columns if "lat" in c.lower()]
lon_cols = [c for c in X.columns if "lon" in c.lower()]

if lat_cols and lon_cols:
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    for df in [X, X_test]:
        rad_lat = np.radians(df[lat_col].fillna(0))
        rad_lon = np.radians(df[lon_col].fillna(0))
        df["coord_x"] = np.cos(rad_lat) * np.cos(rad_lon)
        df["coord_y"] = np.cos(rad_lat) * np.sin(rad_lon)
        df["coord_z"] = np.sin(rad_lat)

# Process datetime and non-numeric categorical columns
cat_cols = []
for col in list(X.columns):
    if col in ["num_missing", "coord_x", "coord_y", "coord_z"]:
        continue
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        for df in [X, X_test]:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df.drop(columns=[col], inplace=True)
    elif not pd.api.types.is_numeric_dtype(X[col]):
        cat_cols.append(col)
        s_train = X[col].astype(str).fillna("missing")
        s_test = X_test[col].astype(str).fillna("missing")

        # String length feature
        X[f"{col}_len"] = s_train.str.len()
        X_test[f"{col}_len"] = s_test.str.len()

        # Frequency encoding
        full_s = pd.concat([s_train, s_test], axis=0)
        freq_map = full_s.value_counts().to_dict()
        X[f"{col}_freq"] = s_train.map(freq_map).fillna(0)
        X_test[f"{col}_freq"] = s_test.map(freq_map).fillna(0)

        # Unified Categorical Encoding for LightGBM
        cat_dtype = pd.CategoricalDtype(categories=full_s.unique(), ordered=False)
        X[col] = s_train.astype(cat_dtype)
        X_test[col] = s_test.astype(cat_dtype)

# Group-level aggregate features for numerical columns grouped by key categoricals
num_cols = [
    c
    for c in X.columns
    if pd.api.types.is_numeric_dtype(X[c])
    and not isinstance(X[c].dtype, pd.CategoricalDtype)
    and X[c].nunique() > 10
]
group_cats = [c for c in cat_cols if 2 <= X[c].nunique() <= 200]

combined = pd.concat([X, X_test], axis=0)
for cat in group_cats:
    for num in num_cols[:6]:  # Select key numerical features
        mean_series = combined.groupby(cat, observed=False)[num].transform("mean")
        std_series = (
            combined.groupby(cat, observed=False)[num].transform("std").fillna(0)
        )

        X[f"{cat}_{num}_mean"] = mean_series.iloc[: len(X)].values
        X_test[f"{cat}_{num}_mean"] = mean_series.iloc[len(X) :].values
        X[f"{cat}_{num}_std"] = std_series.iloc[: len(X)].values
        X_test[f"{cat}_{num}_std"] = std_series.iloc[len(X) :].values

# Prepare CatBoost feature set with explicit string conversion for categoricals
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str).fillna("missing")
    X_test_cb[col] = X_test_cb[col].astype(str).fillna("missing")

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42,
    "n_estimators": 1500,
}

cb_params = {
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": 0,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    X_train_cb = X_cb.iloc[train_idx]
    X_val_cb = X_cb.iloc[val_idx]

    # Train LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    # Train CatBoost
    cb_model = CatBoostRegressor(**cb_params)
    cb_model.fit(
        X_train_cb,
        y_train,
        eval_set=(X_val_cb, y_val),
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=False,
    )

    # Predict and Blend
    val_pred_lgb = lgb_model.predict(X_val)
    val_pred_cb = cb_model.predict(X_val_cb)
    val_pred = 0.5 * val_pred_lgb + 0.5 * val_pred_cb

    test_pred_lgb = lgb_model.predict(X_test)
    test_pred_cb = cb_model.predict(X_test_cb)
    test_pred = 0.5 * test_pred_lgb + 0.5 * test_pred_cb

    oof_preds[val_idx] = np.clip(val_pred, 0, None)
    test_preds += np.clip(test_pred, 0, None) / kf.n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Save submission file
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "number_of_officers": test_preds})
sub.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH} with shape {sub.shape}")
