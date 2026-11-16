import json
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

# 1. Setup required directories
os.makedirs("./working/models", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Load raw data
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

target_col = "Total Enrollment"
assert (
    target_col in train_raw.columns
), f"Target column '{target_col}' not found in train data."

# 3. Perform train/validation split BEFORE feature engineering to strictly prevent leakage
train_df, val_df = train_test_split(train_raw, test_size=0.2, random_state=42)
test_df = test_raw.copy()

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# 4. Categorize feature columns
ignore_cols = [target_col]
if "id" in train_df.columns:
    ignore_cols.append("id")

feature_cols = [c for c in train_df.columns if c not in ignore_cols]

cat_cols = []
num_cols = []
date_cols = []

for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(train_df[c]):
        date_cols.append(c)
    elif pd.api.types.is_numeric_dtype(train_df[c]):
        num_cols.append(c)
    else:
        cat_cols.append(c)

# 5. Feature Engineering Pipeline

# A. Process Datetime Features
for col in date_cols:
    for df in [train_df, val_df, test_df]:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_quarter"] = df[col].dt.quarter
        df[f"{col}_sin_month"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12.0)
        df[f"{col}_cos_month"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12.0)
        df.drop(columns=[col], inplace=True)

# Update numerical columns list with engineered date features
added_date_feats = [
    c for c in train_df.columns if any(c.startswith(f"{dc}_") for dc in date_cols)
]
num_cols = num_cols + added_date_feats

# B. Process Categorical Features & Frequency Encoding
for c in cat_cols:
    train_df[c] = train_df[c].astype(str).fillna("Missing")
    val_df[c] = val_df[c].astype(str).fillna("Missing")
    test_df[c] = test_df[c].astype(str).fillna("Missing")

    # Frequency Encoding computed ONLY on training split
    freq_map = train_df[c].value_counts(normalize=True).to_dict()
    train_df[f"{c}_freq"] = train_df[c].map(freq_map).fillna(0)
    val_df[f"{c}_freq"] = val_df[c].map(freq_map).fillna(0)
    test_df[f"{c}_freq"] = test_df[c].map(freq_map).fillna(0)

if cat_cols:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[cat_cols] = oe.fit_transform(train_df[cat_cols])
    val_df[cat_cols] = oe.transform(val_df[cat_cols])
    test_df[cat_cols] = oe.transform(test_df[cat_cols])

# C. Process Numerical Features (Train-fitted Median Imputation)
for c in num_cols:
    median_val = train_df[c].median()
    if pd.isna(median_val):
        median_val = 0.0
    train_df[c] = train_df[c].fillna(median_val)
    val_df[c] = val_df[c].fillna(median_val)
    test_df[c] = test_df[c].fillna(median_val)

# D. Group Aggregation Features
if len(cat_cols) > 0 and len(num_cols) > 0:
    primary_cat = cat_cols[0]
    primary_num = num_cols[0]

    # Calculate statistics purely on training split
    group_stats = (
        train_df.groupby(primary_cat)[primary_num].agg(["mean", "std"]).reset_index()
    )
    group_stats.columns = [
        primary_cat,
        f"{primary_cat}_{primary_num}_grp_mean",
        f"{primary_cat}_{primary_num}_grp_std",
    ]

    train_df = train_df.merge(group_stats, on=primary_cat, how="left")
    val_df = val_df.merge(group_stats, on=primary_cat, how="left")
    test_df = test_df.merge(group_stats, on=primary_cat, how="left")

    for col_added in [
        f"{primary_cat}_{primary_num}_grp_mean",
        f"{primary_cat}_{primary_num}_grp_std",
    ]:
        train_df[col_added] = train_df[col_added].fillna(0)
        val_df[col_added] = val_df[col_added].fillna(0)
        test_df[col_added] = test_df[col_added].fillna(0)

processed_features = [c for c in train_df.columns if c not in [target_col, "id"]]

# 6. Extract Feature Matrices and Targets
X_train = train_df[processed_features]
y_train = train_df[target_col].values

X_val = val_df[processed_features]
y_val = val_df[target_col].values

X_test = test_df[processed_features]

# Target representations for multi-scale training
y_train_log = np.log1p(np.maximum(0, y_train))
y_val_log = np.log1p(np.maximum(0, y_val))

# 7. Model Zoo Training with Artifact Persistence
val_preds_dict = {}
test_preds_dict = {}

# Model 1: LightGBM Log-Target
lgb_log = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_log.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
joblib.dump(lgb_log, "./working/models/lgb_log.pkl")
val_preds_dict["lgb_log"] = np.expm1(lgb_log.predict(X_val))
test_preds_dict["lgb_log"] = np.expm1(lgb_log.predict(X_test))

# Model 2: XGBoost Log-Target
xgb_log = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)
xgb_log.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False,
)
joblib.dump(xgb_log, "./working/models/xgb_log.pkl")
val_preds_dict["xgb_log"] = np.expm1(xgb_log.predict(X_val))
test_preds_dict["xgb_log"] = np.expm1(xgb_log.predict(X_test))

# Model 3: LightGBM Raw-Target
lgb_raw = lgb.LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_raw.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)
joblib.dump(lgb_raw, "./working/models/lgb_raw.pkl")
val_preds_dict["lgb_raw"] = lgb_raw.predict(X_val)
test_preds_dict["lgb_raw"] = lgb_raw.predict(X_test)

# Model 4: XGBoost Raw-Target
xgb_raw = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)
xgb_raw.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
joblib.dump(xgb_raw, "./working/models/xgb_raw.pkl")
val_preds_dict["xgb_raw"] = xgb_raw.predict(X_val)
test_preds_dict["xgb_raw"] = xgb_raw.predict(X_test)

# 8. Constrained Optimization of Blending Weights
model_names = list(val_preds_dict.keys())
V = np.column_stack([val_preds_dict[m] for m in model_names])
T = np.column_stack([test_preds_dict[m] for m in model_names])


def loss_func(weights):
    w = weights / (np.sum(weights) + 1e-12)
    pred = np.maximum(0, V @ w)
    return np.sqrt(mean_squared_error(y_val, pred))


init_w = np.ones(len(model_names)) / len(model_names)
bounds = [(0, 1)] * len(model_names)
opt_res = minimize(loss_func, init_w, bounds=bounds, method="L-BFGS-B")
opt_weights = opt_res.x / np.sum(opt_res.x)

# 9. Final Holdout Evaluation & Test Inference
val_preds_final = np.maximum(0, V @ opt_weights)
test_preds_final = np.maximum(0, T @ opt_weights)

val_rmse = np.sqrt(mean_squared_error(y_val, val_preds_final))

# 10. Generate Submission File
if "id" in test_raw.columns:
    ids = test_raw["id"]
else:
    ids = pd.Series(range(len(test_raw)), name="id")

sub_df = pd.DataFrame({"id": ids, "Total Enrollment": test_preds_final})
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_rmse}")
