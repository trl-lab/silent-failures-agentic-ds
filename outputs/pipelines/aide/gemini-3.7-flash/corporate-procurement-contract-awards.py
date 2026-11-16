import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from scipy.optimize import minimize_scalar
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

# Set paths
INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

# Load data
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

target_col = "Contract Award Amount (USD)"
y = train_df[target_col].values

# Combine for uniform feature engineering
combined_df = pd.concat(
    [train_df.drop(columns=[target_col]), test_df], ignore_index=True
)

# Date processing & pairwise duration features
date_cols = [
    col
    for col in combined_df.columns
    if "date" in col.lower() or "period" in col.lower() or "time" in col.lower()
]

parsed_dates = {}
for col in date_cols:
    try:
        dt_series = pd.to_datetime(combined_df[col], errors="coerce")
        if dt_series.notna().sum() > 0:
            parsed_dates[col] = dt_series
            combined_df[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(int)
            combined_df[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
            combined_df[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(int)
            combined_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(
                int
            )
            combined_df[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(int)
            combined_df.drop(columns=[col], inplace=True)
    except Exception:
        pass

# Compute pairwise date differences
date_keys = list(parsed_dates.keys())
for i in range(len(date_keys)):
    for j in range(i + 1, len(date_keys)):
        col_i = date_keys[i]
        col_j = date_keys[j]
        diff_days = (parsed_dates[col_i] - parsed_dates[col_j]).dt.days
        combined_df[f"diff_days_{col_i}_{col_j}"] = diff_days.fillna(-999).astype(float)

# Process categorical and text features
cat_cols = []
for col in combined_df.columns:
    if combined_df[col].dtype == "object" or combined_df[col].dtype.name == "category":
        str_col = combined_df[col].astype(str)
        str_lens = str_col.str.len()
        if str_lens.max() > 50:
            combined_df[f"{col}_len"] = str_lens
            combined_df[f"{col}_word_count"] = str_col.str.split().str.len()

        # Frequency encoding
        freq = combined_df[col].value_counts(dropna=False)
        combined_df[f"{col}_freq"] = combined_df[col].map(freq).fillna(0).astype(int)

        combined_df[col] = combined_df[col].fillna("Missing").astype(str)
        cat_cols.append(col)
    elif np.issubdtype(combined_df[col].dtype, np.number):
        combined_df[col] = combined_df[col].fillna(combined_df[col].median())

# Prepare data representations for CatBoost and LightGBM
n_train = len(train_df)

# CatBoost data
X_cb_train = combined_df.iloc[:n_train].copy().reset_index(drop=True)
X_cb_test = combined_df.iloc[n_train:].copy().reset_index(drop=True)

# LightGBM data with ordinal encoding for categorical columns
X_lgb_train = X_cb_train.copy()
X_lgb_test = X_cb_test.copy()

for col in cat_cols:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_lgb_train[col] = oe.fit_transform(X_lgb_train[[col]]).astype(int)
    X_lgb_test[col] = oe.transform(X_lgb_test[[col]]).astype(int)

# 5-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_cb = np.zeros(n_train)
oof_lgb = np.zeros(n_train)
test_cb = np.zeros(len(test_df))
test_lgb = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.04,
    "num_leaves": 45,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_cb_train, y)):
    # --- CatBoost Training ---
    X_tr_cb, y_tr = X_cb_train.iloc[train_idx], y[train_idx]
    X_va_cb, y_va = X_cb_train.iloc[val_idx], y[val_idx]

    train_pool = Pool(X_tr_cb, y_tr, cat_features=cat_cols)
    val_pool = Pool(X_va_cb, y_va, cat_features=cat_cols)
    test_pool = Pool(X_cb_test, cat_features=cat_cols)

    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )
    cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

    oof_cb[val_idx] = cb_model.predict(val_pool)
    test_cb += cb_model.predict(test_pool) / n_splits

    # --- LightGBM Training ---
    X_tr_lgb = X_lgb_train.iloc[train_idx]
    X_va_lgb = X_lgb_train.iloc[val_idx]

    lgb_train = lgb.Dataset(X_tr_lgb, label=y_tr, categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(
        X_va_lgb, label=y_va, reference=lgb_train, categorical_feature=cat_cols
    )

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    oof_lgb[val_idx] = lgb_model.predict(
        X_va_lgb, num_iteration=lgb_model.best_iteration
    )
    test_lgb += (
        lgb_model.predict(X_lgb_test, num_iteration=lgb_model.best_iteration) / n_splits
    )


# Optimal Linear Blend Search
def blend_loss(w):
    pred = w * oof_cb + (1.0 - w) * oof_lgb
    if (y >= 0).all():
        pred = np.clip(pred, 0, None)
    return root_mean_squared_error(y, pred)


res = minimize_scalar(blend_loss, bounds=(0.0, 1.0), method="bounded")
best_w = res.x

oof_blend = best_w * oof_cb + (1.0 - best_w) * oof_lgb
test_blend = best_w * test_cb + (1.0 - best_w) * test_lgb

if (y >= 0).all():
    oof_blend = np.clip(oof_blend, 0, None)
    test_blend = np.clip(test_blend, 0, None)

cv_rmse = root_mean_squared_error(y, oof_blend)
print(f"Optimal Blend Weight (CatBoost): {best_w:.4f}")
print(f"Validation RMSE: {cv_rmse:.4f}")

# Save submission file
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_blend)), "Contract Award Amount (USD)": test_blend}
)
submission_path = os.path.join(WORKING_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission_df.shape}")
