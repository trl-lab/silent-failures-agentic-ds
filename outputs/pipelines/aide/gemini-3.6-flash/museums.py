import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Create output directory
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "REVENUE15"

# Identify non-feature ID columns
drop_cols = [target_col]
for col in ["id", "ID", "index", "MID", "EIN"]:
    if col in train_df.columns and col != target_col:
        drop_cols.append(col)

# Feature engineering: missing value counts per row
train_df["num_missing"] = train_df.isnull().sum(axis=1)
test_df["num_missing"] = test_df.isnull().sum(axis=1)

# Feature engineering: 3-digit ZIP prefix if ZIP column exists
zip_cols = [c for c in train_df.columns if "ZIP" in c.upper()]
if zip_cols:
    zip_col = zip_cols[0]
    train_df["ZIP3"] = train_df[zip_col].astype(str).str[:3]
    test_df["ZIP3"] = test_df[zip_col].astype(str).str[:3]

features = [c for c in train_df.columns if c not in drop_cols]

# Identify categorical features
cat_cols = [
    col
    for col in features
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category"
]

# Frequency encoding for categorical features
for col in cat_cols:
    train_str = train_df[col].fillna("Missing").astype(str)
    test_str = test_df[col].fillna("Missing").astype(str)

    freq = train_str.value_counts()
    freq_col_name = f"{col}_freq"
    train_df[freq_col_name] = train_str.map(freq).fillna(0).astype(float)
    test_df[freq_col_name] = test_str.map(freq).fillna(0).astype(float)

features = [c for c in train_df.columns if c not in drop_cols]
cat_cols = [
    col
    for col in features
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category"
]

# Prepare feature matrices for LightGBM and CatBoost
X_lgb = train_df[features].copy()
X_test_lgb = test_df[features].copy()

X_cb = train_df[features].copy()
X_test_cb = test_df[features].copy()

for col in cat_cols:
    train_str = train_df[col].fillna("Missing").astype(str)
    test_str = test_df[col].fillna("Missing").astype(str)
    categories = pd.concat([train_str, test_str]).unique()

    X_lgb[col] = pd.Categorical(train_str, categories=categories)
    X_test_lgb[col] = pd.Categorical(test_str, categories=categories)

    X_cb[col] = train_str
    X_test_cb[col] = test_str

y = train_df[target_col].values

# 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_cb = np.zeros(len(train_df))
test_preds_lgb = np.zeros(len(test_df))
test_preds_cb = np.zeros(len(test_df))

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
    "min_child_samples": 20,
    "random_state": 42,
    "n_estimators": 1500,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_lgb, y)):
    y_train, y_val = y[train_idx], y[val_idx]

    # LightGBM Training
    X_tr_lgb, X_va_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr_lgb,
        y_train,
        eval_set=[(X_va_lgb, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va_lgb)
    test_preds_lgb += model_lgb.predict(X_test_lgb) / kf.n_splits

    # CatBoost Training
    X_tr_cb, X_va_cb = X_cb.iloc[train_idx], X_cb.iloc[val_idx]
    model_cb = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        cat_features=cat_cols,
    )
    model_cb.fit(
        X_tr_cb,
        y_train,
        eval_set=(X_va_cb, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_cb[val_idx] = model_cb.predict(X_va_cb)
    test_preds_cb += model_cb.predict(X_test_cb) / kf.n_splits


# Optimize blend weight between LightGBM and CatBoost
def loss_func(weights):
    w = weights[0]
    blend = w * oof_lgb + (1 - w) * oof_cb
    return np.sqrt(mean_squared_error(y, blend))


res = minimize(loss_func, [0.5], bounds=[(0, 1)], method="L-BFGS-B")
best_w = res.x[0]

oof_blend = best_w * oof_lgb + (1 - best_w) * oof_cb
cv_rmse = np.sqrt(mean_squared_error(y, oof_blend))
print(f"Validation RMSE: {cv_rmse:.4f}")

# Final blended test predictions
test_preds = best_w * test_preds_lgb + (1 - best_w) * test_preds_cb
submission = pd.DataFrame({"id": np.arange(len(test_df)), "REVENUE15": test_preds})
submission.to_csv("./working/submission.csv", index=False)
