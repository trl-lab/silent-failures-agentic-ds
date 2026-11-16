import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


# Reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "TOTAL_MWH"
features = [c for c in train_df.columns if c != target_col]

y_train = train_df[target_col].values.astype(np.float64)

# Separate categorical and numerical features
cat_cols = []
num_cols = []

for col in features:
    if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Clean and format dataframes
X_train = train_df[features].copy()
X_test = test_df[features].copy()

for c in cat_cols:
    X_train[c] = X_train[c].astype(str).fillna("__MISSING__").astype("category")
    X_test[c] = X_test[c].astype(str).fillna("__MISSING__").astype("category")

for c in num_cols:
    X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

# Cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
test_lgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))

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
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# Convert categoricals to strings for CatBoost
X_train_cat_cb = X_train.copy()
X_test_cat_cb = X_test.copy()
for c in cat_cols:
    X_train_cat_cb[c] = X_train_cat_cb[c].astype(str)
    X_test_cat_cb[c] = X_test_cat_cb[c].astype(str)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    # Data splits
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # --- LightGBM ---
    trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_data = lgb.Dataset(
        X_va, label=y_va, reference=trn_data, categorical_feature=cat_cols
    )

    lgb_model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=3000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_lgb[val_idx] = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    test_lgb += (
        lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits
    )

    # --- CatBoost ---
    X_tr_cb, X_va_cb = X_train_cat_cb.iloc[train_idx], X_train_cat_cb.iloc[val_idx]

    cb_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
        cat_features=cat_cols,
    )

    cb_model.fit(
        X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50, verbose=False
    )

    oof_cat[val_idx] = cb_model.predict(X_va_cb)
    test_cat += cb_model.predict(X_test_cat_cb) / n_splits

# Blend predictions
oof_blend = 0.5 * oof_lgb + 0.5 * oof_cat
test_blend = 0.5 * test_lgb + 0.5 * test_cat

# Post-processing: electricity consumption/sales is non-negative
oof_blend = np.clip(oof_blend, 0, None)
test_blend = np.clip(test_blend, 0, None)

cv_score = root_mean_squared_error(y_train, oof_blend)
print(f"5-Fold CV Out-of-Fold RMSE: {cv_score:.4f}")

# Prepare submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_blend)), "TOTAL_MWH": test_blend})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
