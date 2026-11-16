import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb

try:
    from catboost import CatBoostRegressor, Pool

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Cash Disbursements (US$)"
y = train_df[target_col].values.astype(np.float64)
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Feature Engineering
lgb_features = pd.DataFrame(index=combined_df.index)
cb_features = pd.DataFrame(index=combined_df.index)

cat_cols = []
date_cols = []

for col in combined_df.columns:
    if combined_df[col].dtype == "object":
        sample_vals = combined_df[col].dropna().astype(str)
        is_date = False
        if len(sample_vals) > 0:
            try:
                parsed_dates = pd.to_datetime(combined_df[col], errors="coerce")
                if parsed_dates.notna().sum() > 0.5 * len(sample_vals):
                    is_date = True
                    date_cols.append(col)
                    # Date features
                    lgb_features[f"{col}_year"] = parsed_dates.dt.year.fillna(
                        -1
                    ).astype(int)
                    lgb_features[f"{col}_month"] = parsed_dates.dt.month.fillna(
                        -1
                    ).astype(int)
                    lgb_features[f"{col}_day"] = parsed_dates.dt.day.fillna(-1).astype(
                        int
                    )
                    lgb_features[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek.fillna(
                        -1
                    ).astype(int)
                    lgb_features[f"{col}_quarter"] = parsed_dates.dt.quarter.fillna(
                        -1
                    ).astype(int)

                    cb_features[f"{col}_year"] = lgb_features[f"{col}_year"]
                    cb_features[f"{col}_month"] = lgb_features[f"{col}_month"]
                    cb_features[f"{col}_day"] = lgb_features[f"{col}_day"]
                    cb_features[f"{col}_dayofweek"] = lgb_features[f"{col}_dayofweek"]
                    cb_features[f"{col}_quarter"] = lgb_features[f"{col}_quarter"]
            except Exception:
                is_date = False

        if not is_date:
            cat_cols.append(col)
            # Frequency encoding
            freq = combined_df[col].value_counts(dropna=False)
            lgb_features[f"{col}_freq"] = (
                combined_df[col].map(freq).fillna(0).astype(float)
            )
            cb_features[f"{col}_freq"] = lgb_features[f"{col}_freq"]

            # LightGBM categorical type
            lgb_features[col] = combined_df[col].astype("category")
            # CatBoost categorical string handling
            cb_features[col] = combined_df[col].fillna("__MISSING__").astype(str)
    else:
        lgb_features[col] = combined_df[col]
        cb_features[col] = combined_df[col]

X_lgb_train = lgb_features.iloc[:n_train].copy()
X_lgb_test = lgb_features.iloc[n_train:].copy()

X_cb_train = cb_features.iloc[:n_train].copy()
X_cb_test = cb_features.iloc[n_train:].copy()

# 3. 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y))
test_lgb = np.zeros(len(X_lgb_test))

oof_cb = np.zeros(len(y))
test_cb = np.zeros(len(X_cb_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 47,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_lgb_train, y)):
    # --- LightGBM ---
    X_tr_lgb, y_tr = X_lgb_train.iloc[train_idx], y[train_idx]
    X_val_lgb, y_val = X_lgb_train.iloc[val_idx], y[val_idx]

    dtrain = lgb.Dataset(X_tr_lgb, label=y_tr)
    dval = lgb.Dataset(X_val_lgb, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(-1),
    ]

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=2500,
        valid_sets=[dtrain, dval],
        callbacks=callbacks,
    )

    oof_lgb[val_idx] = model_lgb.predict(
        X_val_lgb, num_iteration=model_lgb.best_iteration
    )
    test_lgb += (
        model_lgb.predict(X_lgb_test, num_iteration=model_lgb.best_iteration)
        / kf.n_splits
    )

    # --- CatBoost ---
    if HAS_CATBOOST:
        X_tr_cb = X_cb_train.iloc[train_idx]
        X_val_cb = X_cb_train.iloc[val_idx]

        train_pool = Pool(X_tr_cb, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_val_cb, y_val, cat_features=cat_cols)
        test_pool = Pool(X_cb_test, cat_features=cat_cols)

        cb_model = CatBoostRegressor(
            iterations=2500,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42 + fold,
            verbose=False,
            early_stopping_rounds=50,
            thread_count=-1,
        )

        cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof_cb[val_idx] = cb_model.predict(val_pool)
        test_cb += cb_model.predict(test_pool) / kf.n_splits

# Evaluate individual models and blend
rmse_lgb = mean_squared_error(y, oof_lgb, squared=False)
print(f"LGBM 5-Fold CV RMSE: {rmse_lgb:.4f}")

if HAS_CATBOOST:
    rmse_cb = mean_squared_error(y, oof_cb, squared=False)
    print(f"CatBoost 5-Fold CV RMSE: {rmse_cb:.4f}")

    # Optimize blend weight
    best_w = 0.5
    best_rmse = float("inf")
    for w in np.linspace(0.0, 1.0, 101):
        blend_oof = w * oof_lgb + (1 - w) * oof_cb
        score = mean_squared_error(y, blend_oof, squared=False)
        if score < best_rmse:
            best_rmse = score
            best_w = w

    print(f"Best Ensemble Weight (LGBM): {best_w:.2f}")
    oof_final = best_w * oof_lgb + (1 - best_w) * oof_cb
    test_final = best_w * test_lgb + (1 - best_w) * test_cb
else:
    oof_final = oof_lgb
    test_final = test_lgb

cv_rmse = mean_squared_error(y, oof_final, squared=False)
print(f"Ensemble 5-Fold CV RMSE: {cv_rmse:.4f}")

# 4. Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_final)), target_col: test_final})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
