import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "APR"
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

os.makedirs("./working", exist_ok=True)

# 2. Feature Engineering
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# Process date columns
date_cols = []
for col in combined.columns:
    if np.issubdtype(combined[col].dtype, np.datetime64):
        date_cols.append(col)
    elif combined[col].dtype == object:
        sample = combined[col].dropna().astype(str).head(20)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample)
                date_cols.append(col)
            except (ValueError, TypeError):
                pass

for col in date_cols:
    dt_series = pd.to_datetime(combined[col], errors="coerce")
    combined[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(int)
    combined[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(int)
    combined[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(int)
    combined[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(int)
    combined[f"{col}_dayofyear"] = dt_series.dt.dayofyear.fillna(-1).astype(int)
    combined[f"{col}_timestamp"] = dt_series.astype("int64") // 10**9
    combined.drop(columns=[col], inplace=True)

# Categorical and numerical column separation
cat_cols = []
num_cols = []
for col in combined.columns:
    if (
        combined[col].dtype == object
        or combined[col].dtype.name == "category"
        or combined[col].dtype == bool
    ):
        cat_cols.append(col)
        combined[col] = combined[col].astype(str).fillna("missing")
    else:
        num_cols.append(col)

# Frequency encoding for categoricals
for col in cat_cols:
    freq_map = combined[col].value_counts(normalize=True).to_dict()
    combined[f"{col}_freq"] = combined[col].map(freq_map).astype(float)

# Separate back into train and test
n_train = len(X_train)
X_train_proc = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test_proc = combined.iloc[n_train:].copy().reset_index(drop=True)

# 3. Model Training with 5-Fold Cross Validation & Out-Of-Fold Target Encoding
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cat_oof_preds = np.zeros(n_train)
cat_test_preds = np.zeros(len(X_test_proc))

lgb_oof_preds = np.zeros(n_train)
lgb_test_preds = np.zeros(len(X_test_proc))

global_mean = np.mean(y)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr = X_train_proc.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X_train_proc.iloc[val_idx].copy()
    y_val = y[val_idx]
    X_te = X_test_proc.copy()

    # Out-of-fold smoothed target encoding for categorical columns
    for c in cat_cols:
        tr_df = pd.DataFrame({c: X_tr[c], "target": y_tr})
        stats = tr_df.groupby(c)["target"].agg(["count", "mean"])
        smoothing = 10.0
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        te_map = smoothed.to_dict()

        te_col_name = f"{c}_target_enc"
        X_tr[te_col_name] = X_tr[c].map(te_map).fillna(global_mean).astype(float)
        X_va[te_col_name] = X_va[c].map(te_map).fillna(global_mean).astype(float)
        X_te[te_col_name] = X_te[c].map(te_map).fillna(global_mean).astype(float)

    # CatBoost model training
    cat_feature_indices = [
        X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns
    ]
    cb_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=False,
        cat_features=cat_feature_indices,
    )
    cb_model.fit(
        X_tr, y_tr, eval_set=(X_va, y_val), early_stopping_rounds=50, verbose=False
    )
    cat_oof_preds[val_idx] = cb_model.predict(X_va)
    cat_test_preds += cb_model.predict(X_te) / kf.n_splits

    # LightGBM model training (convert categoricals to category dtype)
    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()
    X_te_lgb = X_te.copy()
    for c in cat_cols:
        X_tr_lgb[c] = X_tr_lgb[c].astype("category")
        X_va_lgb[c] = X_va_lgb[c].astype("category")
        X_te_lgb[c] = X_te_lgb[c].astype("category")

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "random_state": 42 + fold,
        "verbosity": -1,
        "n_jobs": -1,
    }

    trn_data = lgb.Dataset(X_tr_lgb, label=y_tr)
    val_data = lgb.Dataset(X_va_lgb, label=y_val, reference=trn_data)

    lgb_model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    lgb_oof_preds[val_idx] = lgb_model.predict(
        X_va_lgb, num_iteration=lgb_model.best_iteration
    )
    lgb_test_preds += (
        lgb_model.predict(X_te_lgb, num_iteration=lgb_model.best_iteration)
        / kf.n_splits
    )

# 4. Optimal Weighted Blending
best_weight = 0.5
best_rmse = float("inf")
for w in np.linspace(0.0, 1.0, 101):
    blend_oof = w * cat_oof_preds + (1 - w) * lgb_oof_preds
    rmse = root_mean_squared_error(y, blend_oof)
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w

final_oof_preds = best_weight * cat_oof_preds + (1 - best_weight) * lgb_oof_preds
final_test_preds = best_weight * cat_test_preds + (1 - best_weight) * lgb_test_preds

cb_rmse = root_mean_squared_error(y, cat_oof_preds)
lgb_rmse = root_mean_squared_error(y, lgb_oof_preds)
print(f"CatBoost OOF RMSE: {cb_rmse:.5f}")
print(f"LightGBM OOF RMSE: {lgb_rmse:.5f}")
print(
    f"Ensemble (CatBoost weight {best_weight:.2f}) 5-Fold CV Out-of-Fold RMSE: {best_rmse:.5f}"
)

# 5. Save submission
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "APR": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
