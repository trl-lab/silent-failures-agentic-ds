import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.optimize import minimize

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "projectPayment"
y = train_df[target_col].values.astype(np.float64)
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# 2. Feature Engineering
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

date_cols = []
for col in combined.columns:
    if (
        pd.api.types.is_datetime64_any_dtype(combined[col])
        or "date" in col.lower()
        or "year" in col.lower()
        or "time" in col.lower()
    ):
        try:
            converted = pd.to_datetime(combined[col], errors="coerce")
            if converted.notnull().sum() > 0.5 * len(combined):
                combined[f"{col}_year"] = converted.dt.year
                combined[f"{col}_month"] = converted.dt.month
                combined[f"{col}_day"] = converted.dt.day
                combined[f"{col}_dayofweek"] = converted.dt.dayofweek
                combined[f"{col}_quarter"] = converted.dt.quarter
                date_cols.append(col)
        except Exception:
            pass

# Identify categorical and numerical columns
cat_cols = []
num_cols = []
for col in combined.columns:
    if col in date_cols:
        continue
    if (
        combined[col].dtype == "object"
        or pd.api.types.is_categorical_dtype(combined[col])
        or combined[col].dtype == "string"
    ):
        cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(combined[col]):
        num_cols.append(col)

# Frequency encoding
for col in cat_cols:
    freq = combined[col].value_counts(dropna=False)
    combined[f"{col}_freq"] = combined[col].map(freq).astype(float)
    combined[col] = combined[col].astype(str).fillna("__MISSING__")

# Group aggregate features
for cat in cat_cols[:4]:
    if 1 < combined[cat].nunique() < 1000:
        for num in num_cols[:4]:
            grp_mean = combined.groupby(cat)[num].transform("mean")
            grp_std = combined.groupby(cat)[num].transform("std").fillna(0)
            combined[f"{cat}_{num}_mean"] = grp_mean
            combined[f"{cat}_{num}_std"] = grp_std

# Drop raw date columns
combined = combined.drop(columns=date_cols, errors="ignore")

# Separate back to train and test
n_train = len(X_train)
X_train_proc = combined.iloc[:n_train].copy().reset_index(drop=True)
X_test_proc = combined.iloc[n_train:].copy().reset_index(drop=True)

# Prepare categorical ordinal encoding for LightGBM
cat_features = [col for col in X_train_proc.columns if col in cat_cols]
for col in cat_features:
    unique_vals = {val: i for i, val in enumerate(combined[col].unique())}
    X_train_proc[f"{col}_code"] = X_train_proc[col].map(unique_vals).astype(int)
    X_test_proc[f"{col}_code"] = X_test_proc[col].map(unique_vals).astype(int)

# 3. Model Training & OOF Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cb_oof = np.zeros(n_train)
cb_test = np.zeros(len(X_test_proc))

lgb_oof = np.zeros(n_train)
lgb_test = np.zeros(len(X_test_proc))

# Compute out-of-fold target encoding to enrich both models
smoothing = 10.0
global_mean = y.mean()

for col in cat_features:
    X_train_proc[f"{col}_te"] = np.nan
    X_test_proc[f"{col}_te"] = 0.0

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    for col in cat_features:
        tr_cat = X_train_proc.iloc[train_idx][col]
        tr_y = y[train_idx]
        stats = (
            pd.DataFrame({"cat": tr_cat, "y": tr_y})
            .groupby("cat")["y"]
            .agg(["count", "mean"])
        )
        smooth_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

        val_cat = X_train_proc.iloc[val_idx][col]
        X_train_proc.loc[val_idx, f"{col}_te"] = (
            val_cat.map(smooth_te).fillna(global_mean).values
        )

        test_cat = X_test_proc[col]
        X_test_proc[f"{col}_te"] += (
            test_cat.map(smooth_te).fillna(global_mean).values / kf.n_splits
        )

lgb_feature_cols = [c for c in X_train_proc.columns if c not in cat_features]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
    X_tr_cb, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_val_cb, y_val = X_train_proc.iloc[val_idx], y[val_idx]

    # 1. CatBoost Model
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cb_model.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_val_cb, y_val),
        cat_features=cat_features if len(cat_features) > 0 else None,
        use_best_model=True,
    )
    cb_oof[val_idx] = cb_model.predict(X_val_cb)
    cb_test += cb_model.predict(X_test_proc) / kf.n_splits

    # 2. LightGBM Model
    X_tr_lgb = X_train_proc.iloc[train_idx][lgb_feature_cols]
    X_val_lgb = X_train_proc.iloc[val_idx][lgb_feature_cols]
    X_test_lgb = X_test_proc[lgb_feature_cols]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_oof[val_idx] = lgb_model.predict(X_val_lgb)
    lgb_test += lgb_model.predict(X_test_lgb) / kf.n_splits


# 4. Optimal Blending
def blend_loss(weights):
    w1, w2 = weights
    pred = w1 * cb_oof + w2 * lgb_oof
    return np.sqrt(mean_squared_error(y, pred))


res = minimize(blend_loss, [0.5, 0.5], bounds=[(0, 1), (0, 1)], method="Nelder-Mead")
w1, w2 = res.x / np.sum(res.x)

oof_final = w1 * cb_oof + w2 * lgb_oof
test_final = w1 * cb_test + w2 * lgb_test

cb_cv_rmse = np.sqrt(mean_squared_error(y, cb_oof))
lgb_cv_rmse = np.sqrt(mean_squared_error(y, lgb_oof))
ensemble_cv_rmse = np.sqrt(mean_squared_error(y, oof_final))

print(f"CatBoost 5-Fold CV RMSE: {cb_cv_rmse:.4f}")
print(f"LightGBM 5-Fold CV RMSE: {lgb_cv_rmse:.4f}")
print(f"Ensemble Optimal Weights: CatBoost={w1:.3f}, LightGBM={w2:.3f}")
print(f"Overall 5-Fold CV RMSE: {ensemble_cv_rmse:.4f}")

# 5. Create Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_final)), target_col: test_final})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
