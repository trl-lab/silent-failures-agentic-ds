import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "projectPayment"
y = train[target_col].values.astype(np.float64)
X = train.drop(columns=[target_col]).copy()
X_test = test.copy()

n_train = len(X)
df_all = pd.concat([X, X_test], axis=0, ignore_index=True)

# Process date columns if any exist
for col in list(df_all.columns):
    if "date" in col.lower() or df_all[col].dtype == "object":
        try:
            parsed = pd.to_datetime(df_all[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(df_all):
                df_all[f"{col}_year"] = parsed.dt.year.fillna(-1).astype(int)
                df_all[f"{col}_month"] = parsed.dt.month.fillna(-1).astype(int)
                df_all[f"{col}_day"] = parsed.dt.day.fillna(-1).astype(int)
                df_all[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(int)
                df_all[f"{col}_quarter"] = parsed.dt.quarter.fillna(-1).astype(int)
        except Exception:
            pass

# Identify categorical and numerical columns
cat_cols = []
num_cols = []
for col in df_all.columns:
    if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        cat_cols.append(col)
        df_all[col] = df_all[col].fillna("Missing").astype(str)
    else:
        num_cols.append(col)
        df_all[col] = df_all[col].fillna(df_all[col].median())

# Add frequency encoding
for col in cat_cols:
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)

# Separate back to train and test
X_processed = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test_processed = df_all.iloc[n_train:].copy().reset_index(drop=True)

# Out-of-Fold Target Encoding with m-estimate smoothing
kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y.mean()
m_weight = 10.0

for col in cat_cols:
    te_col_name = f"{col}_te"
    X_processed[te_col_name] = np.nan
    te_test_folds = np.zeros(len(X_test_processed))

    for tr_idx, va_idx in kf.split(X_processed):
        tr_cat = X_processed.iloc[tr_idx][col]
        tr_y = y[tr_idx]

        stats = (
            pd.DataFrame({"cat": tr_cat, "target": tr_y})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        smoothed = (stats["count"] * stats["mean"] + m_weight * global_mean) / (
            stats["count"] + m_weight
        )

        val_cat = X_processed.iloc[va_idx][col]
        X_processed.loc[va_idx, te_col_name] = (
            val_cat.map(smoothed).fillna(global_mean).values
        )

        test_cat = X_test_processed[col]
        te_test_folds += test_cat.map(smoothed).fillna(global_mean).values / kf.n_splits

    X_test_processed[te_col_name] = te_test_folds

# Multi-seed Cross-Validation setup
seeds = [42, 2024, 999]
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test_processed))

for seed in seeds:
    seed_oof = np.zeros(n_train)
    seed_test = np.zeros(len(X_test_processed))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed)):
        X_tr, y_tr = X_processed.iloc[train_idx], y[train_idx]
        X_va, y_va = X_processed.iloc[val_idx], y[val_idx]

        model = CatBoostRegressor(
            iterations=2500,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            cat_features=cat_cols if len(cat_cols) > 0 else None,
            random_seed=seed + fold,
            verbose=0,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=150,
            verbose=False,
        )

        seed_oof[val_idx] = model.predict(X_va)
        seed_test += model.predict(X_test_processed) / kf.n_splits

    oof_preds += seed_oof / len(seeds)
    test_preds += seed_test / len(seeds)

oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {oof_rmse:.6f}")

# Save submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "projectPayment": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
