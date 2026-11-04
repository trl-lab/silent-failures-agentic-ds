import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import OrdinalEncoder

    HAS_CATBOOST = False

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Total Contribution (USD)"
y_train = train_df[target_col].values.astype(np.float64)

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

n_train = len(train_df)
df_all = pd.concat([X_train, X_test], ignore_index=True)

# Parse potential date columns
for col in list(df_all.columns):
    if df_all[col].dtype == "object":
        try:
            parsed = pd.to_datetime(df_all[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(df_all):
                df_all[f"{col}_year"] = parsed.dt.year
                df_all[f"{col}_month"] = parsed.dt.month
                df_all[f"{col}_quarter"] = parsed.dt.quarter
                df_all[f"{col}_day"] = parsed.dt.day
                df_all[f"{col}_dayofweek"] = parsed.dt.dayofweek
                df_all[f"{col}_dayofyear"] = parsed.dt.dayofyear
                df_all.drop(columns=[col], inplace=True)
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

# Group-level aggregations
for c_col in cat_cols[:4]:
    if df_all[c_col].nunique() < 500:
        for n_col in num_cols[:4]:
            grouped = df_all.groupby(c_col)[n_col]
            df_all[f"{c_col}_{n_col}_mean"] = grouped.transform("mean")
            df_all[f"{c_col}_{n_col}_std"] = grouped.transform("std")

# Fill remaining NaNs in numeric features
num_cols = [c for c in df_all.columns if c not in cat_cols]
for col in num_cols:
    df_all[col] = df_all[col].fillna(df_all[col].median())

X_train_proc = df_all.iloc[:n_train].copy()
X_test_proc = df_all.iloc[n_train:].copy()

# Log1p target transformation
y_train_log = np.log1p(np.maximum(0, y_train))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_log = np.zeros(n_train)
test_preds_log = np.zeros(len(X_test_proc))

if HAS_CATBOOST:
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train_log)):
        X_tr, y_tr = X_train_proc.iloc[train_idx], y_train_log[train_idx]
        X_va, y_va = X_train_proc.iloc[val_idx], y_train_log[val_idx]

        model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.05,
            depth=6,
            cat_features=cat_cols,
            eval_metric="RMSE",
            random_seed=42 + fold,
            verbose=0,
        )
        model.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, verbose=False
        )

        oof_preds_log[val_idx] = model.predict(X_va)
        test_preds_log += model.predict(X_test_proc) / kf.n_splits
else:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_all_encoded = df_all.copy()
    if cat_cols:
        df_all_encoded[cat_cols] = oe.fit_transform(df_all_encoded[cat_cols])
    X_tr_enc = df_all_encoded.iloc[:n_train]
    X_te_enc = df_all_encoded.iloc[n_train:]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr_enc, y_train_log)):
        X_tr, y_tr = X_tr_enc.iloc[train_idx], y_train_log[train_idx]
        X_va, y_va = X_tr_enc.iloc[val_idx], y_train_log[val_idx]

        model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, random_state=42 + fold
        )
        model.fit(X_tr, y_tr)

        oof_preds_log[val_idx] = model.predict(X_va)
        test_preds_log += model.predict(X_te_enc) / kf.n_splits

# Inverse transformation back to original scale
oof_preds = np.maximum(0, np.expm1(oof_preds_log))
test_preds = np.maximum(0, np.expm1(test_preds_log))

oof_rmse = root_mean_squared_error(y_train, oof_preds)
print(f"Validation Metric: {oof_rmse:.4f}")

# Save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Total Contribution (USD)": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
