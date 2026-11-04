import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Cash Disbursements (US$)"
y = train_df[target_col].values

# Save test row indices or construct id column
if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

# Prepare features
df_all = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)

# Drop 'id' from features if present
if "id" in df_all.columns:
    df_all = df_all.drop(columns=["id"])

# Process datetime features
datetime_cols = []
for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        datetime_cols.append(col)
    elif df_all[col].dtype == "object":
        try:
            sample_non_null = df_all[col].dropna().iloc[:50]
            if len(sample_non_null) > 0:
                parsed = pd.to_datetime(sample_non_null, errors="coerce")
                if parsed.notnull().mean() > 0.8:
                    df_all[col] = pd.to_datetime(df_all[col], errors="coerce")
                    datetime_cols.append(col)
        except Exception:
            pass

for col in datetime_cols:
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all[f"{col}_quarter"] = df_all[col].dt.quarter
    df_all[f"{col}_dayofyear"] = df_all[col].dt.dayofyear
    df_all.drop(columns=[col], inplace=True)

# Categorical and Numerical identification
cat_cols = []
num_cols = []
for col in df_all.columns:
    if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Fill missing and frequency encode categoricals
for col in cat_cols:
    df_all[col] = df_all[col].fillna("Missing").astype(str)
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)

n_train = len(train_df)
X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

# Add KFold out-of-fold target encoding for categoricals
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    te_tr = np.full(n_train, np.nan)
    te_te = np.zeros(len(X_test))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        group_means = (
            pd.DataFrame({"cat": X_train.iloc[train_idx][col], "target": y[train_idx]})
            .groupby("cat")["target"]
            .mean()
        )
        te_tr[val_idx] = X_train.iloc[val_idx][col].map(group_means).values
        te_te += (
            X_test[col].map(group_means).fillna(y[train_idx].mean()).values
            / kf.n_splits
        )

    global_mean = y.mean()
    te_tr = np.nan_to_num(te_tr, nan=global_mean)
    te_te = np.nan_to_num(te_te, nan=global_mean)
    X_train[f"{col}_te"] = te_tr
    X_test[f"{col}_te"] = te_te

# 5-Fold Cross Validation with CatBoostRegressor
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
        cat_features=cat_cols,
    )

    model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

# Post-processing: Non-negative clipping if target is non-negative
if np.min(y) >= 0:
    oof_preds = np.clip(oof_preds, 0, None)
    test_preds = np.clip(test_preds, 0, None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {cv_rmse:.4f}")

# Save Submission
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": test_ids, "Cash Disbursements (US$)": test_preds})
sub.to_csv("./working/submission.csv", index=False)
