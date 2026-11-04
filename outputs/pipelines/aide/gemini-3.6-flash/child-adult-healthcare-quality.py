import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "State Rate"

# Clean target column and remove missing target rows
train_df[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

y_train = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

if target_col in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[target_col])

n_train = len(X_train_raw)
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# 2. Preprocessing & Data Cleaning
for col in df_all.columns:
    if df_all[col].dtype == "object":
        cleaned_series = (
            df_all[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        converted_numeric = pd.to_numeric(cleaned_series, errors="coerce")
        if converted_numeric.notna().sum() > 0.5 * len(df_all):
            df_all[col] = converted_numeric
            continue

        if any(keyword in col.lower() for keyword in ["date", "year"]):
            parsed_dates = pd.to_datetime(df_all[col], errors="coerce")
            if parsed_dates.notna().sum() > 0.5 * len(df_all):
                df_all[f"{col}_year"] = parsed_dates.dt.year
                df_all[f"{col}_month"] = parsed_dates.dt.month
                df_all[f"{col}_day"] = parsed_dates.dt.day

# Convert non-numeric columns to pandas categorical type
cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna("Missing").astype("category")

X_train = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

# 3. Model Training & 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / kf.n_splits

# 4. Metric Computation
cv_rmse = root_mean_squared_error(y_train, oof_preds)
print(f"Validation Metric: {cv_rmse:.5f}")

# 5. Output Predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "State Rate": test_preds})
submission.to_csv("./working/submission.csv", index=False)
