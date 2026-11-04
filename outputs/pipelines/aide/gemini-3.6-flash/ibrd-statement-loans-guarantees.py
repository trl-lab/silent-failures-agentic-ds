import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Original Principal Amount (US$)"
feature_cols = [c for c in test_df.columns if c in train_df.columns and c != target_col]

# Combine datasets for consistent feature engineering
df_all = pd.concat([train_df[feature_cols], test_df[feature_cols]], ignore_index=True)
n_train = len(train_df)

# Parse date columns and derive elapsed duration metrics
date_cols = [c for c in feature_cols if "date" in c.lower() or "period" in c.lower()]
parsed_date_cols = []
for c in date_cols:
    try:
        dt_s = pd.to_datetime(df_all[c], errors="coerce")
        if dt_s.notna().sum() > 0:
            df_all[f"{c}_year"] = dt_s.dt.year
            df_all[f"{c}_month"] = dt_s.dt.month
            df_all[f"{c}_day"] = dt_s.dt.day
            df_all[f"{c}_dayofweek"] = dt_s.dt.dayofweek
            df_all[f"{c}_quarter"] = dt_s.dt.quarter
            df_all[f"{c}_timestamp"] = dt_s.astype("int64") // 10**9
            parsed_date_cols.append(c)
    except Exception:
        pass

if len(parsed_date_cols) >= 2:
    for i in range(len(parsed_date_cols)):
        for j in range(i + 1, len(parsed_date_cols)):
            c1, c2 = parsed_date_cols[i], parsed_date_cols[j]
            dt1 = pd.to_datetime(df_all[c1], errors="coerce")
            dt2 = pd.to_datetime(df_all[c2], errors="coerce")
            df_all[f"{c1}_diff_{c2}"] = (dt1 - dt2).dt.days

# Encode categorical variables and add frequency encoding
cat_cols = [
    c
    for c in feature_cols
    if df_all[c].dtype == "object" or df_all[c].dtype.name == "category"
]
for c in cat_cols:
    freq = df_all[c].value_counts(normalize=True).to_dict()
    df_all[f"{c}_freq"] = df_all[c].map(freq).fillna(0)
    df_all[c] = df_all[c].astype("category").cat.codes

# Financial column interaction features
fin_cols = [
    c
    for c in df_all.columns
    if (
        "amount" in c.lower()
        or "disbursed" in c.lower()
        or "repaid" in c.lower()
        or "due" in c.lower()
        or "cancelled" in c.lower()
    )
    and df_all[c].dtype in ["int64", "float64", "float32", "int32"]
]
if len(fin_cols) >= 2:
    df_all["fin_sum"] = df_all[fin_cols].sum(axis=1)
    df_all["fin_std"] = df_all[fin_cols].std(axis=1)
    df_all["fin_max"] = df_all[fin_cols].max(axis=1)
    df_all["fin_min"] = df_all[fin_cols].min(axis=1)

# Drop original parsed date strings
cols_to_drop = [c for c in parsed_date_cols if c in df_all.columns]
df_all = df_all.drop(columns=cols_to_drop)

X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

y = train_df[target_col].values
y_log = np.log1p(np.maximum(0, y))

# 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_log[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_log[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=50,
        enable_categorical=True,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_pred_log = model.predict(X_val)
    oof_preds[val_idx] = np.expm1(val_pred_log)

    test_pred_log = model.predict(X_test)
    test_preds += np.expm1(test_pred_log) / kf.n_splits

oof_preds = np.maximum(0, oof_preds)
test_preds = np.maximum(0, test_preds)

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {cv_rmse:.4f}")

# Save final predictions to submission.csv
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Original Principal Amount (US$)": test_preds}
)
sub.to_csv("./working/submission.csv", index=False)
