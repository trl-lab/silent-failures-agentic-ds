import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Gross Issued Amount (US$)"
y = train_df[target_col].values.astype(np.float64)
train_features = train_df.drop(columns=[target_col])

# Combine dataset for feature processing
n_train = len(train_features)
df_all = pd.concat([train_features, test_df], ignore_index=True)

# 2. Advanced Datetime Processing (Calendar, Continuous Elapsed Days & Pairwise Differences)
parsed_dates = {}
for col in list(df_all.columns):
    if df_all[col].dtype == "object":
        try:
            parsed = pd.to_datetime(df_all[col], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(df_all):
                df_all[f"{col}_year"] = parsed.dt.year
                df_all[f"{col}_month"] = parsed.dt.month
                df_all[f"{col}_day"] = parsed.dt.day
                df_all[f"{col}_dayofweek"] = parsed.dt.dayofweek
                df_all[f"{col}_quarter"] = parsed.dt.quarter

                min_date = parsed.min()
                df_all[f"{col}_elapsed_days"] = (parsed - min_date).dt.days
                parsed_dates[col] = parsed
                df_all.drop(columns=[col], inplace=True)
        except Exception:
            pass

# Pairwise date differences for duration metrics
date_cols = list(parsed_dates.keys())
for i in range(len(date_cols)):
    for j in range(i + 1, len(date_cols)):
        c1, c2 = date_cols[i], date_cols[j]
        df_all[f"{c1}_diff_{c2}_days"] = (parsed_dates[c1] - parsed_dates[c2]).dt.days

# 3. Categorical Processing (Frequency Encoding + Native Categoricals)
cat_cols = [
    c
    for c in df_all.columns
    if df_all[c].dtype == "object" or df_all[c].dtype.name == "category"
]
for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna("Missing")
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)
    df_all[col] = df_all[col].astype("category")

# Separate processed train and test sets
X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

# Log target transform if distribution is right-skewed
use_log = np.all(y >= 0) and (np.median(y) > 0) and (np.max(y) / np.median(y) > 10)
y_train_trans = np.log1p(y) if use_log else y

# 4. Cross-Validation & LightGBM Training with Subsampling Regularization
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_trans[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train_trans[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    val_pred = model.predict(X_va)
    t_pred = model.predict(X_test)

    if use_log:
        val_pred = np.clip(np.expm1(val_pred), 0, None)
        t_pred = np.clip(np.expm1(t_pred), 0, None)
    else:
        val_pred = np.clip(val_pred, 0, None)
        t_pred = np.clip(t_pred, 0, None)

    oof_preds[val_idx] = val_pred
    test_preds += t_pred / kf.n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# 5. Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Gross Issued Amount (US$)": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
