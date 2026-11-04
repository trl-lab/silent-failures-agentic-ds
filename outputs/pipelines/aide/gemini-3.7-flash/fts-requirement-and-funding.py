import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "funding"
y = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

# Combine for consistent feature engineering
ntrain = len(train_df)
ntest = len(test_df)
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Identify Column Types
date_cols = []
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        date_cols.append(col)
    elif df[col].dtype == "object":
        try:
            converted = pd.to_datetime(df[col], errors="raise")
            df[col] = converted
            date_cols.append(col)
        except (ValueError, TypeError):
            pass

# Feature engineering on Date columns
for col in date_cols:
    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    df[f"{col}_dayofyear"] = df[col].dt.dayofyear
    df[f"{col}_quarter"] = df[col].dt.quarter
    df[f"{col}_is_month_start"] = df[col].dt.is_month_start.astype(int)
    df[f"{col}_is_month_end"] = df[col].dt.is_month_end.astype(int)

# Differences between date columns if multiple exist
if len(date_cols) >= 2:
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1, col2 = date_cols[i], date_cols[j]
            df[f"diff_days_{col1}_{col2}"] = (
                df[col1] - df[col2]
            ).dt.total_seconds() / (24 * 3600)

df = df.drop(columns=date_cols)

# Identify categorical and numerical columns
cat_cols = []
num_cols = []
for col in df.columns:
    if df[col].dtype == "object" or df[col].dtype.name == "category":
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Categorical Frequency Encoding
for col in cat_cols:
    freq = df[col].value_counts(dropna=False)
    df[f"{col}_freq"] = df[col].map(freq)

# Categorical Group Aggregations on Numerical Features
if len(cat_cols) > 0 and len(num_cols) > 0:
    for c_col in cat_cols[:5]:  # Limit to top categorical columns for efficiency
        for n_col in num_cols[:5]:
            grouped = df.groupby(c_col)[n_col]
            agg_df = grouped.agg(
                mean_val="mean", std_val="std", min_val="min", max_val="max"
            ).reset_index()
            agg_df.columns = [
                c_col,
                f"{c_col}_{n_col}_mean",
                f"{c_col}_{n_col}_std",
                f"{c_col}_{n_col}_min",
                f"{c_col}_{n_col}_max",
            ]
            df = df.merge(agg_df, on=c_col, how="left")

# Label Encoding for remaining categorical columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Handle infinite or NaN values in numeric columns
df = df.replace([np.inf, -np.inf], np.nan)

# Split back to train and test
X = df.iloc[:ntrain].copy().reset_index(drop=True)
X_test = df.iloc[ntrain:].copy().reset_index(drop=True)

# 3. Model Training with 5-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(ntrain)
test_preds = np.zeros(ntest)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / n_splits

    fold_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

# 4. Evaluation and Submission
overall_rmse = root_mean_squared_error(y, oof_preds)
print(f"Overall OOF RMSE: {overall_rmse:.4f}")

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "funding": test_preds})
sub.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
