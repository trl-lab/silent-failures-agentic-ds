import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Bus age"

# Handle target and feature separation
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Keep common feature columns between train and test
common_cols = [col for col in X.columns if col in X_test.columns]
X = X[common_cols].copy()
X_test = X_test[common_cols].copy()

# Preprocess datetime and categorical columns
for col in list(X.columns):
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        for df in (X, X_test):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        X.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)
    elif X[col].dtype == "object" or isinstance(X[col].dtype, pd.CategoricalDtype):
        categories = pd.concat([X[col], X_test[col]]).dropna().unique()
        X[col] = pd.Categorical(X[col], categories=categories)
        X_test[col] = pd.Categorical(X_test[col], categories=categories)

# 5-Fold Cross Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Compute OOF RMSE metric
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {oof_rmse:.5f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Bus age": test_preds})
submission.to_csv("./working/submission.csv", index=False)
