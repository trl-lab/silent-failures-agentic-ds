import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Load datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Amount in USD"
y = train_df[target_col].values

# Drop target from train
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Combine for uniform feature engineering
combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
n_train = len(X_train)

# Feature engineering
for col in combined.columns:
    # Handle datetime features
    if (
        pd.api.types.is_datetime64_any_dtype(combined[col])
        or "date" in col.lower()
        or "time" in col.lower()
    ):
        try:
            dt_col = pd.to_datetime(combined[col], errors="coerce")
            if dt_col.notna().sum() > 0:
                combined[f"{col}_year"] = dt_col.dt.year.fillna(-1).astype(int)
                combined[f"{col}_month"] = dt_col.dt.month.fillna(-1).astype(int)
                combined[f"{col}_day"] = dt_col.dt.day.fillna(-1).astype(int)
                combined[f"{col}_dayofweek"] = dt_col.dt.dayofweek.fillna(-1).astype(
                    int
                )
                combined[f"{col}_quarter"] = dt_col.dt.quarter.fillna(-1).astype(int)
                combined.drop(columns=[col], inplace=True)
                continue
        except Exception:
            pass

    # Handle object / category features
    if combined[col].dtype == "object" or pd.api.types.is_categorical_dtype(
        combined[col]
    ):
        freq = combined[col].value_counts(dropna=False).to_dict()
        combined[f"{col}_freq"] = combined[col].map(freq).fillna(0)
        combined[col] = combined[col].astype(str).astype("category")

# Separate back into train and test
X = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

# 5-fold cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

categorical_features = [
    col for col in X.columns if pd.api.types.is_categorical_dtype(X[col])
]

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": 1000,
        "random_state": 42 + fold,
        "verbose": -1,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=callbacks,
        categorical_feature=(
            categorical_features if len(categorical_features) > 0 else "auto"
        ),
    )

    val_pred = model.predict(X_va)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / n_splits

# Calculate out-of-fold RMSE
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Prepare submission dataframe
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Amount in USD": test_preds}
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
