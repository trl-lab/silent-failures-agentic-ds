import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "BEDS"
features = [c for c in train_df.columns if c != target_col]

# Handle categorical columns
cat_cols = []
for col in features:
    if train_df[col].dtype == "object" or pd.api.types.is_string_dtype(train_df[col]):
        train_df[col] = train_df[col].astype(str).astype("category")
        test_df[col] = test_df[col].astype(str).astype("category")
        cat_cols.append(col)
    elif pd.api.types.is_categorical_dtype(train_df[col]):
        cat_cols.append(col)

X = train_df[features]
y = train_df[target_col].values
X_test = test_df[features]

# Setup 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 1500,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test) / kf.n_splits

# Clip negative predictions if hospital beds cannot be negative
oof_preds = np.clip(oof_preds, 0, None)
test_preds = np.clip(test_preds, 0, None)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Prepare submission
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "BEDS": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
