import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "ACA FUL"
features = [col for col in train_df.columns if col != target_col]

# Preprocessing features
combined = pd.concat([train_df[features], test_df[features]], axis=0).reset_index(
    drop=True
)

# Process date columns or convert object/category columns
cat_cols = []
for col in features:
    if combined[col].dtype == "object" or str(combined[col].dtype) == "category":
        # Check if it looks like a date
        try:
            converted = pd.to_datetime(combined[col], errors="coerce")
            if converted.notna().mean() > 0.6:
                combined[f"{col}_year"] = converted.dt.year
                combined[f"{col}_month"] = converted.dt.month
                combined[f"{col}_day"] = converted.dt.day
                combined = combined.drop(columns=[col])
                continue
        except Exception:
            pass
        # Categorical factorize
        combined[col], _ = pd.factorize(combined[col].astype(str))
        cat_cols.append(col)

# Convert all remaining columns to float/numeric where possible
for col in combined.columns:
    if col not in cat_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

n_train = len(train_df)
X = combined.iloc[:n_train].reset_index(drop=True)
y = train_df[target_col].values
X_test = combined.iloc[n_train:].reset_index(drop=True)

# 5-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.5f}")

# Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_df)), "ACA FUL": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
