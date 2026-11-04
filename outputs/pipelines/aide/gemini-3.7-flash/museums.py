import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from catboost import CatBoostRegressor, Pool

# Load datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "REVENUE15"
features = [c for c in train_df.columns if c != target_col]

# Identify column types
cat_cols = []
num_cols = []

for col in features:
    if (
        train_df[col].dtype == "object"
        or str(train_df[col].dtype) == "category"
        or train_df[col].dtype == "bool"
    ):
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Preprocessing: convert categorical columns to string and fill missing values
for col in cat_cols:
    train_df[col] = train_df[col].fillna("__MISSING__").astype(str)
    test_df[col] = test_df[col].fillna("__MISSING__").astype(str)

for col in num_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

# Add frequency encoding for high-cardinality categorical features
for col in cat_cols:
    freq = train_df[col].value_counts(normalize=True).to_dict()
    freq_col_name = f"{col}_freq"
    train_df[freq_col_name] = train_df[col].map(freq).fillna(0).astype(float)
    test_df[freq_col_name] = test_df[col].map(freq).fillna(0).astype(float)
    num_cols.append(freq_col_name)

feature_cols = cat_cols + num_cols

# Cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    X_train, y_train = (
        train_df.iloc[train_idx][feature_cols],
        train_df.iloc[train_idx][target_col],
    )
    X_val, y_val = (
        train_df.iloc[val_idx][feature_cols],
        train_df.iloc[val_idx][target_col],
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)
    test_pool = Pool(test_df[feature_cols], cat_features=cat_cols)

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

    val_pred = model.predict(val_pool)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(test_pool) / n_splits

    fold_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

# Calculate overall out-of-fold RMSE
oof_rmse = root_mean_squared_error(train_df[target_col], oof_preds)
print(f"Overall OOF RMSE: {oof_rmse:.4f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), target_col: test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
