import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")
print(train.dtypes)
print(train.head())

# Separate target and id
target_col = "Price"
id_col = "id" if "id" in train.columns else None

y = train[target_col].values
X = train.drop(columns=[target_col] + ([id_col] if id_col else []))

X_test = test.drop(columns=[target_col] if target_col in test.columns else [])
if id_col and id_col in test.columns:
    test_ids = test[id_col].values
else:
    test_ids = np.arange(len(test))

# Identify categorical columns (object, category) and numeric columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numeric columns ({len(num_cols)}): {num_cols}")

# Handle any NaN in numeric columns
for col in num_cols:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(X[col].median())

# For categorical columns, convert to category dtype for LightGBM
for col in cat_cols:
    combined = pd.concat(
        [X[col], X_test[col] if col in X_test.columns else pd.Series(dtype=object)],
        axis=0,
    )
    combined = combined.astype("category")
    cats = combined.cat.categories
    X[col] = pd.Categorical(X[col], categories=cats).astype("category")
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col], categories=cats).astype("category")

# Also handle any remaining object columns in test that weren't in train
for col in X_test.columns:
    if col not in X.columns:
        print(f"Warning: column {col} in test but not train, dropping")
        X_test = X_test.drop(columns=[col])

# Align columns
X = X[X_test.columns] if set(X_test.columns).issubset(set(X.columns)) else X
common_cols = [c for c in X_test.columns if c in X.columns]
X = X[common_cols]
X_test = X_test[common_cols]

cat_cols_final = [c for c in cat_cols if c in common_cols]
num_cols_final = [c for c in num_cols if c in common_cols]

print(f"\nFinal features: {len(common_cols)}")
print(f"Categorical: {cat_cols_final}")
print(f"Numeric: {num_cols_final}")

# Create validation split (80/20 random)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LightGBM dataset
dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols_final)
dval = lgb.Dataset(
    X_val, label=y_val, categorical_feature=cat_cols_final, reference=dtrain
)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": 42,
}

# Train with early stopping
model = lgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    valid_sets=[dval],
    valid_names=["validation"],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)

# Evaluate on validation set
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"\nValidation RMSE: {val_rmse:.4f}")
print(f"Best iteration: {model.best_iteration}")

# Feature importance
fi = pd.DataFrame(
    {
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type="gain"),
    }
).sort_values("importance", ascending=False)
print("\nTop features by gain:")
print(fi.head(15).to_string(index=False))

# Retrain on full training data with best iteration count
dtrain_full = lgb.Dataset(X, label=y, categorical_feature=cat_cols_final)
model_full = lgb.train(
    params,
    dtrain_full,
    num_boost_round=model.best_iteration,
    callbacks=[lgb.log_evaluation(-1)],
)

# Predict on test set
test_pred = model_full.predict(X_test, num_iteration=model_full.best_iteration)

# Save submission
submission = pd.DataFrame({"id": np.arange(len(test_pred)), "Price": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print(f"\nSubmission saved to ./working/submission.csv")
print(submission.head())
print(f"\nFinal Validation RMSE: {val_rmse:.4f}")
