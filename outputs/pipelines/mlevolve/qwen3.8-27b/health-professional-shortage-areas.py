import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Create output directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Train dtypes:\n{train.dtypes.to_string()}")
print(
    f"Missing values in train:\n{train.isnull().sum()[train.isnull().sum() > 0].to_string()}"
)
print(f"Target stats:\n{train['HPSA Score'].describe()}")

# Identify target
target_col = "HPSA Score"

# Get test ids
if "id" in test.columns:
    test_ids = test["id"].values.astype(int)
else:
    test_ids = np.arange(len(test))

# Separate features
y = train[target_col].values

# Identify and remove id column from features
id_col = None
for col in ["id", "Id", "ID"]:
    if col in train.columns:
        id_col = col
        break

train_features = train.drop(columns=[target_col])
if id_col and id_col in train_features.columns:
    train_features = train_features.drop(columns=[id_col])

test_features = test.copy()
if id_col and id_col in test_features.columns:
    test_features = test_features.drop(columns=[id_col])

# Align columns between train and test
for col in train_features.columns:
    if col not in test_features.columns:
        test_features[col] = np.nan
for col in test_features.columns:
    if col not in train_features.columns:
        train_features[col] = np.nan

feature_cols = list(train_features.columns)
print(f"Number of features: {len(feature_cols)}")

# Identify categorical columns (object or category dtype)
categorical_cols = []
for col in feature_cols:
    if (
        train_features[col].dtype == "object"
        or train_features[col].dtype.name == "category"
    ):
        categorical_cols.append(col)

print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# Convert categoricals to category dtype for LightGBM
for col in categorical_cols:
    train_features[col] = train_features[col].astype("category")
    test_features[col] = test_features[col].astype("category")
    # Ensure consistent categories between train and test
    all_cats = train_features[col].cat.categories
    test_features[col] = test_features[col].cat.set_categories(all_cats)

# Validation split (15% hold-out)
X_train, X_val, y_train, y_val = train_test_split(
    train_features, y, test_size=0.15, random_state=42
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
print(
    f"Target - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}, Range: [{y_train.min():.4f}, {y_train.max():.4f}]"
)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_estimators": 5000,
    "random_state": 42,
    "num_threads": -1,
}

# Train with early stopping on validation RMSE
model = lgb.LGBMRegressor(**params)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    categorical_feature=categorical_cols,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)

print(f"Best iteration: {model.best_iteration_}")

# Validation prediction and RMSE (official metric)
val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.6f}")

# Feature importance (top 10)
importance = pd.DataFrame(
    {"feature": feature_cols, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
print(f"Top 10 features:\n{importance.head(10).to_string()}")

# Retrain on full training data using best iteration from validation
best_iter = model.best_iteration_
params_full = params.copy()
params_full["n_estimators"] = best_iter

model_full = lgb.LGBMRegressor(**params_full)
model_full.fit(
    train_features,
    y,
    categorical_feature=categorical_cols,
    callbacks=[lgb.log_evaluation(200)],
)

# Test prediction using the full-data model
test_pred = model_full.predict(test_features)

# Create submission with required format: id (int) and HPSA Score
submission = pd.DataFrame({"id": test_ids, "HPSA Score": test_pred})
submission["id"] = submission["id"].astype(int)

# Save submission
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved with {len(submission)} rows")
print(f"Submission head:\n{submission.head()}")
print(f"Prediction stats - Mean: {test_pred.mean():.4f}, Std: {test_pred.std():.4f}")
print(f"Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
print(f"Final Validation Score: {val_rmse}")
