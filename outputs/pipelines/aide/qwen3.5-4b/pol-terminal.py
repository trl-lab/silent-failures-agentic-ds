import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and features
target = "CAPACITY"
feature_cols = [col for col in train_df.columns if col != target]

# Handle missing values
train_df = train_df.dropna(subset=feature_cols + [target])
test_df = test_df.dropna(subset=feature_cols)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])
y_train = train_df[target]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_rmse = np.inf
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Train LightGBM
    lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
    lgb_eval = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Evaluate on validation fold
    predictions_fold = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, predictions_fold)

    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    if fold_rmse < best_rmse:
        best_rmse = fold_rmse
        best_model = model

# Final training on full dataset
lgb_train_final = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(params, lgb_train_final, num_boost_round=1000)

# Generate predictions for test set
test_predictions = final_model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test_df.index, "CAPACITY": test_predictions})
submission.to_csv("./working/submission.csv", index=False)

print(f"Final RMSE on validation: {best_rmse:.4f}")
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and features
target = "CAPACITY"
feature_cols = [col for col in train_df.columns if col != target]

# Handle missing values
train_df = train_df.dropna(subset=feature_cols + [target])
test_df = test_df.dropna(subset=feature_cols)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])
y_train = train_df[target]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_rmse = np.inf
best_model = None
best_params = {}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    # Train LightGBM
    lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold, feature_name=feature_cols)
    lgb_eval = lgb.Dataset(
        X_val_fold, label=y_val_fold, feature_name=feature_cols, reference=lgb_train
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    # Evaluate on validation fold
    predictions_fold = model.predict(X_val_fold)
    fold_rmse = mean_squared_error(y_val_fold, predictions_fold)

    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    if fold_rmse < best_rmse:
        best_rmse = fold_rmse
        best_model = model
        best_params = params

# Final training on full dataset
lgb_train_final = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
final_model = lgb.train(
    best_params,
    lgb_train_final,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Generate predictions for test set
test_predictions = final_model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test_df.index, "CAPACITY": test_predictions})
submission.to_csv("./working/submission.csv", index=False)

print(f"Final RMSE on validation: {best_rmse:.4f}")
print("Submission saved to ./working/submission.csv")
