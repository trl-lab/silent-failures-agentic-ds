import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Target column
target = "y"
feature_cols = [col for col in train_df.columns if col != target]

# Extract features and target
X_train = train_df[feature_cols].values
y_train = train_df[target].values
X_test = test_df[feature_cols].values
test_ids = test_df.index.values

# Create validation split for evaluation
X_train_val = X_train[:500]
y_train_val = y_train[:500]
X_train = X_train[500:]
y_train = y_train[500:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_val_scaled = scaler.transform(X_train_val)
X_test_scaled = scaler.transform(X_test)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 8,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_fold_train = X_train_scaled[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_val = y_train[val_idx]

    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_fold_val)
    rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
    rmse_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.6f}")

# Average RMSE from cross-validation
avg_rmse = np.mean(rmse_scores)
print(f"\nAverage RMSE (5-fold CV): {avg_rmse:.6f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train_scaled, label=y_train)
val_data = lgb.Dataset(X_train_val_scaled, label=y_train_val, reference=train_data)

final_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions for test data
y_test_pred = final_model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({"id": test_ids, "y": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
