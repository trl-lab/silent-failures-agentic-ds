import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os

# Set working directory
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify feature columns (exclude ID and target)
train_features = train_df.columns.drop(["ID", "y"]).tolist()
test_features = test_df.columns.drop(["ID"]).tolist()

# Ensure consistent feature columns
common_features = set(train_features) & set(test_features)
train_features = list(common_features)
test_features = list(common_features)

# Prepare training data
X_train = train_df[train_features].copy()
y_train = train_df["y"].values

# Handle missing values
X_train = X_train.fillna(X_train.mean())
X_test = test_df[test_features].copy()
X_test = X_test.fillna(X_test.mean())

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions from each fold
fold_predictions = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train[val_idx]

    # Prepare LightGBM data
    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, feature_name=train_features
    )
    val_data = lgb.Dataset(
        X_val_fold, label=y_val_fold, feature_name=train_features, reference=train_data
    )

    # Train model with callbacks
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100, verbose=False),
        ],
    )

    # Predict on validation fold
    val_pred = model.predict(X_val_fold)
    fold_predictions.append(val_pred)

    # Calculate R2 for this fold
    val_r2 = r2_score(y_val_fold, val_pred)
    print(f"Fold {fold_idx + 1} R2: {val_r2:.4f}")

# Average R2 across folds
avg_r2 = np.mean(
    [
        r2_score(y_train[val_idx], fold_predictions[i])
        for i, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled))
    ]
)
print(f"Average Validation R2: {avg_r2:.4f}")

# Train final model on all training data
final_train_data = lgb.Dataset(
    X_train_scaled, label=y_train, feature_name=train_features
)
final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100, verbose=False),
    ],
)

# Predict on test data
test_predictions = final_model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({"id": test_df.index, "y": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
