import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Identify features and target
target = "LOANTOASSET"
feature_cols = [col for col in train.columns if col != target]

# Combine train and test for feature extraction
train_features = train[feature_cols]
test_features = test[feature_cols]

# Handle missing values
train_features = train_features.fillna(train_features.mean())
test_features = test_features.fillna(train_features.mean())

# Scale features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

# Define training data
train_data = lgb.Dataset(train_scaled, label=train[target])
test_data = lgb.Dataset(test_scaled, reference=train_data)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = np.zeros(len(train))
best_score = float("inf")
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(train_scaled)):
    X_train, X_val = train_scaled[train_idx], train_scaled[val_idx]
    y_train = train[target].iloc[train_idx]

    train_data_fold = lgb.Dataset(X_train, label=y_train)
    test_data_fold = lgb.Dataset(X_val, reference=train_data_fold)

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
        train_data_fold,
        num_boost_round=500,
        valid_sets=[test_data_fold],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_val)
    val_rmse = mean_squared_error(y_train[val_idx], val_pred)
    predictions[val_idx] = val_pred

    if val_rmse < best_score:
        best_score = val_rmse
        best_model = model

# Final evaluation on training set
final_predictions = best_model.predict(train_scaled)
final_rmse = mean_squared_error(train[target], final_predictions)
print(f"Final RMSE on training set: {final_rmse:.4f}")

# Make predictions on test data
test_predictions = best_model.predict(test_scaled)

# Create submission file
submission = pd.DataFrame({"id": range(len(test)), "LOANTOASSET": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
