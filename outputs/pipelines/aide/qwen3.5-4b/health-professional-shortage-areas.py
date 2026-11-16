import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target and id)
feature_cols = [col for col in train_df.columns if col not in ["HPSA Score", "id"]]

# Create train/validation split for early stopping
train_df_sample = train_df.sample(frac=0.8, random_state=42)
val_df = train_df.iloc[len(train_df_sample) :]

# Prepare data
train_features = train_df_sample[feature_cols].values
train_targets = train_df_sample["HPSA Score"].values
val_features = val_df[feature_cols].values
val_targets = val_df["HPSA Score"].values

# Create LightGBM datasets
train_data = lgb.Dataset(train_features, label=train_targets)
val_data = lgb.Dataset(val_features, label=val_targets, reference=train_data)

# Train model with early stopping
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# 5-fold cross-validation evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train_fold = train_features[train_idx]
    y_train_fold = train_targets[train_idx]
    X_val_fold = train_features[val_idx]
    y_val_fold = train_targets[val_idx]

    train_data_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data_fold = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model_fold = lgb.train(
        params,
        train_data_fold,
        num_boost_round=1000,
        valid_sets=[val_data_fold],
        callbacks=[lgb.early_stopping(50)],
    )

    val_pred = model_fold.predict(X_val_fold)
    cv_rmse.append(mean_squared_error(y_val_fold, val_pred))

# Calculate average RMSE
evaluation_metric = np.sqrt(np.mean(cv_rmse))
print(f"5-Fold Cross-Validation RMSE: {evaluation_metric:.4f}")

# Generate predictions for test data
test_features = test_df[feature_cols].values
test_predictions = model.predict(test_features)

# Create submission file
submission_df = pd.DataFrame(
    {"id": test_df["id"].values, "HPSA Score": test_predictions}
)
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission file saved to ./working/submission.csv")
