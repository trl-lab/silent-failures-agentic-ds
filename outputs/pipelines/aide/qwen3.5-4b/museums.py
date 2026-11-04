import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and feature columns
target_col = "REVENUE15"
feature_cols = [col for col in train_df.columns if col != target_col]

# Combine train and test for id handling
train_df["id"] = train_df.index
test_df["id"] = test_df.index

# Basic data cleaning - handle missing values
train_df = train_df.fillna(train_df.median(numeric_only=True))
test_df = test_df.fillna(test_df.median(numeric_only=True))

# Create feature matrix and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    # Prepare data for LightGBM
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(
        X_val_fold,
        label=(
            X_val_fold[target_col]
            if target_col in X_val_fold.columns
            else np.zeros(len(X_val_fold))
        ),
    )

    # Train model with early stopping
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        verbose_eval=-1,
    )

    # Predict on validation
    y_pred = model.predict(val_data)
    rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
    scores.append(rmse)
    print(f"Fold {fold_idx + 1} RMSE: {rmse:.4f}")

# Calculate CV RMSE
cv_rmse = np.sqrt(np.mean([s**2 for s in scores]))
print(f"\n5-Fold CV RMSE: {cv_rmse:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=500,
    callbacks=[lgb.early_stopping(50, verbose=False)],
    verbose_eval=-1,
)

# Make predictions on test data
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": test_df["id"], "REVENUE15": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission file saved to ./working/submission.csv")
print(f"Final CV RMSE: {cv_rmse:.4f}")
