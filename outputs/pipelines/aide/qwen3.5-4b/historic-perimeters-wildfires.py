import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Check target column
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {train.columns.tolist()}")

# Basic data preprocessing
# Handle potential missing values
train = train.dropna()
test = test.dropna()

# Define feature columns (exclude id and target)
train_features = train.drop(["id", "gisacres"], axis=1)
test_features = test.drop(["id"], axis=1)

# Initialize LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train, X_val = train_features.iloc[train_idx], train_features.iloc[val_idx]
    y_train, y_val = train["gisacres"].iloc[train_idx], train["gisacres"].iloc[val_idx]

    # Train model with early stopping
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks, verbose=-1
    )

    # Predict on validation set
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    cv_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

# Calculate mean RMSE
mean_cv_rmse = np.mean(cv_scores)
print(f"\nMean CV RMSE: {mean_cv_rmse:.4f}")

# Train final model on all training data
model.fit(
    train_features,
    train["gisacres"],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Predict on test set
test_predictions = model.predict(test_features)

# Create submission file
submission = pd.DataFrame({"id": test["id"], "gisacres": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
print(f"Test predictions shape: {test_predictions.shape}")
print(f"Sample predictions: {test_predictions[:5].tolist()}")
