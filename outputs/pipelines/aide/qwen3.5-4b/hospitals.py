import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
X_train = train_df.drop("BEDS", axis=1)
y_train = train_df["BEDS"]
X_test = test_df

# Identify column types and handle missing values
for col in X_train.columns:
    if X_train[col].isna().any():
        X_train[col] = X_train[col].fillna(X_train[col].median())

# Target scaling
y_train_scaled = (y_train - y_train.mean()) / (y_train.std() + 1e-8)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# LightGBM model setup
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

# 5-fold cross-validation for validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
oof_preds = np.zeros(len(y_train))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
    y_tr, y_val = y_train_scaled.iloc[train_idx], y_train_scaled.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
    model.fit(train_data, valid_sets=[val_data], callbacks=callbacks, verbose=0)

    oof_preds[val_idx] = model.predict(X_val)
    cv_scores.append(mean_squared_error(y_val, oof_preds[val_idx]))

# Final model training on all data
final_train_data = lgb.Dataset(X_train_scaled, label=y_train_scaled)
model.fit(
    final_train_data,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
    verbose=0,
)

# Calculate CV RMSE
cv_rmse = np.sqrt(np.mean(cv_scores))
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Generate predictions for test set
test_preds_scaled = model.predict(X_test_scaled)
test_preds = test_preds_scaled * y_train.std() + y_train.mean()

# Save predictions
submission = pd.DataFrame({"id": range(len(test_df)), "BEDS": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
