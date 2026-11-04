import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column and features
target_col = "Total Contribution (USD)"
feature_cols = [col for col in train_df.columns if col != target_col]

# Handle missing values
train_df = train_df.dropna(subset=feature_cols + [target_col])
test_df = test_df.dropna(subset=feature_cols)

# Create feature matrix and target vector
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]

# Create LightGBM model with proper callbacks
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    verbose=-1,
    random_state=42,
    n_jobs=-1,
)

# Use 5-fold cross-validation with early stopping
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Train model
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=50,
        verbose_eval=False,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Calculate RMSE on validation set
    val_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    scores.append(rmse)
    print(f"Fold {fold_idx + 1} RMSE: {rmse:.4f}")

# Calculate final RMSE
final_rmse = np.mean(scores)
print(f"Final RMSE: {final_rmse:.4f}")

# Save predictions for test set
test_pred = model.predict(X_test)
submission = pd.DataFrame(
    {"id": range(len(test_pred)), "Total Contribution (USD)": test_pred}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column and features
target_col = "Total Contribution (USD)"
feature_cols = [col for col in train_df.columns if col != target_col]

# Handle missing values
train_df = train_df.dropna(subset=feature_cols + [target_col])
test_df = test_df.dropna(subset=feature_cols)

# Create feature matrix and target vector
X_train = train_df[feature_cols]
y_train = train_df[target_col].values.astype(np.float64)
X_test = test_df[feature_cols]

# Create LightGBM model with proper callbacks
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    verbose=-1,
    random_state=42,
    n_jobs=-1,
)

# Use 5-fold cross-validation with early stopping
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train[val_idx]

    # Train model
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=50,
        verbose_eval=False,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Calculate RMSE on validation set
    val_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    fold_scores.append(rmse)
    print(f"Fold {fold_idx + 1} RMSE: {rmse:.4f}")

# Calculate final RMSE
final_rmse = np.mean(fold_scores)
print(f"Final RMSE: {final_rmse:.4f}")

# Save predictions for test set
test_pred = model.predict(X_test)
submission = pd.DataFrame(
    {"id": range(len(test_pred)), "Total Contribution (USD)": test_pred}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
