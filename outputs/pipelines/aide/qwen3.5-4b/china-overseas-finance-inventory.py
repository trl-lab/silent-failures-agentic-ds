import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [col for col in train_df.columns if col != "installed_capacity"]
X_train = train_df[feature_cols]
y_train = train_df["installed_capacity"]
X_test = test_df[feature_cols]
test_ids = test_df.index

# Create validation split
np.random.seed(42)
n_samples = len(X_train)
split_idx = int(0.8 * n_samples)
X_train_cv = X_train.iloc[:split_idx]
y_train_cv = y_train.iloc[:split_idx]
X_val = X_train.iloc[split_idx:]
y_val = y_train.iloc[split_idx:]

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Train model
model.fit(X_train_cv, y_train_cv, eval_set=[(X_val, y_val)], callbacks=callbacks)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

# Predict on test set
test_preds = model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": test_ids, "installed_capacity": test_preds})
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

# Define features and target
feature_cols = [col for col in train_df.columns if col != "installed_capacity"]
X_train = train_df[feature_cols].copy()
y_train = train_df["installed_capacity"].values
X_test = test_df[feature_cols].copy()

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store predictions from each fold
fold_preds = []
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train[val_idx]

    # Prepare data for LightGBM
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Train LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    model.fit(train_data, val_data, callbacks=callbacks)

    # Get validation predictions and score
    y_val_pred = model.predict(X_val_fold)
    val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
    fold_scores.append(val_rmse)

    # Store fold predictions for final ensemble
    fold_preds.append(model.predict(X_test))

# Average predictions from all folds
final_preds = np.mean(fold_preds, axis=0)

# Calculate cross-validation RMSE
mean_cv_rmse = np.mean(fold_scores)
print(f"5-Fold Cross-Validation RMSE: {mean_cv_rmse:.4f}")

# Create submission file
submission = pd.DataFrame({"id": range(len(X_test)), "installed_capacity": final_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
