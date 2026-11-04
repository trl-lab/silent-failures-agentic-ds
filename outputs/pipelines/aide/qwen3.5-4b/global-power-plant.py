import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "capacity_mw"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    verbose=-1,
)

# Train with 5-fold cross-validation and evaluate on each fold
best_score = float("inf")
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Create callbacks
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

    # Train model
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=False,
    )

    # Calculate RMSE on validation fold
    val_pred = model.predict(X_val_fold)
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))

    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")

    if fold_rmse < best_score:
        best_score = fold_rmse
        best_model = model

# Evaluate on full validation set (average of all folds)
fold_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    callbacks = [lgb.early_stopping(50, verbose=False)]
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
        verbose=False,
    )

    val_pred = model.predict(X_val_fold)
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    fold_scores.append(fold_rmse)

print(f"Average Cross-Validation RMSE: {np.mean(fold_scores):.4f}")

# Predict on test set
test_predictions = best_model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "capacity_mw": test_predictions})
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
