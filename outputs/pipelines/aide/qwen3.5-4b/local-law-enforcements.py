import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify features and target
target_col = "number_of_officers"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare train data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
test_X = test_df[feature_cols].copy()

# Split train data for validation (70/30)
split_idx = int(len(X_train) * 0.7)
X_val = X_train.iloc[split_idx:]
y_val = y_train.iloc[split_idx:]
X_train_val = X_train.iloc[:split_idx]
y_train_val = y_train.iloc[:split_idx]

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)

# Train with early stopping
callbacks = [
    lgb.early_stopping(50, verbose=False),
    lgb.log_evaluation(100, verbose=False),
]

model.fit(
    X_train_val,
    y_train_val,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=callbacks,
)

# Validate on test set
y_pred = model.predict(X_train_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Generate test predictions
test_predictions = model.predict(test_X)
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "number_of_officers": test_predictions}
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

# Identify target and features
target_col = "number_of_officers"
feature_cols = [col for col in train_df.columns if col != target_col]

# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Convert to LightGBM format
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, reference=train_data)

# Set up 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    bagging_fraction=0.8,
    bagging_freq=1,
    feature_fraction=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Cross-validation evaluation
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    model.fit(
        X_fold_train,
        y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100, verbose=False),
        ],
    )

    y_pred_val = model.predict(X_fold_val)
    rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred_val))
    cv_scores.append(rmse)
    print(f"Fold {fold+1} CV RMSE: {rmse:.4f}")

# Final CV RMSE
cv_rmse = np.mean(cv_scores)
print(f"\nFinal CV RMSE: {cv_rmse:.4f}")

# Train final model on full training data
model_final = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    bagging_fraction=0.8,
    bagging_freq=1,
    feature_fraction=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

model_final.fit(
    X_train,
    y_train,
    eval_set=[(X_test, None)],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100, verbose=False),
    ],
)

# Generate test predictions
y_test_pred = model_final.predict(X_test)

# Create submission
submission = pd.DataFrame(
    {"id": range(len(y_test_pred)), "number_of_officers": y_test_pred}
)
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
