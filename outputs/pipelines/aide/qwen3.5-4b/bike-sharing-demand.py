import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Feature engineering
def create_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["dayofmonth"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.dayofweek

    # Categorical encoding
    df["season"] = df["season"].astype(str)
    df["weather"] = df["weather"].astype(str)
    df["workingday"] = df["workingday"].astype(str)
    df["holiday"] = df["holiday"].astype(str)

    return df


# Apply feature engineering
train_df = create_features(train_df)
test_df = create_features(test_df)

# Define features
feature_cols = [
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "season",
    "weather",
    "workingday",
    "holiday",
    "hour",
    "dayofweek",
    "dayofmonth",
]

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df["count"]

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose_eval=0,
)

# Train model with 5-fold cross-validation
best_score = 0
best_model = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Train model
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Evaluate on validation fold
    y_pred_fold = model.predict(X_val_fold)
    rmsle_fold = np.sqrt(
        np.mean((np.log(y_val_fold + 1) - np.log(y_pred_fold + 1)) ** 2)
    )

    print(f"Fold {fold + 1}: RMSLE = {rmsle_fold:.4f}")

    if rmsle_fold < best_score:
        best_score = rmsle_fold
        best_model = model

# Final evaluation on test data
X_test = test_df[feature_cols]
y_test_pred = best_model.predict(X_test)

# Save predictions
submission = pd.DataFrame({"id": range(len(y_test_pred)), "count": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)

print(f"Final RMSLE on validation: {best_score:.4f}")
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Feature engineering
def create_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["dayofmonth"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["season"] = df["season"].astype(str)
    df["weather"] = df["weather"].astype(str)
    df["workingday"] = df["workingday"].astype(str)
    df["holiday"] = df["holiday"].astype(str)
    return df


# Apply feature engineering
train_df = create_features(train_df)
test_df = create_features(test_df)

# Define features (exclude target and datetime column)
feature_cols = [
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "season",
    "weather",
    "workingday",
    "holiday",
    "hour",
    "dayofweek",
    "dayofmonth",
]

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df["count"]

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose_eval=-1,
    early_stopping_rounds=50,
    force_col_wise=True,
)

# Track best validation performance
best_rmsle = float("inf")
best_model = None
best_fold_rmsle = {}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Train model
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Evaluate on validation fold
    y_pred_fold = model.predict(X_val_fold)

    # Calculate RMSLE (root mean squared log error)
    rmsle_fold = np.sqrt(
        np.mean((np.log(y_val_fold + 1) - np.log(y_pred_fold + 1)) ** 2)
    )
    best_fold_rmsle[fold] = rmsle_fold

    if rmsle_fold < best_rmsle:
        best_rmsle = rmsle_fold
        best_model = model

# Print validation metrics
print("Cross-validation RMSLE scores:")
for fold, rmsle in best_fold_rmsle.items():
    print(f"Fold {fold + 1}: RMSLE = {rmsle:.4f}")
print(f"Best validation RMSLE: {best_rmsle:.4f}")

# Final evaluation on test data
X_test = test_df[feature_cols]
y_test_pred = best_model.predict(X_test)

# Save predictions
submission = pd.DataFrame({"id": range(len(y_test_pred)), "count": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
