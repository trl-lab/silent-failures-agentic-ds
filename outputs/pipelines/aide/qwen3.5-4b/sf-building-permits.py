import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (everything except target and id)
# Assuming 'id' is in test but not train, and 'time_to_approve' is target
train_cols = [col for col in train_df.columns if col != "time_to_approve"]
test_cols = [col for col in test_df.columns if col != "time_to_approve"]

# Ensure column alignment
X_train = train_df[train_cols].copy()
y_train = train_df["time_to_approve"].copy()
X_test = test_df[test_cols].copy()

# Handle potential missing values (simple fill)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# 5-Fold Cross-Validation to estimate RMSE
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Prepare data for LightGBM
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    valid_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    # Parameters
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

    # Train with callbacks
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=callbacks,
        verbose_eval=0,
    )

    # Predict on validation set
    y_pred_fold = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
    cv_scores.append(rmse)

# Calculate average CV RMSE
cv_rmse = np.mean(cv_scores)
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on full training data
train_data_final = lgb.Dataset(X_train, label=y_train)
model_final = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50)],
    verbose_eval=0,
)

# Predict on test set
y_test_pred = model_final.predict(X_test)

# Prepare submission
submission = pd.DataFrame({"id": range(len(X_test)), "time_to_approve": y_test_pred})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
