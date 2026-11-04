import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract features and target
train_id = train_df["id"].values if "id" in train_df.columns else train_df.index.values
train_X = train_df.drop(
    columns=["RATIO_ENROLL_EMP", "id"] if "id" in train_df.columns else ["id"]
)
train_y = train_df["RATIO_ENROLL_EMP"].values

# Create validation split or use KFold
# For simplicity, use KFold with 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prepare model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    early_stopping_rounds=100,
    verbose_eval=-1,
)

# Train with 5-fold cross-validation
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(train_X)):
    X_train_fold = train_X.iloc[train_idx]
    y_train_fold = train_y.iloc[train_idx]
    X_val_fold = train_X.iloc[val_idx]
    y_val_fold = train_y.iloc[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50)],
    )

    y_pred = model.predict(X_val_fold)
    rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
    cv_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

# Final validation RMSE
print(f"Mean CV RMSE: {np.mean(cv_scores):.4f}")

# Predict on test set
test_X = (
    test_df.drop(columns=["id"]) if "id" in test_df.columns else test_df.iloc[:, :-1]
)
test_predictions = model.predict(test_X)

# Create submission
submission = pd.DataFrame({"id": test_id, "RATIO_ENROLL_EMP": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create feature matrix and target
feature_cols = [col for col in train_df.columns if col != "RATIO_ENROLL_EMP"]
train_X = train_df[feature_cols].copy()
train_y = train_df["RATIO_ENROLL_EMP"].values
train_id = train_df.index.values

# Handle missing values (lightgbm handles them natively)
train_X = train_X.fillna(train_X.median())

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_X)):
    X_train_fold = train_X.iloc[train_idx]
    y_train_fold = train_y[train_idx]
    X_val_fold = train_X.iloc[val_idx]
    y_val_fold = train_y[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred_val = model.predict(X_val_fold)
    rmse = mean_squared_error(y_val_fold, y_pred_val, squared=False)
    cv_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

# Final validation RMSE
print(f"\nMean CV RMSE: {np.mean(cv_scores):.4f}")

# Predict on test set
test_X = test_df[feature_cols].copy()
test_X = test_X.fillna(test_X.median())
test_predictions = model.predict(test_X)

# Create submission with proper id format
submission = pd.DataFrame(
    {"id": test_df.index.values, "RATIO_ENROLL_EMP": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")
