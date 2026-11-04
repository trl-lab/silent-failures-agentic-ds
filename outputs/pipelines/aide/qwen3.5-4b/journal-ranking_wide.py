import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude id and target)
train_features = train_df.drop(columns=["id", "H index"])
test_features = test_df.drop(columns=["id"])

# Initialize LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation with proper validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
predictions = {}

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train_fold = train_features.iloc[train_idx]
    y_train_fold = train_df["H index"].iloc[train_idx]
    X_val_fold = train_features.iloc[val_idx]
    y_val_fold = train_df["H index"].iloc[val_idx]

    # Train model with early stopping
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    callbacks = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(-1, verbose=False),
    ]

    model.fit(
        train_data, callbacks=callbacks, valid_sets=[val_data], num_boost_round=1000
    )

    # Store predictions for validation
    val_pred = model.predict(X_val_fold)
    fold_scores.append(mean_squared_error(y_val_fold, val_pred))

    # Store predictions for test data
    test_pred = model.predict(test_features)
    predictions["test"] = test_pred

# Calculate RMSE on validation folds
validation_rmse = np.sqrt(np.mean(fold_scores))
print(f"5-Fold Cross-Validation RMSE: {validation_rmse:.4f}")

# Save test predictions to submission.csv
submission = pd.DataFrame({"id": range(len(test_df)), "H index": predictions["test"]})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
