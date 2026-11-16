import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define feature columns (exclude target and id columns)
feature_cols = [
    col for col in train_df.columns if col not in ["id", "target", "row_id"]
]

# Handle missing values - impute with median
train_imputer = SimpleImputer(strategy="median")
train_df[feature_cols] = train_imputer.fit_transform(train_df[feature_cols])

test_imputer = SimpleImputer(strategy="median")
test_df[feature_cols] = test_imputer.fit_transform(test_df[feature_cols])

# Create target column for training
y = train_df["target"].values
X = train_df[feature_cols].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Set model parameters
params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# Training results for evaluation
validation_scores = []
valid_predictions = []

# Store training history for early stopping
train_history = []

# Train LightGBM model with 5-fold CV
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_fold = X[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X[val_idx]
    y_val_fold = y[val_idx]

    # Create datasets
    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, feature_name=feature_cols
    )
    val_data = lgb.Dataset(
        X_val_fold, label=y_val_fold, feature_name=feature_cols, reference=train_data
    )

    # Callbacks for early stopping
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=callbacks,
        valid_sets=[val_data],
        valid_names=["valid"],
    )

    # Evaluate on validation set
    val_pred = model.predict(X_val_fold)
    mae = mean_absolute_error(y_val_fold, val_pred)
    validation_scores.append(mae)
    valid_predictions.append(val_pred)

    print(f"Fold {fold + 1}: MAE = {mae:.4f}")

# Calculate mean MAE across folds
mean_mae = np.mean(validation_scores)
print(f"\nMean Validation MAE: {mean_mae:.4f}")

# Prepare test predictions
test_X = test_df[feature_cols].values
test_predictions = model.predict(test_X)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "target": test_predictions}
)

# Save submission
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
