import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features (all columns except target and id)
feature_cols = [col for col in train_df.columns if col not in ["entityPayment", "id"]]

# Initialize LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "random_state": 42,
}

# Initialize callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# Initialize KFold for 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store validation predictions and metrics
val_predictions = []
val_rmse_scores = []

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    train_fold = train_df.iloc[train_idx]
    val_fold = train_df.iloc[val_idx]

    # Create train and validation datasets
    train_data = lgb.Dataset(
        train_fold[feature_cols], label=train_fold["entityPayment"]
    )
    val_data = lgb.Dataset(
        val_fold[feature_cols], label=val_fold["entityPayment"], reference=train_data
    )

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
        verbose_eval=False,
    )

    # Get validation predictions
    val_pred = model.predict(val_fold[feature_cols])
    val_rmse = mean_squared_error(val_fold["entityPayment"], val_pred, squared=False)

    val_predictions.append(val_pred)
    val_rmse_scores.append(val_rmse)

# Calculate mean RMSE across folds
mean_val_rmse = np.mean(val_rmse_scores)
print(f"Cross-validation RMSE: {mean_val_rmse:.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(train_df[feature_cols], label=train_df["entityPayment"])
final_model = lgb.train(
    params,
    train_data_final,
    num_boost_round=1000,
    callbacks=callbacks,
    verbose_eval=False,
)

# Generate predictions for test set
test_pred = final_model.predict(test_df[feature_cols])

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "entityPayment": test_pred})

# Save to working directory
submission.to_csv("./working/submission.csv", index=False)
print("Submission file saved to ./working/submission.csv")
