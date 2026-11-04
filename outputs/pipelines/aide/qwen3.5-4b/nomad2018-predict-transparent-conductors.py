import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Split train data for validation (no data leakage)
X_train, X_val, y_train, y_val = train_test_split(
    train_df.drop("formation_energy_ev_natom", axis=1),
    train_df["formation_energy_ev_natom"],
    test_size=0.2,
    random_state=42,
)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "force_col_wise": True,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

# Create data
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train model with early stopping
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)

# Predict on validation set and calculate RMSLE
y_val_pred = model.predict(X_val)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
rmse_log = np.sqrt(mean_squared_error(np.log(y_val + 1), np.log(y_val_pred + 1)))
print(f"Validation RMSLE: {rmse_log:.6f}")

# Predict on test set
test_predictions = model.predict(test_df)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(test_df)), "formation_energy_ev_natom": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
