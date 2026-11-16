import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Ensure working directory exists
import os

os.makedirs("./working", exist_ok=True)

# Define target and features
target = "Meta Critic"
X_train = train.drop(columns=[target])
y_train = train[target]
X_test = test.drop(columns=[target]) if target in test.columns else test

# Handle ID column for submission
test["id"] = test.index
X_test = X_test.reset_index(drop=True)

# LightGBM Model
model = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=1000,
    random_state=42,
    verbose=-1,
)

# Callbacks for early stopping and logging
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-Fold Cross-Validation to estimate RMSE on hold-out validation set
cv_scores = cross_val_score(
    model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
)
# Note: cross_val_score with 'neg_mean_squared_error' returns negative MSE.
# We need RMSE. Let's compute it manually from the scores.
cv_mse = -cv_scores.mean()
cv_rmse = np.sqrt(cv_mse)

print(f"Cross-Validation RMSE: {cv_rmse:.4f}")

# Train final model on full training data
model.fit(X_train, y_train, callbacks=callbacks)

# Predict on test data
predictions = model.predict(X_test)

# Create submission dataframe
submission = pd.DataFrame({"id": test.index, "Meta Critic": predictions})

# Save submission
submission.to_csv("./working/submission.csv", index=False)
