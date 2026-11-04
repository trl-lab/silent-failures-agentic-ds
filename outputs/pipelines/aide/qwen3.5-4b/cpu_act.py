import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Define features and target
feature_cols = [col for col in train.columns if col != "usr"]
X_train = train[feature_cols]
y_train = train["usr"]
X_test = test[feature_cols]

# Split training data into train/validation (80/20)
from sklearn.model_selection import train_test_split

X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Create LightGBM model with simple parameters
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Train with early stopping on validation set
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
model.fit(X_train_full, y_train_full, eval_set=[(X_val, y_val)], callbacks=callbacks)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Validation RMSE: {val_rmse:.6f}")

# Predict on test set
test_predictions = model.predict(X_test)

# Save to submission file
submission = pd.DataFrame({"id": range(len(test_predictions)), "usr": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
