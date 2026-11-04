import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and target
feature_cols = ["DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
X_train = train_df[feature_cols]
y_train = train_df["pm2.5"]

# Time-based split to avoid leakage
train_df = train_df.sort_values("year")
train_df = train_df.sort_values("month")
train_df = train_df.sort_values("day")
train_df = train_df.sort_values("hour")

# Use time-based train/test split
train_size = int(0.8 * len(train_df))
X_train, X_test = (
    train_df.iloc[:train_size][feature_cols],
    train_df.iloc[train_size:][feature_cols],
)
y_train = train_df.iloc[:train_size]["pm2.5"]
y_test = train_df.iloc[train_size:]["pm2.5"]

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Train with early stopping
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE on validation set: {rmse}")


import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and target
feature_cols = ["DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
X_train = train_df[feature_cols]
y_train = train_df["pm2.5"]

# Time-based train/test split to avoid leakage
train_df_sorted = train_df.sort_values(["year", "month", "day", "hour"])
train_size = int(0.8 * len(train_df_sorted))
X_train, y_train = (
    train_df_sorted.iloc[:train_size][feature_cols],
    train_df_sorted.iloc[:train_size]["pm2.5"],
)
X_test, y_test = (
    train_df_sorted.iloc[train_size:][feature_cols],
    train_df_sorted.iloc[train_size:]["pm2.5"],
)

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Train with early stopping
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE on validation set: {rmse}")

# Predict on test set
test_predictions = model.predict(test_df[feature_cols])
submission = pd.DataFrame({"id": range(len(test_df)), "pm2.5": test_predictions})
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and target
feature_cols = ["DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
X_train = train_df[feature_cols]
y_train = train_df["pm2.5"]
X_test = test_df[feature_cols]

# Time-based train/test split to avoid data leakage
train_df_sorted = train_df.sort_values(["year", "month", "day", "hour"])
train_size = int(0.8 * len(train_df_sorted))
X_train, y_train = (
    train_df_sorted.iloc[:train_size][feature_cols],
    train_df_sorted.iloc[:train_size]["pm2.5"],
)
X_test_val, y_test_val = (
    train_df_sorted.iloc[train_size:][feature_cols],
    train_df_sorted.iloc[train_size:]["pm2.5"],
)

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Train with early stopping and log evaluation
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test_val, y_test_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_pred_val = model.predict(X_test_val)
rmse_val = np.sqrt(mean_squared_error(y_test_val, y_pred_val))
print(f"RMSE on validation set: {rmse_val:.4f}")

# Predict on test set
test_predictions = model.predict(X_test)
submission = pd.DataFrame({"id": range(len(test_df)), "pm2.5": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
