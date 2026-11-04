import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
train_df["hour"] = pd.to_datetime(train_df["timestamp"]).dt.hour
train_df["day"] = pd.to_datetime(train_df["timestamp"]).dt.dayofweek
train_df["month"] = pd.to_datetime(train_df["timestamp"]).dt.month
test_df["hour"] = pd.to_datetime(test_df["timestamp"]).dt.hour
test_df["day"] = pd.to_datetime(test_df["timestamp"]).dt.dayofweek
test_df["month"] = pd.to_datetime(test_df["timestamp"]).dt.month

# Create features
X_train = train_df[["meter", "hour", "day", "month"]].copy()
X_test = test_df[["meter", "hour", "day", "month"]].copy()
y_train = train_df["meter_reading"].values

# Split into train/validation (temporal split)
train_size = int(len(X_train) * 0.9)
X_train_valid, X_val = X_train[:train_size], X_train[train_size:]
y_train_valid, y_val = y_train[:train_size], y_train[train_size:]

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose_eval=100,
)

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]

model.fit(
    X_train_valid,
    y_train_valid,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks,
    verbose=0,
)

# Predictions
y_pred = model.predict(X_test)

# Submission
submission = pd.DataFrame({"id": test_df.index, "meter_reading": y_pred})
submission.to_csv("./working/submission.csv", index=False)

print("Evaluation metric on validation set:")
print(f"RMSE: {np.sqrt(mean_squared_log_error(y_val, model.predict(X_val)))}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
train_df["hour"] = pd.to_datetime(train_df["timestamp"]).dt.hour
train_df["day"] = pd.to_datetime(train_df["timestamp"]).dt.dayofweek
train_df["month"] = pd.to_datetime(train_df["timestamp"]).dt.month
test_df["hour"] = pd.to_datetime(test_df["timestamp"]).dt.hour
test_df["day"] = pd.to_datetime(test_df["timestamp"]).dt.dayofweek
test_df["month"] = pd.to_datetime(test_df["timestamp"]).dt.month

# Create features
X_train = train_df[["meter", "hour", "day", "month"]].copy()
X_test = test_df[["meter", "hour", "day", "month"]].copy()
y_train = train_df["meter_reading"].values

# Split into train/validation (temporal split)
train_size = int(len(X_train) * 0.8)
X_train_valid, X_val = X_train[:train_size], X_train[train_size:]
y_train_valid, y_val = y_train[:train_size], y_train[train_size:]

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose_eval=100,
)

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(100)]

model.fit(
    X_train_valid,
    y_train_valid,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks,
    verbose=0,
)

# Predictions
y_pred = model.predict(X_test)

# Submission
submission = pd.DataFrame({"id": test_df.index, "meter_reading": y_pred})
submission.to_csv("./working/submission.csv", index=False)

print("Evaluation metric on validation set:")
print(f"RMSE: {np.sqrt(mean_squared_log_error(y_val, model.predict(X_val)))}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering - create temporal features
train_df["hour"] = pd.to_datetime(train_df["timestamp"]).dt.hour
train_df["day_of_week"] = pd.to_datetime(train_df["timestamp"]).dt.dayofweek
train_df["month"] = pd.to_datetime(train_df["timestamp"]).dt.month
test_df["hour"] = pd.to_datetime(test_df["timestamp"]).dt.hour
test_df["day_of_week"] = pd.to_datetime(test_df["timestamp"]).dt.dayofweek
test_df["month"] = pd.to_datetime(test_df["timestamp"]).dt.month

# Prepare features and target
X_train = train_df[["meter", "hour", "day_of_week", "month"]].copy()
y_train = train_df["meter_reading"].values
X_test = test_df[["meter", "hour", "day_of_week", "month"]].copy()

# Use temporal train/validation split to avoid data leakage
train_size = int(len(X_train) * 0.8)
X_train_val, X_val = X_train[:train_size], X_train[train_size:]
y_train_val, y_val = y_train[:train_size], y_train[train_size:]

# LightGBM model with proper callbacks for LGBM 4.6.0
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

model.fit(
    X_train_val, y_train_val, eval_set=[(X_val, y_val)], callbacks=callbacks, verbose=0
)

# Validate on held-out set
y_val_pred = model.predict(X_val)
val_rmsle = mean_squared_log_error(y_val, y_val_pred, squared=False)

# Generate test predictions
y_test_pred = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(X_test)), "meter_reading": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)

print(f"Validation RMSLE: {val_rmsle:.4f}")
print("Submission saved to ./working/submission.csv")
