import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
# Extract temporal features from date
train_df["date"] = pd.to_datetime(train_df["Date"])
test_df["date"] = pd.to_datetime(test_df["Date"])

# Create features
train_df["day_of_week"] = train_df["date"].dt.dayofweek
test_df["day_of_week"] = test_df["date"].dt.dayofweek

train_df["hour"] = train_df["date"].dt.hour
test_df["hour"] = test_df["date"].dt.hour

# Handle categorical features
train_df["Seasons"] = train_df["Seasons"].astype("category")
train_df["Holiday"] = train_df["Holiday"].astype("category")

test_df["Seasons"] = test_df["Seasons"].astype("category")
test_df["Holiday"] = test_df["Holiday"].astype("category")

# Define feature columns
feature_cols = [
    "Temperature",
    "Humidity",
    "Wind speed",
    "Visibility",
    "Dew point temperature",
    "Solar Radiation",
    "Rainfall",
    "Snowfall",
    "day_of_week",
    "hour",
    "Seasons",
    "Holiday",
]

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df["Rented Bike Count"]

# Prepare test data
X_test = test_df[feature_cols]

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)

# Callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val_fold)
    mse = mean_squared_error(y_val_fold, val_pred)
    cv_scores.append(np.sqrt(mse))  # RMSE

print(f"5-fold CV RMSE: {np.mean(cv_scores):.4f}")

# Predict on test data
test_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": range(len(test_df)), "Rented Bike Count": test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
# Extract temporal features from date
train_df["date"] = pd.to_datetime(train_df["Date"])
test_df["date"] = pd.to_datetime(test_df["Date"])

# Create features
train_df["day_of_week"] = train_df["date"].dt.dayofweek
test_df["day_of_week"] = test_df["date"].dt.dayofweek

train_df["hour"] = train_df["date"].dt.hour
test_df["hour"] = test_df["date"].dt.hour

train_df["day_of_month"] = train_df["date"].dt.day
test_df["day_of_month"] = test_df["date"].dt.day

train_df["month"] = train_df["date"].dt.month
test_df["month"] = test_df["date"].dt.month

train_df["year"] = train_df["date"].dt.year
test_df["year"] = test_df["date"].dt.year

# Handle categorical features
train_df["Seasons"] = train_df["Seasons"].astype("category")
train_df["Holiday"] = train_df["Holiday"].astype("category")

test_df["Seasons"] = test_df["Seasons"].astype("category")
test_df["Holiday"] = test_df["Holiday"].astype("category")

# Define feature columns
feature_cols = [
    "Temperature",
    "Humidity",
    "Wind speed",
    "Visibility",
    "Dew point temperature",
    "Solar Radiation",
    "Rainfall",
    "Snowfall",
    "day_of_week",
    "hour",
    "day_of_month",
    "month",
    "year",
    "Seasons",
    "Holiday",
]

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df["Rented Bike Count"]

# Prepare test data
X_test = test_df[feature_cols]

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1
)

# Callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val_fold)
    mse = mean_squared_error(y_val_fold, val_pred)
    cv_scores.append(np.sqrt(mse))  # RMSE

print(f"5-fold CV RMSE: {np.mean(cv_scores):.4f}")

# Predict on test data
test_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": range(len(test_df)), "Rented Bike Count": test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
# Extract temporal features from date
train_df["date"] = pd.to_datetime(train_df["Date"])
test_df["date"] = pd.to_datetime(test_df["Date"])

# Create features
train_df["day_of_week"] = train_df["date"].dt.dayofweek
test_df["day_of_week"] = test_df["date"].dt.dayofweek

train_df["hour"] = train_df["date"].dt.hour
test_df["hour"] = test_df["date"].dt.hour

train_df["day_of_month"] = train_df["date"].dt.day
test_df["day_of_month"] = test_df["date"].dt.day

train_df["month"] = train_df["date"].dt.month
test_df["month"] = test_df["date"].dt.month

train_df["year"] = train_df["date"].dt.year
test_df["year"] = test_df["date"].dt.year

train_df["day_of_year"] = train_df["date"].dt.dayofyear
test_df["day_of_year"] = test_df["date"].dt.dayofyear

# Handle categorical features
train_df["Seasons"] = train_df["Seasons"].astype("category")
train_df["Holiday"] = train_df["Holiday"].astype("category")

test_df["Seasons"] = test_df["Seasons"].astype("category")
test_df["Holiday"] = test_df["Holiday"].astype("category")

# Define feature columns
feature_cols = [
    "Temperature",
    "Humidity",
    "Wind speed",
    "Visibility",
    "Dew point temperature",
    "Solar Radiation",
    "Rainfall",
    "Snowfall",
    "day_of_week",
    "hour",
    "day_of_month",
    "month",
    "year",
    "day_of_year",
    "Seasons",
    "Holiday",
]

# Prepare training data
X_train = train_df[feature_cols]
y_train = train_df["Rented Bike Count"]

# Prepare test data
X_test = test_df[feature_cols]

# LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# Callbacks
callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val_fold)
    mse = mean_squared_error(y_val_fold, val_pred)
    cv_scores.append(np.sqrt(mse))  # RMSE

print(f"5-fold CV RMSE: {np.mean(cv_scores):.4f}")

# Predict on test data
test_pred = model.predict(X_test)

# Create submission
submission = pd.DataFrame({"id": range(len(test_df)), "Rented Bike Count": test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
