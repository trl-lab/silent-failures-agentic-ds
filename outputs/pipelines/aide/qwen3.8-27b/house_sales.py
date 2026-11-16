import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "price"

# Separate target and identifiers.
y = train[TARGET].astype(float).copy()
train = train.drop(columns=[TARGET])

if "id" in train.columns:
    train = train.drop(columns=["id"])
if "id" in test.columns:
    test = test.drop(columns=["id"])
if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

# Remove rows with missing target, if any.
mask = y.notna()
y = y[mask]
train = train.loc[mask]

# Use the training date minimum as a stable reference for date-derived features.
train_min_date = None
if "date" in train.columns:
    train_min_date = pd.to_datetime(train["date"], errors="coerce").min()


def preprocess(df: pd.DataFrame, train_min_date=None) -> pd.DataFrame:
    df = df.copy()

    # Convert date to useful numeric features and drop the raw date column.
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["date_year"] = dt.dt.year
        df["date_month"] = dt.dt.month
        df["date_dayofweek"] = dt.dt.dayofweek

        if train_min_date is not None:
            df["date_days"] = (dt - train_min_date).dt.days
        else:
            df["date_days"] = (dt - dt.min()).dt.days

        df = df.drop(columns=["date"])

    # Keep numeric features only. This drops address/city/zip if they are strings.
    df = df.select_dtypes(include=[np.number])

    # Ensure all remaining columns are numeric floats.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df


X_train = preprocess(train, train_min_date)
X_test = preprocess(test, train_min_date)

# Drop columns that are all-NaN or constant in the training set.
bad_cols = []
for col in X_train.columns:
    if X_train[col].isna().all() or X_train[col].nunique(dropna=True) <= 1:
        bad_cols.append(col)

X_train = X_train.drop(columns=bad_cols)
X_test = X_test.drop(columns=[c for c in bad_cols if c in X_test.columns])

# Align test columns with training columns.
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = np.nan
X_test = X_test[X_train.columns]

# Fallback if no usable features remain.
if X_train.shape[1] == 0:
    fallback = y.mean()
    print(f"Validation RMSE: {np.sqrt(mean_squared_error(y, fallback)):.4f}")
    pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "price": fallback,
        }
    ).to_csv("./working/submission.csv", index=False)
    raise SystemExit

# Log-transform the positive target for more stable regression.
y_log = np.log1p(y)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
validation_rmses = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_log.iloc[train_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_log.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    pred_log = model.predict(X_val)
    pred_price = np.expm1(pred_log)
    pred_price = np.maximum(pred_price, 0.0)

    rmse = float(np.sqrt(mean_squared_error(y.iloc[val_idx], pred_price)))
    validation_rmses.append(rmse)
    best_iterations.append(max(1, int(model.best_iteration)))

    print(f"Fold {fold} RMSE: {rmse:.4f}")

mean_rmse = float(np.mean(validation_rmses))
std_rmse = float(np.std(validation_rmses))
print(f"Validation RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}")

# Retrain on the full training set using the median best iteration count.
final_rounds = max(1, int(np.median(best_iterations)))
dtrain_full = lgb.Dataset(X_train, label=y_log)
final_model = lgb.train(params, dtrain_full, num_boost_round=final_rounds)

test_pred_log = final_model.predict(X_test)
test_pred_price = np.expm1(test_pred_log)
test_pred_price = np.maximum(test_pred_price, 0.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test)),
        "price": test_pred_price,
    }
)
submission.to_csv("./working/submission.csv", index=False)
