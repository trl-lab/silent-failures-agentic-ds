import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Create working directory if it doesn't exist
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target_col = "Rented Bike Count"


# 2. Feature Engineering Function
def preprocess_data(df, is_train=True):
    df = df.copy()

    # Standardize column names (strip leading/trailing whitespaces if any)
    df.columns = [col.strip() for col in df.columns]

    # Parse Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        # Fallback if parsing with standard format returns NaTs
        if df["Date"].isna().any():
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
        df["DayOfYear"] = df["Date"].dt.dayofyear
        df.drop(columns=["Date"], inplace=True)

    # Cyclic encoding for time variables
    if "Hour" in df.columns:
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)

    if "Month" in df.columns:
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

    # Weather interaction features
    if "Temperature" in df.columns and "Dew point temperature" in df.columns:
        df["Temp_Dew_Diff"] = df["Temperature"] - df["Dew point temperature"]

    if "Temperature" in df.columns and "Humidity" in df.columns:
        # Humidex proxy feature
        df["Temp_Humidity_Ratio"] = df["Temperature"] / (df["Humidity"] + 1.0)

    if "Rainfall" in df.columns:
        df["Is_Raining"] = (df["Rainfall"] > 0).astype(int)

    if "Snowfall" in df.columns:
        df["Is_Snowing"] = (df["Snowfall"] > 0).astype(int)

    # Categorical Columns Encoding
    cat_cols = ["Seasons", "Holiday", "Functioning Day"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


train_df = preprocess_data(train, is_train=True)
test_df = preprocess_data(test, is_train=False)

# Preserve raw Functioning Day for post-processing logic
test_functioning = (
    test_df["Functioning Day"] if "Functioning Day" in test_df.columns else None
)

features = [c for c in train_df.columns if c != target_col]
X = train_df[features]
y = train_df[target_col]
X_test = test_df[features]

# 3. Validation & Model Training
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

cat_cols = ["Seasons", "Holiday", "Functioning Day"]
categorical_features = [c for c in cat_cols if c in X.columns]

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=categorical_features,
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val)
    # Enforce non-negative demand predictions
    val_pred = np.clip(val_pred, 0, None)

    # Zero out demand when Functioning Day is 'No'
    if "Functioning Day" in X_val.columns:
        is_not_functioning = X_val["Functioning Day"].astype(str).str.lower() == "no"
        val_pred[is_not_functioning] = 0

    oof_preds[val_idx] = val_pred

    t_pred = model.predict(X_test)
    t_pred = np.clip(t_pred, 0, None)
    if "Functioning Day" in X_test.columns:
        is_not_func_test = X_test["Functioning Day"].astype(str).str.lower() == "no"
        t_pred[is_not_func_test] = 0

    test_preds += t_pred / kf.n_splits

# Enforce final train post-processing check
if "Functioning Day" in X.columns:
    is_not_functioning_oof = X["Functioning Day"].astype(str).str.lower() == "no"
    oof_preds[is_not_functioning_oof] = 0

oof_rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"5-Fold CV Out-Of-Fold RMSE: {oof_rmse:.4f}")

# 4. Save Submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Rented Bike Count": test_preds}
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
