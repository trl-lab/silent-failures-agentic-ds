import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


def engineer_features(df):
    df = df.copy()
    original_index = df.index

    numeric_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure datetime parsing and chronological ordering for temporal feature extraction
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df["hour"] = df["date"].dt.hour
        df["minute"] = df["date"].dt.minute
        df["dayofweek"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

        # Cyclical time features
        time_minute = df["hour"] * 60 + df["minute"]
        df["sin_time"] = np.sin(2 * np.pi * time_minute / (24 * 60))
        df["cos_time"] = np.cos(2 * np.pi * time_minute / (24 * 60))

    # Multi-scale temporal difference and rolling statistics
    for col in ["CO2", "Light", "Temperature", "Humidity"]:
        if col in df.columns:
            # First-order differences capturing rates of change
            for lag in [1, 3, 5, 15]:
                df[f"{col}_diff_{lag}"] = df[col].diff(lag).fillna(0)

            # Rolling window statistics capturing local trends and volatility
            for window in [5, 15, 30]:
                df[f"{col}_roll_mean_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).mean().fillna(df[col])
                )
                df[f"{col}_roll_std_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).std().fillna(0)
                )

    # Domain interaction features
    if "Light" in df.columns and "CO2" in df.columns:
        df["Light_CO2"] = df["Light"] * df["CO2"]
        df["Light_is_zero"] = (df["Light"] == 0).astype(int)
        df["CO2_per_Light"] = df["CO2"] / (df["Light"] + 1.0)

    if "Temperature" in df.columns and "Humidity" in df.columns:
        df["Temp_Humidity"] = df["Temperature"] * df["Humidity"]

    if "date" in df.columns:
        df.drop(columns=["date"], inplace=True)

    # Restore original ordering
    df = df.loc[original_index]
    return df


# Apply feature engineering
train_features = engineer_features(train_df)
test_features = engineer_features(test_df)

target_col = "Occupancy"
id_col = "id"

features = [col for col in train_features.columns if col not in [target_col, id_col]]

X = train_features[features]
y = pd.to_numeric(train_features[target_col], errors="coerce").astype(int)
X_test = test_features[features]

# Stratified 5-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# Compute validation ROC-AUC score
cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC-AUC: {cv_auc:.6f}")

# Save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": (test_df["id"] if "id" in test_df.columns else np.arange(len(test_df))),
        "Occupancy": test_preds,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved predictions to ./working/submission.csv")
