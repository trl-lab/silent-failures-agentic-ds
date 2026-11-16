import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

# Create working directory if it doesn't exist
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Process date and engineer features
def preprocess_features(df):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek
        df["dayofyear"] = df["date"].dt.dayofyear
        df["quarter"] = df["date"].dt.quarter
        df = df.drop(columns=["date"])

    # Identify index feature columns (excluding target 'ISE' if present)
    index_cols = [
        col
        for col in ["SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]
        if col in df.columns
    ]

    if len(index_cols) > 0:
        # Calculate cross-sectional features across indices
        df["market_mean"] = df[index_cols].mean(axis=1)
        df["market_std"] = df[index_cols].std(axis=1)
        df["market_min"] = df[index_cols].min(axis=1)
        df["market_max"] = df[index_cols].max(axis=1)
        df["market_median"] = df[index_cols].median(axis=1)
        df["market_skew"] = df[index_cols].skew(axis=1)
        df["market_sum"] = df[index_cols].sum(axis=1)
        df["market_spread"] = df["market_max"] - df["market_min"]

    return df


# Target variable
target_col = "ISE"
y_train = train_df[target_col].values

# Extract features
X_train_raw = preprocess_features(train_df.drop(columns=[target_col]))
X_test_raw = preprocess_features(
    test_df.drop(columns=[target_col] if target_col in test_df.columns else [])
)

# Align columns
X_train_raw, X_test_raw = X_train_raw.align(
    X_test_raw, join="left", axis=1, fill_value=0
)

# Fill any missing values
X_train_raw = X_train_raw.fillna(0)
X_test_raw = X_test_raw.fillna(0)

# Setup K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

# Candidate alpha grid for Ridge cross-validation
alphas_grid = np.logspace(-8, 8, 200)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_raw, y_train)):
    X_tr, y_tr = X_train_raw.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_raw.iloc[val_idx], y_train[val_idx]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_test_raw)

    model = RidgeCV(alphas=alphas_grid, store_cv_results=False)
    model.fit(X_tr_scaled, y_tr)

    oof_predictions[val_idx] = model.predict(X_va_scaled)
    test_predictions += model.predict(X_te_scaled) / kf.n_splits

# Compute Out-Of-Fold RMSE
oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
print(f"Validation Metric (OOF RMSE): {oof_rmse:.6f}")

# Save submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_predictions)), "ISE": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
