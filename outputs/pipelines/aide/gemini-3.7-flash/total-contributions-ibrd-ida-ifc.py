import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Total Contribution (USD)"
y = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()


# 2. Feature Engineering
def engineer_features(df):
    df = df.copy()

    # Identify date columns
    for col in df.columns:
        if df[col].dtype == "object":
            # Try to convert to datetime
            try:
                parsed_dates = pd.to_datetime(df[col], errors="coerce")
                # If a significant portion is parsed as dates, extract date features
                if parsed_dates.notna().sum() > 0.5 * len(df):
                    df[f"{col}_year"] = parsed_dates.dt.year.fillna(-1).astype(int)
                    df[f"{col}_month"] = parsed_dates.dt.month.fillna(-1).astype(int)
                    df[f"{col}_day"] = parsed_dates.dt.day.fillna(-1).astype(int)
                    df[f"{col}_dayofweek"] = parsed_dates.dt.dayofweek.fillna(
                        -1
                    ).astype(int)
                    df[f"{col}_quarter"] = parsed_dates.dt.quarter.fillna(-1).astype(
                        int
                    )
                    df.drop(columns=[col], inplace=True)
            except Exception:
                pass

    return df


X_train = engineer_features(X_train)
X_test = engineer_features(X_test)

# Handle categorical columns
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

for col in cat_cols:
    # Frequency encoding
    freq = pd.concat([X_train[col], X_test[col]]).value_counts(dropna=False).to_dict()
    X_train[f"{col}_freq"] = X_train[col].map(freq).fillna(0)
    X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)

    # Categorical code / label encoding
    categories = {
        val: idx
        for idx, val in enumerate(
            pd.concat([X_train[col], X_test[col]]).dropna().unique()
        )
    }
    X_train[col] = X_train[col].map(categories).fillna(-1).astype(int)
    X_test[col] = X_test[col].map(categories).fillna(-1).astype(int)

# Ensure all feature columns match
feature_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[feature_cols].copy()
X_test = X_test[feature_cols].copy()

# 3. Model Training & Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / n_splits

# 4. Evaluation
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold Cross-Validation RMSE: {cv_rmse:.4f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Total Contribution (USD)": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
