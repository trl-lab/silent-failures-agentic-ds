import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "NET_GEN"


# 2. Preprocess and feature engineering
def preprocess_and_feature_engineer(train, test):
    df_all = pd.concat(
        [train.assign(is_train=1), test.assign(is_train=0, **{target_col: np.nan})],
        ignore_index=True,
    )

    # Process datetime columns
    datetime_cols = df_all.select_dtypes(
        include=["datetime64", "datetime"]
    ).columns.tolist()
    for col in df_all.columns:
        if col not in datetime_cols and df_all[col].dtype == "object":
            try:
                converted = pd.to_datetime(df_all[col], errors="raise")
                # If conversion succeeded without raising
                df_all[col] = converted
                datetime_cols.append(col)
            except (ValueError, TypeError):
                pass

    for col in datetime_cols:
        df_all[f"{col}_year"] = df_all[col].dt.year
        df_all[f"{col}_month"] = df_all[col].dt.month
        df_all[f"{col}_day"] = df_all[col].dt.day
        df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
        df_all[f"{col}_timestamp"] = df_all[col].astype("int64") // 10**9
        df_all.drop(columns=[col], inplace=True)

    # Process categorical / object columns
    cat_cols = df_all.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df_all[col] = df_all[col].astype("category")

    # Numerical feature interactions on key numeric columns
    num_cols = [
        c
        for c in df_all.select_dtypes(include=[np.number]).columns
        if c not in [target_col, "is_train"]
    ]

    # Extract train and test back
    train_processed = (
        df_all[df_all["is_train"] == 1]
        .drop(columns=["is_train"])
        .reset_index(drop=True)
    )
    test_processed = (
        df_all[df_all["is_train"] == 0]
        .drop(columns=["is_train", target_col])
        .reset_index(drop=True)
    )

    return train_processed, test_processed


train_clean, test_clean = preprocess_and_feature_engineer(train_df, test_df)

features = [c for c in train_clean.columns if c != target_col]
X = train_clean[features]
y = train_clean[target_col].values
X_test = test_clean[features]

# 3. 5-Fold Cross Validation with LightGBM
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_clean))
test_preds = np.zeros(len(test_clean))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 15,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        callbacks=callbacks,
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# 4. Save test predictions
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "NET_GEN": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print(
    f"Submission saved successfully to ./working/submission.csv. Shape: {submission.shape}"
)
