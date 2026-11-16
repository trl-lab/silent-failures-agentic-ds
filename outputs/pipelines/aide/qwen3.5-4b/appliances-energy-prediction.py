import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Feature engineering for date - fixed to handle concatenated date/time format
def extract_date_features(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Handle cases where parsing failed due to concatenated format
    if df["date"].isna().any():
        # Try to fix concatenated date and time format
        df["date"] = df["date"].apply(
            lambda x: pd.to_datetime(
                x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else pd.NaT,
                format=None,
                errors="coerce",
            )
        )

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df


# Handle any remaining NaN dates
train_df = extract_date_features(train_df)
test_df = extract_date_features(test_df)

# Fill any remaining NaN dates with today's date or remove problematic rows
train_df["date"] = train_df["date"].fillna(
    train_df["date"].mode()[0]
    if len(train_df["date"].mode()) > 0
    else pd.Timestamp("2016-01-01")
)
test_df["date"] = test_df["date"].fillna(
    test_df["date"].mode()[0]
    if len(test_df["date"].mode()) > 0
    else pd.Timestamp("2016-01-01")
)

# Define features and target
feature_cols = [c for c in train_df.columns if c not in ["date", "Appliances"]]
X_train = train_df[feature_cols].values
y_train = train_df["Appliances"].values
X_test = test_df[feature_cols].values

# Train/val split for evaluation
X_train_split, X_val, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# LightGBM parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Train with early stopping
train_data = lgb.Dataset(X_train_split, label=y_train_split)
val_data = lgb.Dataset(X_val, label=y_val_split, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val_split, y_val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

# 5-fold cross-validation on training data
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train = X_train[train_idx]
    X_fold_val = X_train[val_idx]
    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]

    train_data_fold = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data_fold = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data_fold)

    model_fold = lgb.train(
        params,
        train_data_fold,
        num_boost_round=1000,
        valid_sets=[train_data_fold, val_data_fold],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_fold_pred = model_fold.predict(X_fold_val)
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
    cv_scores.append(fold_rmse)
    print(f"Fold {fold_idx + 1} RMSE: {fold_rmse:.4f}")

cv_rmse = np.mean(cv_scores)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Generate test predictions
test_pred = model.predict(X_test)

# Save predictions
submission = pd.DataFrame({"id": range(len(test_pred)), "Appliances": test_pred})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
