import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Preprocess date features
ref_date = pd.to_datetime("2015-01-01")


def extract_date_features(df):
    dates = pd.to_datetime(df["Open Date"])
    df["open_year"] = dates.dt.year
    df["open_month"] = dates.dt.month
    df["open_day"] = dates.dt.day
    df["open_dayofweek"] = dates.dt.dayofweek
    df["days_open"] = (ref_date - dates).dt.days
    return df


train_df = extract_date_features(train_df)
test_df = extract_date_features(test_df)

# Partition P-variables into domain-specific functional subgroups
p_geo = [f"P{i}" for i in range(1, 12) if f"P{i}" in train_df.columns]
p_demo = [f"P{i}" for i in range(12, 24) if f"P{i}" in train_df.columns]
p_ops = [f"P{i}" for i in range(24, 38) if f"P{i}" in train_df.columns]
p_all = [c for c in train_df.columns if c.startswith("P")]

for df in [train_df, test_df]:
    for group_name, cols in [
        ("p_geo", p_geo),
        ("p_demo", p_demo),
        ("p_ops", p_ops),
        ("p_all", p_all),
    ]:
        if cols:
            df[f"{group_name}_sum"] = df[cols].sum(axis=1)
            df[f"{group_name}_mean"] = df[cols].mean(axis=1)
            df[f"{group_name}_std"] = df[cols].std(axis=1)
            df[f"{group_name}_max"] = df[cols].max(axis=1)
            df[f"{group_name}_min"] = df[cols].min(axis=1)

cat_cols = ["City", "City Group", "Type"]
feature_cols = [c for c in train_df.columns if c not in ["Id", "Open Date", "revenue"]]

X = train_df[feature_cols].copy()
y = train_df["revenue"].values
X_test = test_df[feature_cols].copy()

# 5-fold Cross-Validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for train_idx, val_idx in kf.split(X, y):
    X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
    X_val, y_val = X.iloc[val_idx].copy(), y[val_idx]
    X_test_fold = X_test.copy()

    y_train_log = np.log1p(y_train)
    global_mean = np.mean(y_train_log)

    # Out-of-fold target encoding for categorical columns
    for col in cat_cols:
        means = (
            pd.DataFrame({"cat": X_train[col], "target": y_train_log})
            .groupby("cat", observed=False)["target"]
            .mean()
        )

        X_train[col + "_te"] = X_train[col].map(means).fillna(global_mean).astype(float)
        X_val[col + "_te"] = X_val[col].map(means).fillna(global_mean).astype(float)
        X_test_fold[col + "_te"] = (
            X_test_fold[col].map(means).fillna(global_mean).astype(float)
        )

    fold_features = [c for c in X_train.columns if c not in cat_cols]

    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )

    model.fit(X_train[fold_features], y_train_log)

    val_preds_log = model.predict(X_val[fold_features])
    test_preds_log = model.predict(X_test_fold[fold_features])

    oof_preds[val_idx] = np.expm1(val_preds_log)
    test_preds += np.expm1(test_preds_log) / kf.n_splits

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Save test predictions
submission = pd.DataFrame({"id": np.arange(len(test_df)), "revenue": test_preds})
submission.to_csv("./working/submission.csv", index=False)
