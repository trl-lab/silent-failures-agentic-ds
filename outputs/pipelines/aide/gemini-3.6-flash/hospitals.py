import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "BEDS"
features = [c for c in train_df.columns if c != target_col]

# Identify feature types
categorical_cols = []
numerical_cols = []

for c in features:
    if train_df[c].dtype == "object" or train_df[c].dtype.name == "category":
        categorical_cols.append(c)
    else:
        numerical_cols.append(c)

# Concatenate for consistent processing
df_all = pd.concat([train_df[features], test_df[features]], axis=0).reset_index(
    drop=True
)

# Frequency encoding for categorical columns
for col in categorical_cols:
    freq = df_all[col].value_counts(normalize=True)
    df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0)

# Group aggregations for numerical columns by primary categoricals
group_cols = [c for c in ["STATE", "TYPE", "COUNTY"] if c in categorical_cols]

for g_col in group_cols:
    for num_col in numerical_cols:
        grp = df_all.groupby(g_col)[num_col]
        g_mean = grp.transform("mean")
        g_std = grp.transform("std").fillna(0)

        df_all[f"{num_col}_grp_{g_col}_mean"] = g_mean
        df_all[f"{num_col}_grp_{g_col}_std"] = g_std
        df_all[f"{num_col}_ratio_{g_col}"] = (df_all[num_col] + 1e-5) / (g_mean + 1e-5)

# Convert categorical columns to category dtype for LightGBM
for col in categorical_cols:
    df_all[col] = df_all[col].astype("category")

# Separate processed features back into train and test sets
X_train = df_all.iloc[: len(train_df)].copy()
y_train = train_df[target_col].values
X_test = df_all.iloc[len(train_df) :].copy()

# Setup 5-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

# Log1p transformation on positive targets
y_train_log = np.log1p(np.maximum(0, y_train))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "random_state": 42,
    "n_estimators": 1500,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr_log = X_train.iloc[train_idx], y_train_log[train_idx]
    X_va, y_va_log = X_train.iloc[val_idx], y_train_log[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_preds_log = model.predict(X_va)
    oof_preds[val_idx] = np.clip(np.expm1(val_preds_log), 0, None)

    t_preds_log = model.predict(X_test)
    test_preds += np.clip(np.expm1(t_preds_log), 0, None) / kf.n_splits

# Compute Out-of-Fold Root Mean Squared Error
cv_rmse = root_mean_squared_error(y_train, oof_preds)
print(f"5-Fold CV RMSE: {cv_rmse:.4f}")

# Create and save final submission file
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "BEDS": test_preds})

os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
