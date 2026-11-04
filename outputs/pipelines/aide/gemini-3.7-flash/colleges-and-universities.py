import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# Define file paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUTPUT_DIR = "./working"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load datasets
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "RATIO_ENROLL_EMP"
drop_cols = [target_col] if target_col in train_df.columns else []
base_feature_cols = [c for c in train_df.columns if c not in drop_cols]

# Identify base categorical and numerical features
categorical_cols = []
numeric_cols = []

for col in base_feature_cols:
    if pd.api.types.is_object_dtype(train_df[col]) or pd.api.types.is_categorical_dtype(
        train_df[col]
    ):
        categorical_cols.append(col)
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
    else:
        numeric_cols.append(col)
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

# Combine for consistent feature engineering (excluding target)
combined = pd.concat(
    [train_df[base_feature_cols], test_df[base_feature_cols]], axis=0
).reset_index(drop=True)
n_train = len(train_df)

# 1. Frequency encoding for categorical columns
for col in categorical_cols:
    freq = combined[col].value_counts(normalize=True)
    combined[f"{col}_freq"] = combined[col].map(freq).fillna(0).astype(np.float32)

# 2. Group aggregation features for moderate-cardinality categorical columns
group_cols = [col for col in categorical_cols if 2 <= combined[col].nunique() <= 150]
top_numeric_cols = numeric_cols[:10]  # Limit to top numeric features to avoid explosion

for g_col in group_cols:
    for num_col in top_numeric_cols:
        grouped = combined.groupby(g_col)[num_col].agg(["mean", "std", "min", "max"])
        grouped.columns = [
            f"{num_col}_by_{g_col}_{stat}" for stat in ["mean", "std", "min", "max"]
        ]
        combined = combined.merge(grouped, on=g_col, how="left")

# 3. Pairwise ratios and differences for top numeric columns
for i in range(len(top_numeric_cols)):
    for j in range(i + 1, min(i + 4, len(top_numeric_cols))):
        c1 = top_numeric_cols[i]
        c2 = top_numeric_cols[j]
        combined[f"{c1}_div_{c2}"] = combined[c1] / (combined[c2].abs() + 1e-5)
        combined[f"{c1}_diff_{c2}"] = combined[c1] - combined[c2]

# Ensure categorical types for LightGBM
for col in categorical_cols:
    combined[col] = combined[col].astype("category")

# Separate back into train and test
X = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()
y = train_df[target_col].values

feature_cols = list(X.columns)

# Set up 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(
        X_train_fold, label=y_train_fold, categorical_feature=categorical_cols
    )
    val_data = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        reference=train_data,
        categorical_feature=categorical_cols,
    )

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    fold_rmse = root_mean_squared_error(y_val_fold, val_pred)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

# Overall Out-of-Fold RMSE
cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"Overall OOF RMSE: {cv_rmse:.5f}")

# Generate submission file
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "RATIO_ENROLL_EMP": test_preds}
)

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission to {SUBMISSION_PATH} with shape {submission.shape}")
