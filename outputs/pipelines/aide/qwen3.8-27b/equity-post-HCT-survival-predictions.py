import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

# Create working directory
os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate target and time
y = train["efs"]
time = train["efs_time"]
feature_cols = [col for col in train.columns if col not in ["efs", "efs_time", "ID"]]

# Split into train and validation
X_train, X_val, y_train, y_val, time_train, time_val = train_test_split(
    train[feature_cols], y, time, test_size=0.2, random_state=42
)


# Preprocessing functions
def fit_preprocess(X_train):
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    num_medians = {col: X_train[col].median() for col in num_cols}
    cat_categories = {}
    for col in cat_cols:
        categories = X_train[col].dropna().unique().tolist()
        if "Unknown" not in categories:
            categories.append("Unknown")
        cat_categories[col] = categories
    return num_medians, cat_categories, cat_cols, num_cols


def transform(X, num_medians, cat_categories, cat_cols, num_cols):
    X = X.copy()
    for col in num_cols:
        if col in X.columns:
            X[col] = X[col].fillna(num_medians[col])
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].fillna("Unknown")
            valid_cats = set(cat_categories[col])
            X[col] = X[col].apply(lambda x: x if x in valid_cats else "Unknown")
            X[col] = X[col].astype("category")
    return X


# Fit preprocessing on training split
num_medians, cat_categories, cat_cols, num_cols = fit_preprocess(X_train)
X_train_processed = transform(X_train, num_medians, cat_categories, cat_cols, num_cols)
X_val_processed = transform(X_val, num_medians, cat_categories, cat_cols, num_cols)

# Train LightGBM model
train_data = lgb.Dataset(X_train_processed, label=y_train, categorical_feature=cat_cols)
val_data = lgb.Dataset(
    X_val_processed, label=y_val, categorical_feature=cat_cols, reference=train_data
)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[val_data],
    callbacks=[lgb.log_evaluation(-1)],
)

# Predict on validation set
val_pred = model.predict(X_val_processed)


# Concordance index computation
def concordance_index(time, event, risk):
    time = np.asarray(time)
    event = np.asarray(event)
    risk = np.asarray(risk)
    n = len(time)
    concordant = 0
    total = 0
    for i in range(n):
        j = np.arange(i + 1, n)
        if len(j) == 0:
            continue
        # Case 1: Both events
        mask_both = (event[i] == 1) & (event[j] == 1)
        mask_both_valid = mask_both & (time[i] != time[j])
        if np.any(mask_both_valid):
            sub_time_i_less = mask_both_valid & (time[i] < time[j])
            sub_time_i_greater = mask_both_valid & (time[i] > time[j])
            concordant_both = np.sum(sub_time_i_less & (risk[i] > risk[j])) + np.sum(
                sub_time_i_greater & (risk[i] < risk[j])
            )
            total_both = np.sum(mask_both_valid)
            concordant += concordant_both
            total += total_both
        # Case 2: i event, j censored
        mask_i_event_j_censor = (event[i] == 1) & (event[j] == 0)
        mask_valid2 = mask_i_event_j_censor & (time[i] < time[j])
        if np.any(mask_valid2):
            concordant2 = np.sum(mask_valid2 & (risk[i] > risk[j]))
            total2 = np.sum(mask_valid2)
            concordant += concordant2
            total += total2
        # Case 3: i censored, j event
        mask_i_censor_j_event = (event[i] == 0) & (event[j] == 1)
        mask_valid3 = mask_i_censor_j_event & (time[j] < time[i])
        if np.any(mask_valid3):
            concordant3 = np.sum(mask_valid3 & (risk[j] > risk[i]))
            total3 = np.sum(mask_valid3)
            concordant += concordant3
            total += total3
    return concordant / total if total > 0 else 0.5


# Compute and print concordance index
c_index = concordance_index(time_val.values, y_val.values, val_pred)
print(f"Concordance Index: {c_index:.4f}")

# Retrain on full training set
num_medians_full, cat_categories_full, cat_cols_full, num_cols_full = fit_preprocess(
    train[feature_cols]
)
X_train_full_processed = transform(
    train[feature_cols],
    num_medians_full,
    cat_categories_full,
    cat_cols_full,
    num_cols_full,
)
X_test_processed = transform(
    test[feature_cols],
    num_medians_full,
    cat_categories_full,
    cat_cols_full,
    num_cols_full,
)

train_data_full = lgb.Dataset(
    X_train_full_processed, label=y, categorical_feature=cat_cols_full
)
model_full = lgb.train(
    params, train_data_full, num_boost_round=200, callbacks=[lgb.log_evaluation(-1)]
)

# Generate test predictions
test_pred = model_full.predict(X_test_processed)

# Save submission
submission = pd.DataFrame({"id": range(len(test)), "efs": test_pred})
submission.to_csv("./working/submission.csv", index=False)
