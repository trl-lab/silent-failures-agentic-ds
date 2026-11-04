import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Set random seed for reproducibility
RANDOM_STATE = 42
N_SPLITS = 5

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and features
target_col = "APPETENCY"
features = [c for c in train_df.columns if c != target_col]

# Process target: map {-1, 1} or other binary formats to {0, 1}
y_raw = train_df[target_col]
if set(y_raw.unique()).issubset({-1, 1}):
    y = (y_raw == 1).astype(int).values
elif set(y_raw.unique()).issubset({0, 1}):
    y = y_raw.astype(int).values
else:
    # In case of categorical/string labels
    classes = np.sort(y_raw.unique())
    y = (y_raw == classes[-1]).astype(int).values

X = train_df[features].copy()
X_test = test_df[features].copy()

# Identify categorical and numerical features
categorical_cols = []
numerical_cols = []

for col in features:
    if X[col].dtype == "object" or X[col].dtype.name == "category":
        categorical_cols.append(col)
        # Convert to string and fill missing values for CatBoost
        X[col] = X[col].astype(str).fillna("missing")
        X_test[col] = X_test[col].astype(str).fillna("missing")
    else:
        numerical_cols.append(col)
        # Ensure numerical types are float/int
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

cat_params = {
    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 6,
    "eval_metric": "AUC",
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "task_type": "CPU",
    "early_stopping_rounds": 50,
}

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
    val_pool = Pool(X_val, y_val, cat_features=categorical_cols)
    test_pool = Pool(X_test, cat_features=categorical_cols)

    model = CatBoostClassifier(**cat_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict_proba(val_pool)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(test_pool)[:, 1] / N_SPLITS

    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold + 1} ROC AUC: {fold_auc:.5f}")

overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall Out-of-Fold ROC AUC: {overall_auc:.5f}")

# Prepare submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "APPETENCY": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
