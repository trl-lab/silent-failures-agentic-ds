import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Paths
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)
submission_path = os.path.join(output_dir, "submission.csv")

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "current_status"
y = train_df[target_col].copy()
if y.dtype == object or y.dtype.name == "category":
    # Convert binary targets to 0/1 integers
    unique_vals = sorted(y.dropna().unique())
    label_map = {val: i for i, val in enumerate(unique_vals)}
    y = y.map(label_map).astype(int)
else:
    y = y.astype(int)

X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Feature Preprocessing: Datetime extraction and categorical encoding preparation
all_dfs = [X, X_test]
for df in all_dfs:
    for col in list(df.columns):
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df.drop(columns=[col], inplace=True)
        elif df[col].dtype == "object" or df[col].dtype.name == "category":
            # Check if strings are parseable datetimes
            try:
                dt_series = pd.to_datetime(df[col], errors="coerce")
                if dt_series.notnull().mean() > 0.8:
                    df[f"{col}_year"] = dt_series.dt.year
                    df[f"{col}_month"] = dt_series.dt.month
                    df[f"{col}_day"] = dt_series.dt.day
                    df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                    df.drop(columns=[col], inplace=True)
            except Exception:
                pass

# Identify categorical features
cat_cols = []
for col in X.columns:
    if (
        X[col].dtype == "object"
        or X[col].dtype.name == "category"
        or X[col].dtype == "bool"
    ):
        cat_cols.append(col)
        X[col] = X[col].fillna("Missing").astype(str)
        X_test[col] = X_test[col].fillna("Missing").astype(str)
    else:
        # Numeric columns fillna with median
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

print(f"Starting 5-Fold Stratified CV with CatBoost...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]

    train_pool = Pool(X_train_f, y_train_f, cat_features=cat_cols)
    val_pool = Pool(X_val_f, y_val_f, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=100,
        early_stopping_rounds=50,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict_proba(val_pool)[:, 1]
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_val_f, val_pred)
    print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

    test_preds += model.predict_proba(test_pool)[:, 1] / n_splits

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall Out-of-Fold ROC-AUC: {overall_auc:.5f}")

# Generate Submission File
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "current_status": test_preds}
)
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission.shape}")
