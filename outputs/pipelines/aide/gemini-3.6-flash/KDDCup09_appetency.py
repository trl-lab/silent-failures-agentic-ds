import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

# Load train and test datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate target and input features
target_col = "APPETENCY"
y_raw = train[target_col].copy()

# Map target robustly to binary integers (0 and 1)
if y_raw.dtype == object or str(y_raw.dtype) == "category":
    y = (
        y_raw.astype(str)
        .str.strip()
        .apply(lambda x: 1 if x in ["1", "1.0"] else 0)
        .values
    )
else:
    y = (y_raw > 0).astype(int).values

X = train.drop(columns=[target_col]).copy()
X_test = test.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Align test set column order
X_test = X_test[X.columns]

# Feature Engineering: Compute row-wise missing value counts
X["n_missing"] = X.isnull().sum(axis=1)
X_test["n_missing"] = X_test.isnull().sum(axis=1)

# Categorise columns into categorical and numerical
cat_cols = []
num_cols = []

for col in X.columns:
    if col == "n_missing":
        num_cols.append(col)
        continue
    if X[col].dtype == "object" or str(X[col].dtype) == "category":
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Fill missing categorical values and add frequency encoding features
for col in cat_cols:
    X[col] = X[col].astype(str).fillna("missing")
    X_test[col] = X_test[col].astype(str).fillna("missing")

    # Combined frequency encoding
    freq_map = pd.concat([X[col], X_test[col]]).value_counts(normalize=True)
    X[f"{col}_freq"] = X[col].map(freq_map).astype(float)
    X_test[f"{col}_freq"] = X_test[col].map(freq_map).astype(float)

# Stratified K-Fold validation set up
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)
    test_pool = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=100,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
    test_preds += model.predict_proba(test_pool)[:, 1] / skf.n_splits

# Compute overall OOF ROC AUC metric
oof_auc = roc_auc_score(y, oof_preds)
print(f"OOF ROC AUC: {oof_auc:.5f}")

# Save final predictions to submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "APPETENCY": test_preds})
submission.to_csv("./working/submission.csv", index=False)
