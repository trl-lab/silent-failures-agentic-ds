import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "isFraud"
ID_COL = "TransactionID"

train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

y = train[TARGET].astype(int).values

# Keep only columns present in both train and test, excluding id/target.
feature_cols = [
    col for col in train.columns if col not in (ID_COL, TARGET) and col in test.columns
]

drop_cols = set()

# Drop columns that are all missing in train.
for col in feature_cols:
    if train[col].isna().all():
        drop_cols.add(col)

feature_cols = [col for col in feature_cols if col not in drop_cols]

cat_cols = []
num_cols = []

# Identify and filter categorical/numeric features.
for col in feature_cols:
    is_cat_dtype = train[col].dtype == object or train[col].dtype.name == "category"
    if is_cat_dtype:
        nunique = train[col].nunique(dropna=True)
        if 1 < nunique <= 1000:
            cat_cols.append(col)
        else:
            drop_cols.add(col)
    else:
        nunique = train[col].nunique(dropna=True)
        if nunique <= 1:
            drop_cols.add(col)
        else:
            num_cols.append(col)

feature_cols = [col for col in feature_cols if col not in drop_cols]
cat_cols = [col for col in cat_cols if col in feature_cols]

# Encode low-cardinality categoricals using train-only mappings.
for col in cat_cols:
    unique_values = train[col].dropna().unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    unknown_code = len(mapping)

    train[col] = train[col].map(mapping).fillna(unknown_code).astype("int32")
    test[col] = test[col].map(mapping).fillna(unknown_code).astype("int32")

# Coerce remaining features to numeric and remove infinities.
for col in feature_cols:
    if col in cat_cols:
        continue
    train[col] = pd.to_numeric(train[col], errors="coerce")
    test[col] = pd.to_numeric(test[col], errors="coerce")
    train[col] = train[col].replace([np.inf, -np.inf], np.nan)
    test[col] = test[col].replace([np.inf, -np.inf], np.nan)

# Final safety check: only numeric dtypes should remain.
feature_cols = [
    col for col in feature_cols if pd.api.types.is_numeric_dtype(train[col])
]
cat_cols = [col for col in cat_cols if col in feature_cols]

X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()

# Use memory-efficient dtypes.
for col in feature_cols:
    if col in cat_cols:
        X_train[col] = X_train[col].astype("int32")
        X_test[col] = X_test[col].astype("int32")
    else:
        X_train[col] = X_train[col].astype("float32")
        X_test[col] = X_test[col].astype("float32")

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "seed": 42,
    "verbose": -1,
    "num_threads": os.cpu_count() or 4,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
best_iterations = []
num_boost_round = 1000

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y), start=1):
    X_tr = X_train.iloc[train_idx]
    y_tr = y[train_idx]
    X_va = X_train.iloc[val_idx]
    y_va = y[val_idx]

    train_set = lgb.Dataset(
        X_tr,
        label=y_tr,
        categorical_feature=cat_cols,
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        X_va,
        label=y_va,
        categorical_feature=cat_cols,
        reference=train_set,
        free_raw_data=True,
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    best_iter = model.best_iteration if model.best_iteration > 0 else num_boost_round
    best_iterations.append(best_iter)

    val_pred = model.predict(X_va, num_iteration=best_iter)
    fold_auc = roc_auc_score(y_va, val_pred)
    aucs.append(fold_auc)

    print(f"Fold {fold}: ROC AUC = {fold_auc:.6f}, best_iteration = {best_iter}")

mean_auc = float(np.mean(aucs))
std_auc = float(np.std(aucs))
print(f"CV ROC AUC: {mean_auc:.6f} (+/- {std_auc:.6f})")

# Train final model on all training data using the average early-stopping iteration.
final_num_boost = max(1, int(np.mean(best_iterations)))
final_train_set = lgb.Dataset(
    X_train,
    label=y,
    categorical_feature=cat_cols,
    free_raw_data=True,
)

final_model = lgb.train(
    params,
    final_train_set,
    num_boost_round=final_num_boost,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_test, num_iteration=final_num_boost)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "isFraud": test_pred.astype(float),
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

print(f"Saved submission to {os.path.join(WORK_DIR, 'submission.csv')}")
