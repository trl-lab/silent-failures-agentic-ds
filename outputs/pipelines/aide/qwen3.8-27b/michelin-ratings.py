import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "awardFor"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Keep only rows with a valid target.
train = train.dropna(subset=[TARGET])
y_raw = train[TARGET].copy()
train = train.drop(columns=[TARGET])
if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

# Use only columns present in both train and test to avoid shape mismatches.
common_cols = [c for c in train.columns if c in test.columns]
train = train[common_cols].copy()
test = test[common_cols].copy()

# Drop obvious identifier-like columns.
id_like = {
    "id",
    "index",
    "idx",
    "row_id",
    "rowid",
    "uuid",
    "uid",
    "key",
    "identifier",
    "identifier_id",
    "record_id",
}
drop_ids = [c for c in train.columns if str(c).lower() in id_like]
train = train.drop(columns=drop_ids, errors="ignore")
test = test.drop(columns=drop_ids, errors="ignore")

# Drop columns that are entirely missing or constant in the training data.
for col in list(train.columns):
    if train[col].isna().all() or train[col].nunique(dropna=True) <= 1:
        train = train.drop(columns=[col])
        test = test.drop(columns=[col], errors="ignore")

# Separate numeric and categorical columns.
num_cols = []
cat_cols = []

for col in train.columns:
    if pd.api.types.is_datetime64_any_dtype(train[col]):
        train[col] = pd.to_datetime(train[col], errors="coerce")
        test[col] = pd.to_datetime(test[col], errors="coerce")
        train[col] = train[col].astype("int64") // 86_400_000_000_000
        test[col] = test[col].astype("int64") // 86_400_000_000_000
        train[col] = train[col].replace({-9223372036854775808: np.nan})
        test[col] = test[col].replace({-9223372036854775808: np.nan})
        num_cols.append(col)
    elif pd.api.types.is_numeric_dtype(train[col]):
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
        train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)
        num_cols.append(col)
    else:
        cat_cols.append(col)

# Drop extremely high-cardinality categorical columns that are likely identifiers.
if len(train) > 100:
    for col in list(cat_cols):
        if train[col].nunique(dropna=True) > 0.99 * len(train):
            train = train.drop(columns=[col])
            test = test.drop(columns=[col], errors="ignore")
            cat_cols.remove(col)

# Encode target.
classes = np.unique(y_raw)
class_to_idx = {c: i for i, c in enumerate(classes)}
y = y_raw.map(class_to_idx).astype(int)


def encode_categoricals(df, categorical_columns, categories=None, fit=False):
    """
    Label-encode categorical columns.
    Unknown categories become -1, which LightGBM treats as missing.
    """
    X = df.copy()
    if categories is None:
        categories = {}

    for col in categorical_columns:
        if col not in X.columns:
            continue

        s = X[col]

        if fit:
            non_null = s.dropna()
            cats = sorted(non_null.astype(str).unique())
            categories[col] = {str(v): i for i, v in enumerate(cats)}

        mapping = categories.get(col, {})
        X[col] = s.map(lambda x: mapping.get(str(x), -1) if pd.notna(x) else -1)
        X[col] = X[col].astype("int32")

    return X, categories


def filter_constant_features(X_train, X_val, categorical_columns):
    """
    Remove columns that are constant in the training fold.
    """
    keep = []
    for col in X_train.columns:
        if X_train[col].nunique(dropna=True) > 1:
            keep.append(col)

    X_train = X_train[keep]
    X_val = X_val[keep]
    cat_fold = [c for c in keep if c in categorical_columns]
    return X_train, X_val, cat_fold


# If there is only one target class, the problem is trivial.
if len(classes) == 1:
    print("Validation Accuracy: 1.000000")
    submission = pd.DataFrame(
        {"id": np.arange(len(test), dtype=int), TARGET: classes[0]}
    )
    submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
    raise SystemExit

# Choose objective/metric.
if len(classes) == 2:
    objective = "binary"
    metric = "binary_logloss"
else:
    objective = "multiclass"
    metric = "multi_logloss"

params = {
    "objective": objective,
    "metric": metric,
    "num_class": len(classes),
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "force_row_wise": True,
}

# Stratified 5-fold CV when possible; otherwise fall back to one stratified split.
class_counts = np.bincount(y)
min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0

if len(train) >= 10 and min_class_count >= 2:
    n_splits = min(5, len(train), min_class_count)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(train, y))
else:
    tr_idx, va_idx = train_test_split(
        np.arange(len(train)),
        test_size=0.2,
        random_state=42,
        stratify=y if min_class_count >= 2 else None,
    )
    splits = [(tr_idx, va_idx)]

fold_accs = []
best_iterations = []

for tr_idx, va_idx in splits:
    X_tr_raw = train.iloc[tr_idx]
    X_va_raw = train.iloc[va_idx]
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    X_tr_enc, fold_categories = encode_categoricals(
        X_tr_raw, cat_cols, categories=None, fit=True
    )
    X_va_enc, _ = encode_categoricals(
        X_va_raw, cat_cols, categories=fold_categories, fit=False
    )

    X_tr_enc, X_va_enc, cat_fold = filter_constant_features(
        X_tr_enc, X_va_enc, cat_cols
    )

    if X_tr_enc.shape[1] == 0:
        pred_va = np.full(len(y_va), int(y_tr.mode().iloc[0]))
        fold_accs.append(accuracy_score(y_va, pred_va))
        best_iterations.append(1)
        continue

    dtrain = lgb.Dataset(X_tr_enc, label=y_tr, categorical_feature=cat_fold)
    dval = lgb.Dataset(X_va_enc, reference=dtrain, categorical_feature=cat_fold)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        valid_names=["val"],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    best_iterations.append(max(1, int(model.best_iteration)))

    if len(classes) == 2:
        prob = model.predict(X_va_enc, num_iteration=model.best_iteration)
        pred_va = (prob > 0.5).astype(int)
    else:
        prob = model.predict(X_va_enc, num_iteration=model.best_iteration)
        pred_va = np.argmax(prob, axis=1)

    fold_accs.append(accuracy_score(y_va, pred_va))

mean_acc = float(np.mean(fold_accs))
std_acc = float(np.std(fold_accs))

print(f"Validation Accuracy: {mean_acc:.6f}")
print(f"Validation Accuracy Std: {std_acc:.6f}")

# Refit on the full training set using the average best iteration from CV.
X_train_enc, full_categories = encode_categoricals(
    train, cat_cols, categories=None, fit=True
)
X_test_enc, _ = encode_categoricals(
    test, cat_cols, categories=full_categories, fit=False
)

X_train_enc, X_test_enc, cat_final = filter_constant_features(
    X_train_enc, X_test_enc, cat_cols
)

if X_train_enc.shape[1] == 0:
    pred_idx = np.full(len(test), int(y.mode().iloc[0]))
else:
    final_rounds = (
        int(np.clip(np.mean(best_iterations), 50, 1000)) if best_iterations else 500
    )
    dtrain_final = lgb.Dataset(X_train_enc, label=y, categorical_feature=cat_final)

    final_model = lgb.train(
        params,
        dtrain_final,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    if len(classes) == 2:
        prob = final_model.predict(X_test_enc)
        pred_idx = (prob > 0.5).astype(int)
    else:
        prob = final_model.predict(X_test_enc)
        pred_idx = np.argmax(prob, axis=1)

pred_labels = classes[pred_idx]

submission = pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: pred_labels})

submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
