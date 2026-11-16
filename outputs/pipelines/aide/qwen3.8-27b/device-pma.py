import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

os.makedirs("./working", exist_ok=True)

TARGET = "decision_code"
SEED = 42
N_SPLITS = 5
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 50

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train.parquet")

train = train.dropna(subset=[TARGET]).reset_index(drop=True)

# Convert target to binary 0/1 if needed.
if pd.api.types.is_bool_dtype(train[TARGET]) or np.issubdtype(
    train[TARGET].dtype, np.number
):
    y = train[TARGET].astype(float)
else:
    classes = sorted(train[TARGET].dropna().unique().tolist())
    if len(classes) != 2:
        # Fallback for unexpected non-binary targets: use the most common class as 0 and others as 1.
        most_common = train[TARGET].mode().iloc[0]
        y = (train[TARGET] != most_common).astype(float)
    else:
        mapping = {classes[0]: 0.0, classes[1]: 1.0}
        y = train[TARGET].map(mapping).astype(float)

# Identify feature columns.
id_like = {
    "id",
    "index",
    "row_id",
    "rowid",
    "uuid",
    "identifier",
    "patient_id",
    "device_id",
}
feature_cols = [
    col for col in train.columns if col != TARGET and col.lower() not in id_like
]

X_all = train[feature_cols].copy()
X_test = test.reindex(columns=feature_cols).copy()
if TARGET in X_test.columns:
    X_test = X_test.drop(columns=[TARGET])

cat_cols = []
drop_cols = []

for col in feature_cols:
    s_train = X_all[col]

    # Drop features with no usable information in training data.
    if s_train.isna().all():
        drop_cols.append(col)
        continue
    if s_train.dropna().nunique() <= 1:
        drop_cols.append(col)
        continue

    # Convert datetime features to numeric days since the minimum observed date.
    if pd.api.types.is_datetime64_any_dtype(s_train):

        def datetime_to_days(series: pd.Series) -> pd.Series:
            series = pd.to_datetime(series, errors="coerce")
            if not series.notna().any():
                return pd.Series(np.nan, index=series.index, dtype=float)
            min_date = series.min()
            return ((series - min_date).dt.total_seconds() / 86400.0).astype(float)

        X_all[col] = datetime_to_days(s_train)
        X_test[col] = datetime_to_days(X_test[col])
        continue

    # Keep numeric features numeric and replace infinities with NaN.
    if pd.api.types.is_numeric_dtype(s_train):
        X_all[col] = pd.to_numeric(X_all[col], errors="coerce").astype(float)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").astype(float)
        X_all[col] = X_all[col].replace([np.inf, -np.inf], np.nan)
        X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
        continue

    # Handle object/string/categorical features.
    if (
        pd.api.types.is_object_dtype(s_train)
        or pd.api.types.is_string_dtype(s_train)
        or pd.api.types.is_categorical_dtype(s_train)
    ):
        s_train_obj = s_train.astype("object")
        s_test_obj = X_test[col].astype("object")
        n_unique = s_train_obj.dropna().nunique()

        # High-cardinality text/categorical features are frequency-encoded using train data only.
        if n_unique > 100:
            train_str = s_train_obj.fillna("__missing__").astype(str)
            test_str = s_test_obj.fillna("__missing__").astype(str)
            freq = train_str.value_counts()
            X_all[col] = train_str.map(freq).astype(float)
            X_test[col] = test_str.map(freq).fillna(0.0).astype(float)
            continue

        # Low-cardinality features are converted to aligned pandas categories.
        train_values = s_train_obj.astype("object")
        test_values = s_test_obj.astype("object")
        train_categories = pd.unique(train_values.dropna())

        X_all[col] = pd.Categorical(train_values, categories=train_categories)
        X_test[col] = pd.Categorical(test_values, categories=train_categories)
        cat_cols.append(col)
        continue

    # Fallback: convert unknown types to strings and frequency-encode if needed.
    s_train_obj = s_train.astype("object")
    s_test_obj = X_test[col].astype("object")
    train_str = s_train_obj.fillna("__missing__").astype(str)
    test_str = s_test_obj.fillna("__missing__").astype(str)
    freq = train_str.value_counts()
    X_all[col] = train_str.map(freq).astype(float)
    X_test[col] = test_str.map(freq).fillna(0.0).astype(float)

X_all = X_all.drop(columns=drop_cols, errors="ignore")
X_test = X_test.drop(columns=drop_cols, errors="ignore")

# Remove any columns that became constant after preprocessing.
for col in X_all.columns:
    if X_all[col].dropna().nunique() <= 1:
        drop_cols.append(col)
X_all = X_all.drop(columns=drop_cols, errors="ignore")
X_test = X_test.drop(columns=drop_cols, errors="ignore")

cat_cols = [col for col in cat_cols if col in X_all.columns]

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 20,
    "seed": SEED,
    "num_threads": -1,
    "verbosity": -1,
    "max_bin": 255,
}

# Fallback if no usable features remain.
if X_all.shape[1] == 0 or y.nunique() < 2:
    cv_auc = 0.5
    test_pred = np.full(len(X_test), 0.5, dtype=float)
else:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_aucs = []
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_all, y), start=1):
        X_train_fold = X_all.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X_all.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        dtrain = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            X_valid_fold,
            label=y_valid_fold,
            categorical_feature=cat_cols,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        valid_pred = model.predict(X_valid_fold, num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y_valid_fold, valid_pred)
        fold_aucs.append(fold_auc)
        best_iterations.append(max(1, int(model.best_iteration)))
        print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

    cv_auc = float(np.mean(fold_aucs))
    final_boost_rounds = int(np.mean(best_iterations))

    dtrain_full = lgb.Dataset(
        X_all,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_boost_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )
    test_pred = final_model.predict(X_test, num_iteration=final_boost_rounds)
    test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "decision_code": test_pred.astype(float),
    }
)
submission.to_csv("./working/submission.csv", index=False)

print(f"CV ROC AUC: {cv_auc:.6f}")
