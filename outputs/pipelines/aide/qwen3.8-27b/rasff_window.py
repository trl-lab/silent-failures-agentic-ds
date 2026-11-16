import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
TARGET = "risk_decision"
ID_LIKE = {"id", "index", "row_id", "sample_id", "uid", "identifier"}

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

if TARGET not in train.columns:
    raise ValueError(f"Target column {TARGET} not found in train data.")

y_raw = train[TARGET]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
classes = label_encoder.classes_

feature_cols = [
    c
    for c in train.columns
    if c in test.columns and c != TARGET and c.lower() not in ID_LIKE
]
if not feature_cols:
    feature_cols = [c for c in train.columns if c in test.columns and c != TARGET]

X_train = train[feature_cols].copy()
X_test = test[feature_cols].copy()


def _make_hashable(series: pd.Series) -> pd.Series:
    def _one(x):
        if isinstance(x, (list, tuple, dict, set)):
            return str(x)
        try:
            if pd.isna(x):
                return np.nan
        except (TypeError, ValueError):
            pass
        if isinstance(
            x, (str, bytes, int, float, bool, np.integer, np.floating, type(None))
        ):
            return x
        return str(x)

    return series.map(_one)


for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(
        X_train[c]
    ) or pd.api.types.is_datetime64_any_dtype(X_test[c]):
        X_train[c] = X_train[c].astype("int64")
        X_test[c] = X_test[c].astype("int64")
    elif pd.api.types.is_bool_dtype(X_train[c]) or pd.api.types.is_bool_dtype(
        X_test[c]
    ):
        X_train[c] = X_train[c].astype("int8")
        X_test[c] = X_test[c].astype("int8")
    elif pd.api.types.is_numeric_dtype(X_train[c]):
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    else:
        train_series = X_train[c]
        if train_series.notna().sum() > 0:
            numeric_train = pd.to_numeric(train_series, errors="coerce")
            if numeric_train.notna().sum() >= 0.8 * train_series.notna().sum():
                X_train[c] = numeric_train
                X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
            else:
                X_train[c] = _make_hashable(train_series).astype("category")
                X_test[c] = _make_hashable(X_test[c]).astype("category")
        else:
            X_train[c] = X_train[c].astype("category")
            X_test[c] = _make_hashable(X_test[c]).astype("category")

    if pd.api.types.is_numeric_dtype(X_train[c]):
        X_train[c] = X_train[c].replace([np.inf, -np.inf], np.nan)
        X_test[c] = X_test[c].replace([np.inf, -np.inf], np.nan)

drop_cols = []
for c in feature_cols:
    if X_train[c].isna().all() or X_train[c].nunique(dropna=True) <= 1:
        drop_cols.append(c)
if drop_cols:
    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

feature_cols = X_train.columns.tolist()
cat_cols = [c for c in feature_cols if pd.api.types.is_categorical_dtype(X_train[c])]

if len(feature_cols) == 0 or len(np.unique(y)) < 2:
    majority = classes[np.bincount(y).argmax()] if len(y) > 0 else 0
    submission = pd.DataFrame({"id": np.arange(len(test)), "risk_decision": majority})
    submission.to_csv("./working/submission.csv", index=False)
    print(f"CV Accuracy: {1.0 if len(np.unique(y)) < 2 else 0.0:.6f}")
    raise SystemExit(0)


def align_categories(
    df: pd.DataFrame, ref_df: pd.DataFrame, cat_columns: list
) -> pd.DataFrame:
    for c in cat_columns:
        if c in df.columns and c in ref_df.columns:
            ref_categories = ref_df[c].cat.categories
            df[c] = pd.Categorical(df[c], categories=ref_categories)
    return df


params = {
    "objective": "multiclass",
    "num_class": len(classes),
    "metric": "multi_error",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "is_unbalance": True,
    "seed": 42,
    "verbose": -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
best_iterations = []

for tr_idx, va_idx in skf.split(X_train, y):
    X_tr = X_train.iloc[tr_idx].copy()
    X_va = X_train.iloc[va_idx].copy()
    align_categories(X_va, X_tr, cat_cols)

    usable_cols = [c for c in X_tr.columns if X_tr[c].notna().any()]
    X_tr = X_tr[usable_cols]
    X_va = X_va[usable_cols]
    cat_cols_fold = [c for c in cat_cols if c in usable_cols]

    y_tr = y[tr_idx]
    y_va = y[va_idx]

    if len(np.unique(y_tr)) < 2:
        pred_class = np.full(len(y_va), y_tr[0])
        fold_scores.append(accuracy_score(y_va, pred_class))
        best_iterations.append(1)
        continue

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols_fold)
    dvalid = lgb.Dataset(
        X_va, label=y_va, categorical_feature=cat_cols_fold, reference=dtrain
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=-1),
        ],
    )
    best_iter = max(1, int(model.best_iteration))
    best_iterations.append(best_iter)

    val_pred = model.predict(X_va, num_iteration=best_iter)
    val_pred_labels = np.argmax(val_pred, axis=1)
    fold_scores.append(accuracy_score(y_va, val_pred_labels))

cv_accuracy = float(np.mean(fold_scores))
print(f"CV Accuracy: {cv_accuracy:.6f}")

best_iteration = max(1, int(np.mean(best_iterations))) if best_iterations else 1
X_tr_full = X_train.copy()
cat_cols_full = [c for c in cat_cols if X_tr_full[c].notna().any()]
dtrain_full = lgb.Dataset(X_tr_full, label=y, categorical_feature=cat_cols_full)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=best_iteration,
    callbacks=[lgb.log_evaluation(period=-1)],
)

X_te = X_test.copy()
align_categories(X_te, X_tr_full, cat_cols)
test_pred = final_model.predict(X_te, num_iteration=best_iteration)
test_pred_labels = np.argmax(test_pred, axis=1)
test_pred_raw = label_encoder.inverse_transform(test_pred_labels)

submission = pd.DataFrame({"id": np.arange(len(test)), "risk_decision": test_pred_raw})
submission.to_csv("./working/submission.csv", index=False)
