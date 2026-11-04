import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

os.makedirs("./working", exist_ok=True)

TARGET = "outcomes"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

y = train[TARGET].copy()
train = train.drop(columns=[TARGET])

id_like = [
    c
    for c in train.columns
    if str(c).strip().lower()
    in {"id", "index", "row_id", "rowid", "uid", "uuid", "identifier"}
]
train = train.drop(columns=id_like, errors="ignore")
test = test.drop(columns=id_like, errors="ignore")

feature_cols = [c for c in train.columns if c in test.columns]
train = train[feature_cols]
test = test[feature_cols]

valid_mask = y.notna()
y = y.loc[valid_mask]
train = train.loc[valid_mask]


def save_submission(outcomes):
    outcomes = np.asarray(outcomes)
    if outcomes.ndim == 0:
        outcomes = np.full(len(test), outcomes.item(), dtype=outcomes.dtype)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "outcomes": outcomes,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if len(train) == 0:
    save_submission("")
    print("5-Fold CV Accuracy: 0.000000")
    raise SystemExit


def is_mostly_numeric(series: pd.Series) -> bool:
    s = series.dropna().head(1000)
    if s.empty:
        return False
    converted = pd.to_numeric(s, errors="coerce")
    return converted.notna().mean() > 0.9


def datetime_to_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    return s.map(lambda x: x.timestamp() if pd.notna(x) else np.nan).astype("float64")


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in train_df.columns:
        if col not in test_df.columns:
            continue

        train_is_dt = pd.api.types.is_datetime64_any_dtype(train_df[col])
        test_is_dt = pd.api.types.is_datetime64_any_dtype(test_df[col])

        if train_is_dt or test_is_dt:
            train_df[col] = datetime_to_numeric(train_df[col])
            test_df[col] = datetime_to_numeric(test_df[col])
            continue

        train_is_obj = train_df[col].dtype == object
        test_is_obj = test_df[col].dtype == object

        if train_is_obj or test_is_obj:
            if is_mostly_numeric(train_df[col]) or is_mostly_numeric(test_df[col]):
                train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
                test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
            else:
                if train_df[col].dtype != object:
                    train_df[col] = train_df[col].astype(str)
                if test_df[col].dtype != object:
                    test_df[col] = test_df[col].astype(str)

                nunique = train_df[col].nunique(dropna=True)

                if nunique <= 100:
                    train_df[col] = train_df[col].astype("category")
                    test_df[col] = pd.Categorical(
                        test_df[col], categories=train_df[col].cat.categories
                    )
                else:
                    freq = train_df[col].value_counts(dropna=False)
                    train_df[col] = train_df[col].map(freq).astype("float64")
                    test_df[col] = test_df[col].map(freq).fillna(0.0).astype("float64")
            continue

        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    drop_cols = []
    for col in train_df.columns:
        if train_df[col].isna().all():
            drop_cols.append(col)
        elif train_df[col].dtype.name == "category":
            if train_df[col].nunique(dropna=True) <= 1:
                drop_cols.append(col)
        else:
            if train_df[col].nunique(dropna=True) <= 1:
                drop_cols.append(col)

    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    return train_df, test_df


train, test = preprocess(train, test)

if train.shape[1] == 0:
    most_common = y.value_counts().index[0] if len(y) else ""
    save_submission(most_common)
    print("5-Fold CV Accuracy: 0.000000")
    raise SystemExit

X = train

label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
n_classes = len(label_encoder.classes_)

if n_classes == 1:
    save_submission(label_encoder.classes_[0])
    print("5-Fold CV Accuracy: 1.000000")
    raise SystemExit

cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]


def make_params(overrides: dict) -> dict:
    params = {
        "objective": "multiclass" if n_classes > 2 else "binary",
        "metric": "multi_error" if n_classes > 2 else "binary_error",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
        "num_threads": 4,
    }
    params.update(overrides)
    if n_classes > 2:
        params["num_class"] = n_classes
    return params


def predict_labels(
    model: lgb.Booster, data: pd.DataFrame, num_iteration=None
) -> np.ndarray:
    proba = model.predict(data, num_iteration=num_iteration)
    if proba.ndim == 1:
        return (proba >= 0.5).astype(int)
    return np.argmax(proba, axis=1)


try:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(skf.split(X, y_enc))
except Exception:
    n_splits = min(5, max(2, len(X)))
    if n_splits < 2:
        fold_indices = [(np.arange(len(X)), np.arange(len(X)))]
    else:
        try:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_indices = list(kf.split(X, y_enc))
        except Exception:
            fold_indices = [(np.arange(len(X)), np.arange(len(X)))]


configs = [
    {},
    {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "lambda_l1": 0.5,
        "lambda_l2": 2.0,
    },
    {
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "lambda_l1": 2.0,
        "lambda_l2": 2.0,
    },
    {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
    },
    {
        "learning_rate": 0.02,
        "num_leaves": 31,
        "min_child_samples": 50,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.6,
        "lambda_l1": 3.0,
        "lambda_l2": 5.0,
    },
    {
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 20,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "lambda_l1": 1.0,
        "lambda_l2": 3.0,
    },
]

callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
results = []

for i, overrides in enumerate(configs):
    params = make_params(overrides)
    val_accuracies = []
    best_iterations = []

    for tr_idx, va_idx in fold_indices:
        if len(va_idx) == 0:
            continue

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_enc[tr_idx], y_enc[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dvalid = lgb.Dataset(
            X_va,
            label=y_va,
            reference=dtrain,
            categorical_feature=cat_cols,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        best_iter = model.best_iteration
        if best_iter is None or best_iter <= 0:
            best_iter = 1
        best_iterations.append(int(best_iter))

        pred = predict_labels(model, X_va, num_iteration=best_iter)
        val_accuracies.append(accuracy_score(y_va, pred))

    mean_acc = float(np.mean(val_accuracies)) if val_accuracies else 0.0
    results.append(
        {
            "mean_acc": mean_acc,
            "overrides": overrides,
            "params": params,
            "best_iterations": best_iterations,
        }
    )
    print(f"Config {i}: CV accuracy={mean_acc:.6f}, best_iterations={best_iterations}")

best_result = max(results, key=lambda r: r["mean_acc"])
cv_accuracy = best_result["mean_acc"]
print(f"5-Fold CV Accuracy: {cv_accuracy:.6f}")
print(f"Selected overrides: {best_result['overrides']}")

if best_result["best_iterations"]:
    final_best_iter = int(np.median(best_result["best_iterations"]))
else:
    final_best_iter = 500
final_best_iter = max(1, final_best_iter)

dtrain_full = lgb.Dataset(X, label=y_enc, categorical_feature=cat_cols)
final_model = lgb.train(
    best_result["params"],
    dtrain_full,
    num_boost_round=final_best_iter,
    callbacks=[lgb.log_evaluation(-1)],
)

if len(test) == 0:
    test_outcomes = np.array([], dtype=label_encoder.classes_.dtype)
else:
    test_pred = predict_labels(final_model, test, num_iteration=final_best_iter)
    test_outcomes = label_encoder.inverse_transform(test_pred)

save_submission(test_outcomes)
