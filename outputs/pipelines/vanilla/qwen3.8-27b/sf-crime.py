import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/sf-crime/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/sf-crime/data/test.parquet"
TARGET = "Category"
RANDOM_STATE = 42


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Dates" in out.columns:
        dt = pd.to_datetime(out["Dates"], errors="coerce")
        out["year"] = dt.dt.year
        out["month"] = dt.dt.month
        out["day"] = dt.dt.day
        out["hour"] = dt.dt.hour
        out["minute"] = dt.dt.minute
        out["dayofweek"] = dt.dt.dayofweek
        out["dayofyear"] = dt.dt.dayofyear
        out["quarter"] = dt.dt.quarter
        out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        out["is_night"] = ((dt.dt.hour < 6) | (dt.dt.hour >= 18)).astype(int)
        out = out.drop(columns=["Dates"])

    drop_cols = [c for c in ["Address"] if c in out.columns]
    out = out.drop(columns=drop_cols)

    cat_cols = [c for c in ["Descript", "DayOfWeek", "PdDistrict", "Resolution"] if c in out.columns]
    for col in cat_cols:
        out[col] = out[col].fillna("missing").astype(str).astype("category")

    for col in out.columns:
        if out[col].dtype != "category":
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def make_model(n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)

    if len(train_df) == 0:
        pd.DataFrame({"id": np.arange(len(test_df)), "Category": 0.0}).to_csv("predictions.csv", index=False)
        return

    le = LabelEncoder()
    y_full = le.fit_transform(train_df[TARGET].astype(str))
    n_classes = len(le.classes_)

    if n_classes == 1:
        pd.DataFrame({"id": np.arange(len(test_df)), "Category": 0.0}).to_csv("predictions.csv", index=False)
        return

    X_full = preprocess(train_df.drop(columns=[TARGET]))
    X_test = preprocess(test_df)

    common_cols = [c for c in X_full.columns if c in X_test.columns]
    X_full = X_full[common_cols]
    X_test = X_test[common_cols]

    all_nan = [c for c in X_full.columns if X_full[c].isna().all()]
    if all_nan:
        X_full = X_full.drop(columns=all_nan)
        X_test = X_test.drop(columns=all_nan, errors="ignore")

    try:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_full, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full
        )
    except ValueError:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_full, y_full, test_size=0.2, random_state=RANDOM_STATE
        )

    model = make_model(3000)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0)],
    )

    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None or best_iter <= 0:
        best_iter = 3000

    final_model = make_model(int(best_iter))
    final_model.fit(X_full, y_full)

    proba = final_model.predict_proba(X_test)
    proba = np.asarray(proba, dtype=np.float64)

    if proba.shape[1] != n_classes:
        aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
        for i, cls in enumerate(final_model.classes_):
            idx = int(cls)
            if 0 <= idx < n_classes:
                aligned[:, idx] = proba[:, i]
        proba = aligned

    proba = np.clip(proba, 1e-15, None)
    proba /= proba.sum(axis=1, keepdims=True)

    y_pred = np.argmax(proba, axis=1).astype(np.int64)

    present = set(y_pred.tolist())
    missing = [c for c in range(n_classes) if c not in present]
    used = set()
    for cls in missing:
        candidates = [i for i in range(len(y_pred)) if i not in used]
        if not candidates:
            candidates = list(range(len(y_pred)))
        scores = proba[candidates, cls]
        best_local = int(np.argmax(scores))
        best_row = candidates[best_local]
        y_pred[best_row] = cls
        used.add(best_row)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df)),
            "Category": y_pred.astype(float),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()