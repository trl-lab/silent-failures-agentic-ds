import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype

TARGET = "number_health_problems"
WORK_DIR = "./working"
ID_COLS = {
    "id",
    "index",
    "row_id",
    "sample_id",
    "observation_id",
    "patient_id",
    "subject_id",
    "participant_id",
    "case_id",
    "unnamed: 0",
}


def cast_predictions(predictions, y_raw):
    predictions = np.asarray(predictions)
    if len(y_raw) == 0:
        return predictions

    try:
        numeric_y = pd.to_numeric(y_raw, errors="coerce")
    except Exception:
        numeric_y = pd.Series([np.nan] * len(y_raw))

    if numeric_y.notna().sum() == len(y_raw) and len(y_raw) > 0:
        try:
            values = numeric_y.to_numpy(dtype=float)
            if np.isfinite(values).all() and np.all(values % 1 == 0):
                return predictions.astype(int)
        except Exception:
            pass

    if is_numeric_dtype(y_raw):
        try:
            return predictions.astype(y_raw.dtype)
        except Exception:
            return predictions

    return predictions


def save_submission(predictions):
    os.makedirs(WORK_DIR, exist_ok=True)
    submission = pd.DataFrame(
        {"id": np.arange(len(predictions), dtype=int), TARGET: predictions}
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)


def make_model(cat_cols, n_estimators=300):
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=2,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        categorical_feature=cat_cols,
    )


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    for col in list(train.columns):
        if col.lower() in ID_COLS:
            train = train.drop(columns=[col])

    for col in list(test.columns):
        if col.lower() in ID_COLS:
            test = test.drop(columns=[col])

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    if len(train) == 0 or TARGET not in train.columns:
        print("Validation accuracy: 0.000000")
        save_submission(np.full(len(test), 0))
        return

    train = train.dropna(subset=[TARGET])
    y_raw = train[TARGET]
    X_train = train.drop(columns=[TARGET])
    X_test = test.copy()

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    try:
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
    except Exception:
        pass

    drop_cols = X_train.columns[X_train.isna().all()].tolist()
    if drop_cols:
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols)

    for col in X_train.columns:
        if is_numeric_dtype(X_train[col]) and not is_numeric_dtype(X_test[col]):
            try:
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            except Exception:
                X_test[col] = np.nan

    cat_cols = []
    for col in X_train.columns:
        if not is_numeric_dtype(X_train[col]):
            cat_cols.append(col)

    for col in cat_cols:
        X_train[col] = X_train[col].astype(object).fillna("missing").astype(str)
        X_test[col] = X_test[col].astype(object).fillna("missing").astype(str)
        train_categories = pd.Categorical(X_train[col]).categories
        X_train[col] = pd.Categorical(X_train[col], categories=train_categories)
        X_test[col] = pd.Categorical(X_test[col], categories=train_categories)

    if X_train.shape[1] == 0 or len(X_train) == 0:
        if len(y_raw) > 0:
            majority = y_raw.mode().iloc[0]
            val_acc = float((y_raw == majority).mean())
        else:
            majority = 0
            val_acc = 0.0

        save_submission(cast_predictions(np.full(len(X_test), majority), y_raw))
        print(f"Validation accuracy: {val_acc:.6f}")
        return

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    if y.nunique() == 1:
        val_acc = 1.0
        save_submission(
            cast_predictions(np.full(len(X_test), le.inverse_transform([0])[0]), y_raw)
        )
        print(f"Validation accuracy: {val_acc:.6f}")
        return

    min_class_count = int(pd.Series(y).value_counts().min())

    if len(y) >= 2 and min_class_count >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X_train, y))
    elif len(y) >= 2 and min_class_count >= 2:
        skf = StratifiedKFold(n_splits=min_class_count, shuffle=True, random_state=42)
        folds = list(skf.split(X_train, y))
    else:
        if len(y) >= 2:
            test_size = 0.2 if len(y) >= 10 else 0.5
            tr_idx, va_idx = train_test_split(
                np.arange(len(y)), test_size=test_size, random_state=42
            )
            folds = [(tr_idx, va_idx)]
        else:
            val_acc = 1.0
            save_submission(
                cast_predictions(
                    np.full(len(X_test), le.inverse_transform([0])[0]), y_raw
                )
            )
            print(f"Validation accuracy: {val_acc:.6f}")
            return

    val_preds = np.full(len(y), -1, dtype=int)
    best_iters = []
    use_early_stopping = len(y) >= 2 and min_class_count >= 2

    for tr_idx, va_idx in folds:
        y_tr = y[tr_idx]

        if len(y_tr) == 0:
            val_preds[va_idx] = -1
            continue

        if len(y_tr) < 2 or np.unique(y_tr).size < 2:
            val_preds[va_idx] = np.full(len(va_idx), y_tr[0])
            continue

        model = make_model(cat_cols)

        if use_early_stopping:
            model.fit(
                X_train.iloc[tr_idx],
                y[tr_idx],
                eval_set=[(X_train.iloc[va_idx], y[va_idx])],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )
        else:
            model.fit(X_train.iloc[tr_idx], y[tr_idx])

        val_preds[va_idx] = model.predict(X_train.iloc[va_idx])

        if use_early_stopping:
            best_iter = getattr(model, "best_iteration_", 0)
            if best_iter and best_iter > 0:
                best_iters.append(int(best_iter))

    val_acc = accuracy_score(y, val_preds)

    if len(X_test) == 0:
        save_submission(cast_predictions(np.array([]), y_raw))
        print(f"Validation accuracy: {val_acc:.6f}")
        return

    final_n_estimators = 300
    if best_iters:
        final_n_estimators = max(50, int(np.mean(best_iters)))

    final_model = make_model(cat_cols, n_estimators=final_n_estimators)
    final_model.fit(X_train, y)

    test_pred_idx = final_model.predict(X_test)
    test_pred = le.inverse_transform(test_pred_idx)

    save_submission(cast_predictions(test_pred, y_raw))
    print(f"Validation accuracy: {val_acc:.6f}")


if __name__ == "__main__":
    main()
