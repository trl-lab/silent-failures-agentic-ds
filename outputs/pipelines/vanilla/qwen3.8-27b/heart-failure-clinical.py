import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/heart-failure-clinical/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/heart-failure-clinical/data/test.parquet"
TARGET = "DEATH_EVENT"
RANDOM_STATE = 42


def save_predictions(n, values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(n, float(values))
    elif values.size == 1 and n != 1:
        values = np.full(n, float(values[0]))
    elif values.size != n:
        values = np.full(n, 0.5)
    values = np.clip(values, 0.0, 1.0)
    pd.DataFrame({"id": np.arange(n, dtype=int), TARGET: values}).to_csv("predictions.csv", index=False)


def prepare_features(df, target):
    df = df.copy()
    if target in df.columns:
        df = df.drop(columns=[target])

    cat_cols = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            df[col] = s.astype(int)
        elif pd.api.types.is_numeric_dtype(s):
            df[col] = s.astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            non_null = s.dropna()
            converted = None
            if len(non_null) > 0:
                try:
                    converted = pd.to_numeric(s.astype("object"), errors="coerce")
                except Exception:
                    converted = None

            if converted is not None and converted.notna().sum() >= 0.8 * len(non_null):
                df[col] = converted.replace([np.inf, -np.inf], np.nan)
            else:
                df[col] = s.astype("object")
                df[col] = df[col].where(df[col].notna(), "missing")
                cat_cols.append(col)

    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        df = df.drop(columns=all_nan_cols)

    cat_cols = [c for c in cat_cols if c in df.columns]
    return df, cat_cols


def encode_categoricals(train_df, test_df, cat_cols):
    for col in cat_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue

        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        categories = list(train_df[col].unique())
        mapping = {value: idx for idx, value in enumerate(categories)}

        train_df[col] = train_df[col].map(mapping).fillna(-1).astype(int)
        test_df[col] = test_df[col].map(mapping).fillna(-1).astype(int)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        save_predictions(len(test_df), 0.5)
        return

    train_df = train_df[train_df[TARGET].notna()]
    if len(train_df) == 0:
        save_predictions(len(test_df), 0.5)
        return

    y = train_df[TARGET].copy()
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors="coerce")

    valid = y.notna()
    train_df = train_df[valid]
    y = y[valid]

    if len(train_df) == 0:
        save_predictions(len(test_df), 0.5)
        return

    if y.nunique() == 2:
        classes = sorted(y.unique())
        y = y.map({classes[0]: 0, classes[1]: 1})
    else:
        y = y.astype(int)

    X, cat_cols = prepare_features(train_df.drop(columns=[TARGET]), TARGET)
    X_test, _ = prepare_features(test_df, TARGET)

    common_cols = [c for c in X.columns if c in X_test.columns]
    X = X[common_cols]
    X_test = X_test[common_cols]
    cat_cols = [c for c in cat_cols if c in common_cols]

    if len(X) == 0 or len(common_cols) == 0:
        save_predictions(len(X_test), 0.5)
        return

    encode_categoricals(X, X_test, cat_cols)

    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=all_nan_cols, errors="ignore")

    if X.shape[1] == 0:
        save_predictions(len(X_test), 0.5)
        return

    if y.nunique() < 2:
        save_predictions(len(X_test), float(y.iloc[0]) if len(y) > 0 else 0.5)
        return

    final_estimators = 1000
    if len(X) >= 20:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=y,
            )

            if y_val.nunique() == 2:
                val_model = LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=-1,
                    min_child_samples=20,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                )

                val_model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_val, y_val),
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
                )

                best_iter = getattr(val_model, "best_iteration_", 0)
                if best_iter and best_iter > 0:
                    final_estimators = max(100, int(best_iter) + 1)
        except Exception:
            pass

    model = LGBMClassifier(
        n_estimators=final_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X, y)

    if len(X_test) == 0:
        proba = np.array([], dtype=float)
    else:
        proba = model.predict_proba(X_test)[:, 1]

    save_predictions(len(X_test), proba)


if __name__ == "__main__":
    main()