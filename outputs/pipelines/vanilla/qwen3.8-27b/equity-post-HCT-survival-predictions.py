import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from pandas.api.types import is_object_dtype, is_categorical_dtype, is_string_dtype
except Exception:
    def is_object_dtype(x):
        return False

    def is_categorical_dtype(x):
        return False

    def is_string_dtype(x):
        return False


TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/equity-post-HCT-survival-predictions/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/equity-post-HCT-survival-predictions/data/test.parquet"

TARGET = "efs"
TIME_COL = "efs_time"
ID_COL = "ID"
RANDOM_STATE = 42

MISSING_TOKENS = {
    "",
    "none",
    "n/a",
    "na",
    "null",
    "nan",
    "missing",
    "not available",
    "unknown",
    "unspecified",
}


def is_missing_token(value):
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_TOKENS
    return False


def clean_categorical_series(series):
    s = series.astype("string")
    s = s.mask(s.str.strip().str.lower().isin(MISSING_TOKENS), other=pd.NA)
    s = s.fillna("__MISSING__")
    return s.astype(str)


def to_numeric_series(series):
    if is_string_dtype(series) or is_object_dtype(series):
        s = series.astype("string")
        s = s.mask(s.str.strip().str.lower().isin(MISSING_TOKENS), other=pd.NA)
        s = pd.to_numeric(s, errors="coerce")
    else:
        s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def is_categorical_column(train_col, test_col):
    return (
        is_object_dtype(train_col)
        or is_categorical_dtype(train_col)
        or is_string_dtype(train_col)
        or is_object_dtype(test_col)
        or is_categorical_dtype(test_col)
        or is_string_dtype(test_col)
    )


def concordance_index_score(time, score, event):
    time = np.asarray(time, dtype=float)
    score = np.asarray(score, dtype=float)
    event = np.asarray(event, dtype=float)

    mask = np.isfinite(time) & np.isfinite(score) & np.isfinite(event)
    if mask.sum() < 2:
        return np.nan

    time = time[mask]
    score = score[mask]
    event = (event > 0.5).astype(float)

    if len(np.unique(time)) < 2 or not np.any(event > 0):
        return np.nan

    unique_scores, inverse = np.unique(score, return_inverse=True)
    n_scores = len(unique_scores)

    bit = [0] * (n_scores + 1)

    def bit_add(i, delta):
        while i <= n_scores:
            bit[i] += delta
            i += i & -i

    def bit_sum(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    order = np.argsort(time, kind="mergesort")
    time_sorted = time[order]
    score_sorted = score[order]
    event_sorted = event[order]
    inv_sorted = inverse[order]

    total_concordant = 0.0
    total_pairs = 0.0
    later_count = 0

    i = len(time_sorted) - 1
    while i >= 0:
        current_time = time_sorted[i]
        j = i
        while j >= 0 and time_sorted[j] == current_time:
            j -= 1

        start = j + 1

        for k in range(start, i + 1):
            if event_sorted[k] == 1:
                idx = int(inv_sorted[k]) + 1
                less = bit_sum(idx - 1)
                equal = bit_sum(idx) - less
                total_concordant += less + 0.5 * equal
                total_pairs += later_count

        for k in range(start, i + 1):
            bit_add(int(inv_sorted[k]) + 1, 1)
            later_count += 1

        i = j

    if total_pairs <= 0:
        return np.nan

    return float(total_concordant / total_pairs)


def make_models():
    models = {}

    if LGBMClassifier is not None:
        try:
            models["lgbm"] = LGBMClassifier(
                n_estimators=800,
                learning_rate=0.04,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        except Exception:
            pass

    if XGBClassifier is not None:
        try:
            models["xgb"] = XGBClassifier(
                n_estimators=800,
                learning_rate=0.04,
                max_depth=6,
                min_child_weight=3,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
            )
        except Exception:
            pass

    if CatBoostClassifier is not None:
        try:
            models["cat"] = CatBoostClassifier(
                iterations=800,
                learning_rate=0.04,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=RANDOM_STATE,
                verbose=0,
                allow_writing_files=False,
            )
        except Exception:
            pass

    return models


def write_predictions(n_test, predictions):
    predictions = np.asarray(predictions, dtype=float)
    if predictions.size != n_test:
        if predictions.size == 0:
            predictions = np.full(n_test, 0.5, dtype=float)
        else:
            predictions = np.resize(predictions, n_test)

    predictions = np.where(np.isfinite(predictions), predictions, 0.5)
    predictions = np.clip(predictions, 0.0, 1.0)

    pd.DataFrame(
        {
            "id": np.arange(n_test, dtype=int),
            "efs": predictions,
        }
    ).to_csv("predictions.csv", index=False)


def main():
    test_df = pd.read_parquet(TEST_PATH)
    n_test = len(test_df)

    try:
        train_df = pd.read_parquet(TRAIN_PATH)
    except Exception:
        train_df = pd.DataFrame()

    try:
        if train_df.empty or TARGET not in train_df.columns:
            write_predictions(n_test, np.full(n_test, 0.5, dtype=float))
            return

        train_df = train_df.dropna(subset=[TARGET]).copy()
        train_df[TARGET] = train_df[TARGET].astype(int)

        if TIME_COL in train_df.columns:
            train_df[TIME_COL] = pd.to_numeric(train_df[TIME_COL], errors="coerce")
        else:
            train_df[TIME_COL] = np.nan

        if ID_COL in train_df.columns:
            train_df = train_df.drop(columns=[ID_COL])
        if ID_COL in test_df.columns:
            test_df = test_df.drop(columns=[ID_COL])

        if TARGET in test_df.columns:
            test_df = test_df.drop(columns=[TARGET])

        exclude_cols = {ID_COL, TARGET, TIME_COL}
        feature_cols = [
            col
            for col in train_df.columns
            if col in test_df.columns and col not in exclude_cols
        ]

        categorical_cols = []
        numeric_cols = []

        for col in feature_cols:
            if is_categorical_column(train_df[col], test_df[col]):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

        for col in categorical_cols:
            train_df[col] = clean_categorical_series(train_df[col])
            test_df[col] = clean_categorical_series(test_df[col])

        for col in numeric_cols:
            train_df[col] = to_numeric_series(train_df[col])
            test_df[col] = to_numeric_series(test_df[col])

        categorical_cols = [
            col
            for col in categorical_cols
            if train_df[col].notna().sum() > 0 and train_df[col].nunique(dropna=True) > 1
        ]

        numeric_cols = [
            col
            for col in numeric_cols
            if train_df[col].notna().sum() > 0 and train_df[col].nunique(dropna=True) > 1
        ]

        if not categorical_cols and not numeric_cols:
            write_predictions(n_test, np.full(n_test, 0.5, dtype=float))
            return

        def encode_features(df, reference_dummy_columns=None):
            parts = []

            if numeric_cols:
                num = df[numeric_cols].astype(float)
                num = num.replace([np.inf, -np.inf], np.nan)
                parts.append(num)

            if categorical_cols:
                dummies = pd.get_dummies(
                    df[categorical_cols],
                    prefix=categorical_cols,
                    dtype=float,
                )
                if reference_dummy_columns is not None:
                    dummies = dummies.reindex(columns=reference_dummy_columns, fill_value=0.0)
                parts.append(dummies)

            X = pd.concat(parts, axis=1)
            X = X.loc[:, ~X.columns.duplicated()]
            X = X.replace([np.inf, -np.inf], np.nan)
            return X

        X_all = encode_features(train_df)

        if categorical_cols:
            reference_dummy_columns = pd.get_dummies(
                train_df[categorical_cols],
                prefix=categorical_cols,
                dtype=float,
            ).columns
        else:
            reference_dummy_columns = None

        X_test = encode_features(test_df, reference_dummy_columns)
        X_test = X_test.reindex(columns=X_all.columns, fill_value=0.0)

        y_all = train_df[TARGET].values
        time_all = train_df[TIME_COL].values

        if len(np.unique(y_all)) < 2:
            write_predictions(n_test, np.full(n_test, float(np.mean(y_all)), dtype=float))
            return

        models = make_models()
        if not models:
            write_predictions(n_test, np.full(n_test, float(np.mean(y_all)), dtype=float))
            return

        val_scores = {}
        if train_test_split is not None and len(X_all) >= 10:
            try:
                X_train, X_val, y_train, y_val, time_train, time_val = train_test_split(
                    X_all,
                    y_all,
                    time_all,
                    test_size=0.2,
                    random_state=RANDOM_STATE,
                    stratify=y_all,
                )

                for name, model in models.items():
                    try:
                        model.fit(X_train, y_train)
                        val_pred = model.predict_proba(X_val)[:, 1]
                        val_scores[name] = concordance_index_score(time_val, y_val, val_pred)
                    except Exception:
                        val_scores[name] = np.nan
            except Exception:
                val_scores = {name: np.nan for name in models}
        else:
            val_scores = {name: np.nan for name in models}

        test_predictions = []
        weights = []

        for name, model in models.items():
            try:
                model.fit(X_all, y_all)
                pred = model.predict_proba(X_test)[:, 1]

                score = val_scores.get(name, np.nan)
                if np.isfinite(score):
                    if score < 0.5:
                        pred = 1.0 - pred
                        score = 1.0 - score
                    weight = max(float(score), 0.05)
                else:
                    weight = 1.0

                test_predictions.append(pred)
                weights.append(weight)
            except Exception:
                continue

        if test_predictions:
            weights = np.asarray(weights, dtype=float)
            if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
                weights = np.ones(len(test_predictions), dtype=float)
            final_pred = np.average(np.vstack(test_predictions), axis=0, weights=weights)
        else:
            final_pred = np.full(n_test, float(np.mean(y_all)), dtype=float)

        write_predictions(n_test, final_pred)

    except Exception:
        write_predictions(n_test, np.full(n_test, 0.5, dtype=float))


if __name__ == "__main__":
    main()