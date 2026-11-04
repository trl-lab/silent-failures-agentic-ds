import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "Overall"
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def drop_id_columns(df):
    id_names = {"id", "index", "idx", "row_id", "rowid"}
    cols = [c for c in df.columns if str(c).lower() in id_names]
    return df.drop(columns=cols, errors="ignore")


train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

train = drop_id_columns(train)
test = drop_id_columns(test)

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

y = pd.to_numeric(train[TARGET], errors="coerce")
y = y.replace([np.inf, -np.inf], np.nan)
mask = y.notna()
train = train.loc[mask].copy()
y = y.loc[mask].copy()
X = train.drop(columns=[TARGET]).copy()

common_cols = [c for c in X.columns if c in test.columns]
X = X[common_cols].copy()
test = test[common_cols].copy()


def to_numeric_safe(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(np.nan, index=s.index)


def normalize_numeric_series(s):
    s = to_numeric_safe(s)
    if pd.api.types.is_integer_dtype(s):
        if s.isna().any():
            return s.astype("float64")
        if pd.api.types.is_unsigned_integer_dtype(s):
            return s.astype("float64")
        return s.astype("int64", errors="ignore")
    if pd.api.types.is_float_dtype(s):
        return s.astype("float64", errors="ignore")
    if pd.api.types.is_bool_dtype(s):
        return s.astype("float64")
    return s.astype("float64", errors="ignore")


def safe_str(x):
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return str(x)


def is_datetime_like(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    dtype_str = str(s.dtype).lower()
    return (
        dtype_str.startswith("timestamp")
        or dtype_str.startswith("date")
        or dtype_str.startswith("time")
        or dtype_str.startswith("duration")
        or dtype_str.startswith("month_day")
    )


def convert_datetime(s):
    if not pd.api.types.is_datetime64_any_dtype(s):
        try:
            s = pd.to_datetime(s, errors="coerce")
        except Exception:
            s = pd.Series(pd.NaT, index=s.index)

    if pd.api.types.is_datetime64_any_dtype(s):
        s = s.astype("int64", errors="ignore")
        if pd.api.types.is_datetime64_any_dtype(s):
            s = s.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    if pd.api.types.is_integer_dtype(s):
        min_int = np.iinfo(np.int64).min
        s = s.where(s != min_int, np.nan)
    elif not pd.api.types.is_numeric_dtype(s):
        s = to_numeric_safe(s)

    return s


def supported_dtype(s):
    return (
        pd.api.types.is_numeric_dtype(s)
        or is_datetime_like(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or pd.api.types.is_bool_dtype(s)
    )


def preprocess_features(fit_df, dataframes):
    cols = fit_df.columns.tolist()
    fit = fit_df[cols].copy()
    transformed = [df[cols].copy() for df in dataframes]
    active_cat = []

    for c in cols:
        fit_col = fit[c]

        if not supported_dtype(fit_col):
            fit[c] = fit_col.astype(object)
            for df in transformed:
                df[c] = df[c].astype(object)
            fit_col = fit[c]

        if (
            pd.api.types.is_categorical_dtype(fit_col)
            or pd.api.types.is_object_dtype(fit_col)
            or pd.api.types.is_string_dtype(fit_col)
        ):
            converted = to_numeric_safe(fit_col)
            non_null = fit_col.notna().sum()

            if non_null > 0 and converted.notna().sum() >= 0.9 * non_null:
                fit[c] = normalize_numeric_series(converted)
                for df in transformed:
                    df[c] = normalize_numeric_series(df[c])
            else:
                fit[c] = fit[c].map(safe_str)
                for df in transformed:
                    df[c] = df[c].map(safe_str)

                cats = list(fit[c].dropna().unique())
                if len(cats) == 0:
                    fit[c] = np.nan
                    for df in transformed:
                        df[c] = np.nan
                else:
                    fit[c] = pd.Categorical(fit[c], categories=cats, ordered=False)
                    for df in transformed:
                        df[c] = pd.Categorical(df[c], categories=cats, ordered=False)
                    active_cat.append(c)

        elif is_datetime_like(fit_col):
            fit[c] = normalize_numeric_series(convert_datetime(fit_col))
            for df in transformed:
                df[c] = normalize_numeric_series(convert_datetime(df[c]))

        else:
            fit[c] = normalize_numeric_series(fit_col)
            for df in transformed:
                df[c] = normalize_numeric_series(df[c])

        if not pd.api.types.is_categorical_dtype(fit[c]):
            fit[c] = fit[c].replace([np.inf, -np.inf], np.nan)
        for df in transformed:
            if not pd.api.types.is_categorical_dtype(df[c]):
                df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    valid_cols = [c for c in fit.columns if fit[c].notna().any()]
    fit = fit[valid_cols]
    transformed = [df[valid_cols] for df in transformed]
    active_cat = [c for c in active_cat if c in valid_cols]

    return fit, transformed, active_cat


params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbose": -1,
}
NUM_ROUNDS = 2000


def train_and_evaluate(fit_df, fit_y, val_df, val_y):
    fit_enc, val_list, active_cats = preprocess_features(fit_df, [val_df])
    val_enc = val_list[0]

    if fit_enc.shape[1] == 0:
        pred = np.full(len(val_y), float(fit_y.mean()))
        return float(np.sqrt(mean_squared_error(val_y, pred))), 1

    train_set = lgb.Dataset(
        fit_enc,
        label=fit_y,
        categorical_feature=active_cats,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        val_enc,
        label=val_y,
        categorical_feature=active_cats,
        reference=train_set,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    best_iter = getattr(model, "best_iteration", 0)
    if best_iter is None or best_iter <= 0:
        try:
            best_iter = model.current_iteration()
        except Exception:
            best_iter = NUM_ROUNDS
    best_iter = int(max(1, best_iter))

    pred = model.predict(val_enc, num_iteration=best_iter)
    rmse = float(np.sqrt(mean_squared_error(val_y, pred)))
    return rmse, best_iter


n = len(y)
if n == 0:
    print("CV RMSE: nan")
    submission = pd.DataFrame({"id": np.arange(len(test)), "Overall": np.nan})
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    sys.exit(0)

if n >= 10:
    n_splits = 5
elif n >= 4:
    n_splits = 2
else:
    n_splits = 1

scores = []
best_iters = []

if n_splits > 1:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(X):
        rmse, best_iter = train_and_evaluate(
            X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx]
        )
        scores.append(rmse)
        best_iters.append(best_iter)
elif n >= 2:
    tr_idx = np.arange(n - 1)
    va_idx = np.array([n - 1])
    rmse, best_iter = train_and_evaluate(
        X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx]
    )
    scores.append(rmse)
    best_iters.append(best_iter)
else:
    pred = np.full(n, float(y.mean()))
    scores.append(float(np.sqrt(mean_squared_error(y, pred))))
    best_iters.append(1)

cv_rmse = float(np.mean(scores)) if scores else float("nan")
print(f"CV RMSE: {cv_rmse:.6f}")

if X.shape[1] == 0:
    test_pred = np.full(len(test), float(y.mean()))
else:
    X_final_enc, test_list, final_active_cats = preprocess_features(X, [test])
    test_enc = test_list[0]

    if X_final_enc.shape[1] == 0:
        test_pred = np.full(len(test), float(y.mean()))
    else:
        final_rounds = int(np.median(best_iters)) if best_iters else 500
        final_rounds = int(max(1, min(final_rounds, NUM_ROUNDS)))

        train_set_final = lgb.Dataset(
            X_final_enc,
            label=y,
            categorical_feature=final_active_cats,
            free_raw_data=False,
        )
        final_model = lgb.train(params, train_set_final, num_boost_round=final_rounds)
        test_pred = final_model.predict(test_enc)

test_pred = np.asarray(test_pred, dtype=float)
test_pred = np.nan_to_num(
    test_pred,
    nan=float(y.mean()),
    posinf=float(y.max()),
    neginf=float(y.min()),
)

submission = pd.DataFrame({"id": np.arange(len(test)), "Overall": test_pred})
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
