import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")
np.random.seed(42)

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)
TARGET = "entityPayment"

train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

id_cols = set()
for col in train.columns:
    lc = str(col).lower().strip()
    if lc in {
        "id",
        "index",
        "idx",
        "row_id",
        "rowid",
        "identifier",
        "entityid",
        "customerid",
        "accountid",
        "payerid",
        "providerid",
    }:
        id_cols.add(col)
    elif lc.startswith("id_") or lc.endswith("_id"):
        id_cols.add(col)

drop_cols = set(id_cols)
for col in train.columns:
    lc = str(col).lower().strip()
    if lc in {"target", "label", "y", "target_value", "targetvalue", "entitypayment"}:
        drop_cols.add(col)

group_col = None
for cand in [
    "entity_id",
    "entityid",
    "customer_id",
    "customerid",
    "account_id",
    "accountid",
    "payer_id",
    "payerid",
    "provider_id",
    "providerid",
    "id",
]:
    if cand in train.columns and cand != TARGET:
        group_col = cand
        break

if group_col is None:
    for col in train.columns:
        if col == TARGET:
            continue
        lc = str(col).lower().strip()
        if (
            lc
            in {
                "id",
                "identifier",
                "entityid",
                "customerid",
                "accountid",
                "payerid",
                "providerid",
            }
            or lc.startswith("id_")
            or lc.endswith("_id")
        ):
            if train[col].nunique(dropna=True) < len(train):
                group_col = col
                break

common_cols = [
    c for c in train.columns if c in test.columns and c != TARGET and c not in drop_cols
]
if not common_cols:
    common_cols = [c for c in train.columns if c != TARGET and c not in drop_cols]

X = train[common_cols].copy()
X_test = test[[c for c in common_cols if c in test.columns]].copy()
for c in common_cols:
    if c not in X_test.columns:
        X_test[c] = np.nan


def robust_to_numeric(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str)
    s = (
        s.str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")


def datetime_to_numeric(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = pd.to_datetime(s, errors="coerce")
    vals = pd.Series(np.nan, index=s.index, dtype=float)
    mask = s.notna()
    if mask.any():
        vals[mask] = s[mask].astype("int64") / 1e9
    return vals


def timedelta_to_numeric(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_timedelta(s, errors="coerce").dt.total_seconds()


y = robust_to_numeric(train[TARGET])
y = y.replace([np.inf, -np.inf], np.nan)
valid = y.notna()
X = X.loc[valid].copy()
y = y.loc[valid].copy()
groups = (
    train[group_col].loc[valid].copy()
    if group_col is not None and group_col in train.columns
    else None
)


def clean_features(X, X_test):
    for df in (X, X_test):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    cat_cols = []
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = datetime_to_numeric(X[col])
            X_test[col] = datetime_to_numeric(X_test[col])
        elif pd.api.types.is_timedelta64_dtype(X[col]):
            X[col] = timedelta_to_numeric(X[col])
            X_test[col] = timedelta_to_numeric(X_test[col])
        elif pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        elif (
            pd.api.types.is_string_dtype(X[col])
            or X[col].dtype == object
            or X[col].dtype.name == "string"
        ):
            sample = X[col].dropna().head(2000)
            if len(sample) > 0:
                num_sample = robust_to_numeric(sample)
                if num_sample.notna().mean() > 0.8:
                    X[col] = robust_to_numeric(X[col])
                    X_test[col] = robust_to_numeric(X_test[col])
                else:
                    dt_sample = pd.to_datetime(sample, errors="coerce")
                    if dt_sample.notna().mean() > 0.8:
                        X[col] = datetime_to_numeric(X[col])
                        X_test[col] = datetime_to_numeric(X_test[col])
                    else:
                        cats = X[col].dropna().unique()
                        dtype = pd.CategoricalDtype(categories=cats)
                        X[col] = X[col].astype(dtype)
                        X_test[col] = X_test[col].astype(dtype)
                        cat_cols.append(col)
            else:
                X[col] = X[col].astype("category")
                X_test[col] = X_test[col].astype("category")
                cat_cols.append(col)
        elif X[col].dtype.name == "category":
            cats = X[col].cat.categories
            dtype = pd.CategoricalDtype(categories=cats)
            X[col] = X[col].astype(dtype)
            X_test[col] = X_test[col].astype(dtype)
            cat_cols.append(col)

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_numeric_dtype(
            X_test[col]
        ):
            try:
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            except Exception:
                X_test[col] = np.nan

    X = X.loc[:, X.notna().any()]
    X_test = X_test.loc[:, X.columns]

    keep = []
    for col in X.columns:
        try:
            if X[col].nunique(dropna=True) > 1:
                keep.append(col)
        except Exception:
            keep.append(col)

    if keep:
        X = X[keep]
        X_test = X_test[keep]
    else:
        X = X.iloc[:, []]
        X_test = X_test.iloc[:, :]

    cat_cols = [c for c in cat_cols if c in X.columns]
    return X, X_test, cat_cols


X, X_test, cat_cols = clean_features(X, X_test)

if X.empty or X.shape[1] == 0:
    fallback = float(y.mean()) if len(y) > 0 else 0.0
    cv_rmse = float(np.sqrt(np.mean((y - fallback) ** 2))) if len(y) > 0 else 0.0
    print(f"5-fold CV RMSE: {cv_rmse:.6f}")
    submission = pd.DataFrame({"id": np.arange(len(test)), "entityPayment": fallback})
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
else:
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "seed": 42,
        "num_threads": -1,
    }
    num_boost_round = 5000
    callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    cat_feature = cat_cols if cat_cols else "auto"

    if len(X) < 5:
        dtrain_final = lgb.Dataset(X, label=y, categorical_feature=cat_feature)
        final_model = lgb.train(params, dtrain_final, num_boost_round=num_boost_round)
        train_pred = np.asarray(
            final_model.predict(X, num_iteration=num_boost_round), dtype=float
        )
        cv_rmse = float(root_mean_squared_error(y, train_pred))
        print(f"In-sample RMSE (insufficient data for 5-fold CV): {cv_rmse:.6f}")

        if X_test.empty:
            test_pred = np.array([], dtype=float)
        else:
            test_pred = np.asarray(
                final_model.predict(X_test, num_iteration=num_boost_round), dtype=float
            )
    else:
        use_group = False
        if groups is not None:
            groups_clean = groups.astype(str).fillna("missing")
            if groups_clean.nunique() >= 5 and groups_clean.nunique() < len(
                groups_clean
            ):
                use_group = True

        scores = []
        best_iters = []

        if use_group:
            splitter = GroupKFold(n_splits=5)
            split_iter = splitter.split(X, y, groups=groups_clean)
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=42)
            split_iter = splitter.split(X, y)

        for tr_idx, va_idx in split_iter:
            dtrain = lgb.Dataset(
                X.iloc[tr_idx], label=y.iloc[tr_idx], categorical_feature=cat_feature
            )
            dvalid = lgb.Dataset(
                X.iloc[va_idx],
                label=y.iloc[va_idx],
                reference=dtrain,
                categorical_feature=cat_feature,
            )
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dvalid],
                callbacks=callbacks,
            )
            bi = getattr(model, "best_iteration", 0)
            if bi is None or bi <= 0:
                bi = num_boost_round
            best_iters.append(bi)
            pred = model.predict(X.iloc[va_idx], num_iteration=bi)
            scores.append(root_mean_squared_error(y.iloc[va_idx], pred))

        cv_rmse = float(np.mean(scores))
        cv_std = float(np.std(scores))
        print(f"5-fold CV RMSE: {cv_rmse:.6f} (+/- {cv_std:.6f})")

        valid_best = [b for b in best_iters if b and b > 0]
        final_rounds = int(np.median(valid_best)) if valid_best else num_boost_round
        final_rounds = max(1, min(final_rounds, num_boost_round))

        dtrain_final = lgb.Dataset(X, label=y, categorical_feature=cat_feature)
        final_model = lgb.train(params, dtrain_final, num_boost_round=final_rounds)

        if X_test.empty:
            test_pred = np.array([], dtype=float)
        else:
            test_pred = np.asarray(
                final_model.predict(X_test, num_iteration=final_rounds), dtype=float
            )

    if len(y) > 0:
        test_pred = np.nan_to_num(
            test_pred,
            nan=float(y.mean()),
            posinf=float(y.max()),
            neginf=float(y.min()),
        )
        if y.min() >= 0:
            test_pred = np.maximum(test_pred, 0.0)

    if len(test_pred) != len(test):
        test_pred = np.full(len(test), float(y.mean()) if len(y) > 0 else 0.0)

    submission = pd.DataFrame({"id": np.arange(len(test)), "entityPayment": test_pred})
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    print(f"Saved submission with {len(submission)} rows.")
