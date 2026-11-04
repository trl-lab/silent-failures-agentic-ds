import re
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/local-government-renewable-action/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/local-government-renewable-action/data/test.parquet"
TARGET = "Size of Purchase (MW)"

DROP_COLS = [
    "Ref. No.",
    "Buyer Name",
    "Utility Name",
    "Developer Name",
    "Project Name",
    "Project Location",
    "Source 1",
    "Source 2",
    "Source 3",
    "Source 4",
    "Source 5",
    "Notes",
]

NUMERIC_LIKE_COLS = [
    "City Latitude",
    "City Longitude",
    "Announcement Year",
    "Announcement Year (or Operations Year if Announcement Year unavailable)",
    "Contract Length (Year)",
    "Operation Start Year",
    "Estimated Annual Generation (MWh)",
    "Energy Price in Year 1 ($/MWh)",
]

CAT_COLS = [
    "State",
    "ISO or FERC Region",
    "Technology",
    "Transaction Type",
    "DC/AC",
    "Utility Type",
    "Project State",
]

MISSING_STRINGS = {"", "nan", "none", "null", "nat", "na", "n/a", "unknown"}


def _is_missing(x):
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _clean_cat_value(x):
    if _is_missing(x):
        return "missing"
    s = str(x).strip()
    if s.lower() in MISSING_STRINGS:
        return "missing"
    return s


def _clean_numeric_value(x):
    if _is_missing(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()
    if s.lower() in MISSING_STRINGS:
        return np.nan

    s = s.replace(",", "").replace("$", "").replace(" ", "")
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if match:
        try:
            return float(match.group(0))
        except Exception:
            return np.nan

    return np.nan


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return s.map(_clean_numeric_value).astype(float)


def _replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def _parse_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Announcement Date" not in df.columns:
        return df

    date_series = df["Announcement Date"].astype(object, copy=False)
    try:
        mask = pd.isna(date_series)
        date_series = date_series.copy()
        date_series[mask] = np.nan
        dt = pd.to_datetime(date_series, errors="coerce")
    except Exception:
        dt = pd.Series(pd.NaT, index=df.index)

    df["Announcement Date Year"] = dt.dt.year.astype(float)
    df["Announcement Date Month"] = dt.dt.month.astype(float)
    df["Announcement Date Day"] = dt.dt.day.astype(float)
    df = df.drop(columns=["Announcement Date"], errors="ignore")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET in df.columns:
        df[TARGET] = _to_numeric_series(df[TARGET])

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    for col in NUMERIC_LIKE_COLS:
        if col in df.columns:
            df[col] = _to_numeric_series(df[col])

    df = _parse_date_features(df)

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].map(_clean_cat_value).astype(object)

    df = _replace_inf_with_nan(df)

    non_numeric_cols = [
        c for c in df.columns
        if c != TARGET and not pd.api.types.is_numeric_dtype(df[c])
    ]
    drop_cols = [c for c in non_numeric_cols if c not in CAT_COLS]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_cols,
            )
        )

    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", make_onehot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_cols))

    if not transformers:
        return ColumnTransformer(transformers=[("passthrough", "passthrough", [])], remainder="drop")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(numeric_cols: list, categorical_cols: list, model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", model),
        ]
    )


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def sanitize_predictions(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)


def write_predictions(test_len: int, values: np.ndarray) -> None:
    values = sanitize_predictions(values)
    pd.DataFrame(
        {
            "id": np.arange(test_len, dtype=int),
            TARGET: values.astype(float),
        }
    ).to_csv("predictions.csv", index=False)


def main() -> None:
    train_raw = pd.read_parquet(TRAIN_PATH)
    test_raw = pd.read_parquet(TEST_PATH)

    train = clean_dataframe(train_raw)
    test = clean_dataframe(test_raw)

    if TARGET in test.columns:
        test = test.drop(columns=[TARGET], errors="ignore")

    if TARGET not in train.columns:
        write_predictions(len(test), np.zeros(len(test), dtype=float))
        return

    y = train[TARGET]
    X = train.drop(columns=[TARGET], errors="ignore")

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in CAT_COLS if c in X.columns]
    feature_cols = numeric_cols + categorical_cols
    feature_cols = [c for c in feature_cols if X[c].notna().any()]

    if not feature_cols:
        fallback = float(np.nanmean(y)) if y.notna().any() else 0.0
        write_predictions(len(test), np.full(len(test), fallback, dtype=float))
        return

    X = X[feature_cols]
    test = test.reindex(columns=feature_cols)

    y = y.replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    X_all = X[valid]
    y_all = y[valid].astype(float)

    fallback = float(np.nanmean(y_all)) if len(y_all) > 0 else 0.0

    if len(X_all) < 50:
        write_predictions(len(test), np.full(len(test), fallback, dtype=float))
        return

    if len(X_all) >= 100:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=42,
        )
    else:
        X_train, X_val, y_train, y_val = X_all, X_all, y_all, y_all

    models = {
        "lgbm": LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
        "cat": CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }

    scores = {}
    for name, model in models.items():
        pipe = build_pipeline(numeric_cols, categorical_cols, model)
        try:
            pipe.fit(X_train, y_train)
            pred_val = pipe.predict(X_val)
            scores[name] = safe_rmse(y_val, pred_val)
        except Exception:
            scores[name] = np.inf

    weights = {}
    for name, score in scores.items():
        if np.isfinite(score) and score > 0:
            weights[name] = 1.0 / score
        else:
            weights[name] = 0.0

    total_weight = sum(weights.values())
    if total_weight <= 0:
        finite_models = [name for name, score in scores.items() if np.isfinite(score)]
        if finite_models:
            weights = {name: 1.0 for name in finite_models}
            total_weight = float(len(finite_models))
        else:
            weights = {"lgbm": 1.0}
            total_weight = 1.0

    for name in weights:
        weights[name] /= total_weight

    final_predictions = np.zeros(len(test), dtype=float)
    used_weight = 0.0

    for name, model in models.items():
        if weights.get(name, 0.0) <= 0:
            continue

        pipe = build_pipeline(numeric_cols, categorical_cols, model)
        try:
            pipe.fit(X_all, y_all)
            pred_test = pipe.predict(test)
            pred_test = sanitize_predictions(pred_test)
            final_predictions += weights[name] * pred_test
            used_weight += weights[name]
        except Exception:
            continue

    if used_weight > 0:
        final_predictions /= used_weight
    else:
        final_predictions = np.full(len(test), fallback, dtype=float)

    write_predictions(len(test), final_predictions)


if __name__ == "__main__":
    main()