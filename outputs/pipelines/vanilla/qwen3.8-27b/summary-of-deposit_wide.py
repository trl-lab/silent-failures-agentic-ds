import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/summary-of-deposit_wide/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/summary-of-deposit_wide/data/test.parquet"
TARGET = "DEPSUM"

CAT_MAX_UNIQUE = 120
DROP_NUMERIC_ID_PATTERNS = (
    "CERT",
    "UNINUMBR",
    "DOCKET",
    "FDICDBS",
    "RSSDID",
    "STCNTY",
    "ZIP",
    "CNTYNUMB",
    "CSABR",
    "DIVISIONB",
    "METROBR",
    "MICROBR",
    "MSABR",
    "STCNTYBR",
    "STNUMBR",
    "ZIPBR",
    "RSSDHCR",
)
KEEP_HIGH_UNIQUE_NUMERIC = {"SIMS_LATITUDE", "SIMS_LONGITUDE"}


def is_categorical_series(s: pd.Series) -> bool:
    return s.dtype == object or s.dtype == bool or str(s.dtype) == "category"


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def select_features(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    if TARGET in train.columns:
        train = train.drop(columns=[TARGET])
    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test = test[common_cols]

    drop_cols = set()

    for col in train.columns:
        if train[col].nunique(dropna=True) <= 1:
            drop_cols.add(col)
        if train[col].isna().mean() > 0.99:
            drop_cols.add(col)

    for col in train.columns:
        if col in drop_cols:
            continue

        if is_categorical_series(train[col]):
            if train[col].nunique(dropna=True) > CAT_MAX_UNIQUE:
                drop_cols.add(col)
        else:
            upper = col.upper()
            if any(pattern in upper for pattern in DROP_NUMERIC_ID_PATTERNS):
                drop_cols.add(col)
                continue

            n_unique = train[col].nunique(dropna=True)
            if (
                n_unique > max(1000, int(0.95 * len(train)))
                and col not in KEEP_HIGH_UNIQUE_NUMERIC
            ):
                drop_cols.add(col)

    keep_cols = [c for c in train.columns if c not in drop_cols]
    return train[keep_cols], test[keep_cols]


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    cat_cols = [c for c in train.columns if is_categorical_series(train[c])]
    num_cols = [c for c in train.columns if c not in cat_cols]

    for col in cat_cols:
        train[col] = train[col].astype("object").fillna("missing")
        test[col] = test[col].astype("object").fillna("missing")
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category")

    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    return train, test, cat_cols


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    train_raw = pd.read_parquet(TRAIN_PATH)
    test_raw = pd.read_parquet(TEST_PATH)

    y = train_raw[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    y = y.loc[valid]
    train_raw = train_raw.loc[valid]

    train_raw = clean_frame(train_raw)
    test_raw = clean_frame(test_raw)

    X_train, X_test = select_features(train_raw, test_raw)
    X_train, X_test, cat_cols = prepare_features(X_train, X_test)

    if X_train.shape[1] == 0:
        preds = np.full(len(X_test), float(y.mean()))
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y, test_size=0.2, random_state=42
        )

        model_specs = {
            "lgbm": LGBMRegressor(
                n_estimators=900,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            "xgb": XGBRegressor(
                n_estimators=900,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            ),
            "cat": CatBoostRegressor(
                iterations=900,
                learning_rate=0.04,
                depth=6,
                random_seed=42,
                verbose=0,
            ),
        }

        val_scores = {}
        fitted_models = {}

        for name, model in model_specs.items():
            try:
                if name == "cat" and cat_cols:
                    model.fit(X_tr, y_tr, cat_features=cat_cols)
                else:
                    model.fit(X_tr, y_tr)

                val_pred = model.predict(X_val)
                val_scores[name] = rmse(y_val, val_pred)
                fitted_models[name] = model
            except Exception:
                continue

        if not fitted_models:
            preds = np.full(len(X_test), float(y.mean()))
        else:
            weights = {name: 1.0 / max(score, 1e-9) for name, score in val_scores.items()}
            total_weight = sum(weights.values())

            test_preds = np.zeros(len(X_test), dtype=float)
            for name, model in fitted_models.items():
                test_preds += (weights[name] / total_weight) * model.predict(X_test)

            preds = test_preds

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_raw), dtype=int),
            "DEPSUM": preds.astype(float),
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()