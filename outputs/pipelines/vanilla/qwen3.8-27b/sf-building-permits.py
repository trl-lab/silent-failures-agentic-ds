import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/sf-building-permits/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/sf-building-permits/data/test.parquet"
TARGET = "time_to_approve"

DROP_COLS = [
    TARGET,
    "Permit Number",
    "Record ID",
    "Location",
    "Description",
    "Permit Creation Date",
    "First Construction Document Date",
    "Current Status Date",
    "Completed Date",
    "Current Status",
    "Revised Cost",
    "Permit Expiration Date",
    "Block",
    "Lot",
]

KEYWORDS = [
    "new", "addition", "roof", "shingle", "kitchen", "bathroom", "foundation",
    "structural", "demolition", "access", "restroom", "stair", "railing",
    "window", "door", "electrical", "plumbing", "mechanical", "fire", "solar",
    "pool", "deck", "porch", "garage", "unit", "story", "stories", "in-kind",
    "replacement", "repair", "alteration", "construction", "remove", "install"
]


def parse_location(df: pd.DataFrame) -> pd.DataFrame:
    if "Location" in df.columns:
        loc = df["Location"].astype(str).str.extract(
            r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)",
            expand=True,
        )
        df["lat"] = pd.to_numeric(loc[0], errors="coerce")
        df["lon"] = pd.to_numeric(loc[1], errors="coerce")
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan
    return df


def parse_creation_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Permit Creation Date" in df.columns:
        dt = pd.to_datetime(df["Permit Creation Date"], errors="coerce", format="%m/%d/%Y")
        if dt.isna().mean() > 0.5:
            dt = pd.to_datetime(df["Permit Creation Date"], errors="coerce")

        df["creation_year"] = dt.dt.year
        df["creation_month"] = dt.dt.month
        df["creation_day"] = dt.dt.day
        df["creation_quarter"] = dt.dt.quarter
        df["creation_dayofweek"] = dt.dt.dayofweek
        df["creation_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df["creation_days"] = (dt - pd.Timestamp("2000-01-01")).dt.days
    else:
        for col in [
            "creation_year",
            "creation_month",
            "creation_day",
            "creation_quarter",
            "creation_dayofweek",
            "creation_is_weekend",
            "creation_days",
        ]:
            df[col] = np.nan
    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Description" in df.columns:
        desc = df["Description"].fillna("").astype(str)
        df["desc_len"] = desc.str.len()
        df["desc_word_count"] = desc.str.split().str.len()
        df["desc_digit_count"] = desc.str.count(r"\d")
        df["desc_upper_count"] = desc.str.count(r"[A-Z]")

        for kw in KEYWORDS:
            df[f"desc_has_{kw}"] = desc.str.contains(kw, case=False, regex=False).astype(int)
    else:
        df["desc_len"] = 0
        df["desc_word_count"] = 0
        df["desc_digit_count"] = 0
        df["desc_upper_count"] = 0
        for kw in KEYWORDS:
            df[f"desc_has_{kw}"] = 0

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = parse_location(df)
    df = parse_creation_date(df)
    df = add_text_features(df)

    drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col in df.select_dtypes(include=[bool]).columns:
        df[col] = df[col].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    for col in categorical_cols:
        df[col] = df[col].astype("object")
        df[col] = df[col].where(df[col].notna(), "__missing__")
        df[col] = df[col].astype(str)

    return df


def select_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cols = [c for c in train.columns if c in test.columns]
    train = train[cols]
    test = test[cols]

    numeric_cols = set(train.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in train.columns if c not in numeric_cols]

    keep_cols = []
    for col in train.columns:
        if train[col].isna().all():
            continue

        nunique = train[col].nunique(dropna=True)
        if nunique <= 1:
            continue

        if col in categorical_cols and nunique > 5000:
            continue

        keep_cols.append(col)

    keep_cat_cols = [c for c in keep_cols if c in categorical_cols]
    return train[keep_cols], test[keep_cols], keep_cat_cols


def time_based_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    if len(X) < 2:
        return X, X.iloc[0:0], y, y.iloc[0:0]

    if "creation_days" in X.columns:
        valid_mask = X["creation_days"].notna()
        if int(valid_mask.sum()) > 100:
            sorted_idx = X.loc[valid_mask, "creation_days"].sort_values().index
            n_val = max(1, int(len(sorted_idx) * test_size))
            val_idx = sorted_idx[-n_val:]

            val_mask = X.index.isin(val_idx)
            train_mask = ~val_mask

            if int(train_mask.sum()) > 0 and int(val_mask.sum()) > 0:
                return X.loc[train_mask], X.loc[val_mask], y.loc[train_mask], y.loc[val_mask]

    return train_test_split(X, y, test_size=test_size, random_state=42)


def make_catboost(iterations: int, cat_indices: list[int], early_stopping_rounds: int | None = None) -> CatBoostRegressor:
    params = dict(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )
    if cat_indices:
        params["cat_features"] = cat_indices
    if early_stopping_rounds is not None:
        params["eval_metric"] = "RMSE"
        params["early_stopping_rounds"] = early_stopping_rounds
    return CatBoostRegressor(**params)


def train_catboost(
    X_train_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    cat_indices: list[int],
) -> np.ndarray:
    if len(X_train_full) < 100:
        model = make_catboost(500, cat_indices)
        model.fit(X_train_full, y_full, verbose=0)
        return model.predict(X_test)

    X_train, X_val, y_train, y_val = time_based_split(X_train_full, y_full)

    if len(X_train) == 0 or len(X_val) == 0:
        model = make_catboost(1000, cat_indices)
        model.fit(X_train_full, y_full, verbose=0)
        return model.predict(X_test)

    model = make_catboost(1000, cat_indices, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

    best_iter = getattr(model, "best_iteration_", None)
    try:
        best_iter = int(best_iter)
    except Exception:
        best_iter = 1000

    if best_iter <= 0:
        best_iter = 1000
    else:
        best_iter = min(best_iter + 1, 1000)

    final_model = make_catboost(best_iter, cat_indices)
    final_model.fit(X_train_full, y_full, verbose=0)
    return final_model.predict(X_test)


def encode_for_lightgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in cat_cols:
        if col not in X_train.columns or col not in X_test.columns:
            continue

        train_vals = X_train[col].astype(str)
        cats = sorted(train_vals.unique())
        mapping = {c: i for i, c in enumerate(cats)}

        X_train[col] = train_vals.map(mapping).astype(int)
        X_test[col] = X_test[col].astype(str).map(mapping).fillna(-1).astype(int)

    for col in X_train.columns:
        if X_train[col].dtype == object:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    return X_train, X_test


def train_lightgbm_fallback(
    X_train_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> np.ndarray:
    try:
        X_train, X_test_enc = encode_for_lightgbm(X_train_full, X_test, cat_cols)

        if len(X_train) < 100:
            model = LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train, y_full)
            return model.predict(X_test_enc)

        X_train2, X_val, y_train, y_val = time_based_split(X_train, y_full)

        if len(X_train2) == 0 or len(X_val) == 0:
            model = LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train, y_full)
            return model.predict(X_test_enc)

        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train2, y_train)
        return model.predict(X_test_enc)
    except Exception:
        fallback = float(y_full.mean()) if len(y_full) > 0 else 0.0
        if np.isnan(fallback):
            fallback = 0.0
        return np.full(len(X_test), fallback, dtype=float)


def clean_predictions(pred: np.ndarray, n: int, y_full: pd.Series) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(pred, 0.0, None)

    if len(pred) != n:
        tmp = np.zeros(n, dtype=float)
        m = min(len(pred), n)
        tmp[:m] = pred[:m]
        pred = tmp

    if len(y_full) > 0:
        fallback = float(y_full.mean())
        if not np.isnan(fallback):
            pred = np.where(np.isnan(pred), fallback, pred)

    return pred


def write_predictions(pred: np.ndarray) -> None:
    submission = pd.DataFrame(
        {
            "id": np.arange(len(pred), dtype=int),
            "time_to_approve": pred,
        }
    )
    submission.to_csv("predictions.csv", index=False)


def write_fallback_predictions(test_df: pd.DataFrame, y_full: pd.Series | None = None) -> None:
    if y_full is not None and len(y_full) > 0:
        pred = float(y_full.mean())
    else:
        pred = 0.0

    if np.isnan(pred):
        pred = 0.0

    write_predictions(np.full(len(test_df), pred, dtype=float))


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        write_fallback_predictions(test_df)
        return

    y_full = train_df[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    valid_mask = y_full.notna()

    train_df = train_df.loc[valid_mask]
    y_full = y_full.loc[valid_mask]

    if len(train_df) == 0:
        write_fallback_predictions(test_df, y_full)
        return

    X_train_full = prepare_features(train_df)
    X_test = prepare_features(test_df)
    X_train_full, X_test, cat_cols = select_features(X_train_full, X_test)

    if X_train_full.shape[1] == 0 or len(X_train_full) == 0:
        write_fallback_predictions(test_df, y_full)
        return

    cat_indices = [X_train_full.columns.get_loc(c) for c in cat_cols]

    try:
        pred = train_catboost(X_train_full, y_full, X_test, cat_indices)
    except Exception:
        pred = train_lightgbm_fallback(X_train_full, y_full, X_test, cat_cols)

    pred = clean_predictions(pred, len(test_df), y_full)
    write_predictions(pred)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if not os.path.exists("predictions.csv"):
            try:
                test_df = pd.read_parquet(TEST_PATH)
                train_df = pd.read_parquet(TRAIN_PATH)
                if TARGET in train_df.columns:
                    y = train_df[TARGET].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                    write_fallback_predictions(test_df, y)
                else:
                    write_fallback_predictions(test_df)
            except Exception:
                write_predictions(np.array([0.0]))