import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/mobile-home-parks/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/mobile-home-parks/data/test.parquet"
TARGET = "SIZE"

DROP_COLS = [
    "OBJECTID",
    "MHPID",
    "NAME",
    "ADDRESS",
    "TELEPHONE",
    "WEBSITE",
    "SOURCEDATE",
    "VAL_DATE",
    "NAICS_DESC",
]

MISSING_VALUES = [
    "",
    "not available",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "<na>",
]


def clean_categorical(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip().str.lower()
    return series.replace(MISSING_VALUES, "MISSING")


def is_missing_series(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip().str.lower()
    return series.isin(MISSING_VALUES) | series.isna()


def save_predictions(predictions: np.ndarray) -> None:
    predictions = np.asarray(predictions).ravel()
    out = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "SIZE": predictions,
        }
    )
    out.to_csv("predictions.csv", index=False)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        save_predictions(np.full(len(test_df), "MISSING", dtype=object))
        return

    train_df = train_df.dropna(subset=[TARGET])
    y_raw = train_df[TARGET].astype(str).str.strip()
    y_missing = y_raw.str.lower().isin(MISSING_VALUES)
    train_df = train_df[~y_missing].copy()
    y = y_raw[~y_missing]

    train_df = train_df.drop(columns=[TARGET], errors="ignore")
    test_df = test_df.drop(columns=[TARGET], errors="ignore")

    for col in DROP_COLS:
        if col in train_df.columns:
            train_df[f"{col}_MISSING"] = is_missing_series(train_df[col]).astype(int)
            if col in test_df.columns:
                test_df[f"{col}_MISSING"] = is_missing_series(test_df[col]).astype(int)
            else:
                test_df[f"{col}_MISSING"] = 0

    drop_cols = [c for c in DROP_COLS if c in train_df.columns or c in test_df.columns]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    common_cols = [c for c in train_df.columns if c in test_df.columns]
    train_df = train_df[common_cols].copy()
    test_df = test_df[common_cols].copy()

    if train_df.empty or y.empty or y.nunique() <= 1:
        majority = y.mode().iloc[0] if not y.empty else "MISSING"
        save_predictions(np.full(len(test_df), majority, dtype=object))
        return

    if test_df.empty:
        save_predictions(np.array([], dtype=object))
        return

    cat_cols = []
    for col in train_df.columns:
        if pd.api.types.is_numeric_dtype(train_df[col]) or pd.api.types.is_numeric_dtype(test_df[col]):
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
            train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
            test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)
        else:
            train_df[col] = clean_categorical(train_df[col])
            test_df[col] = clean_categorical(test_df[col])
            cat_cols.append(col)

    if "UNITS" in train_df.columns:
        train_df["UNITS_MISSING"] = train_df["UNITS"].isna().astype(int)
        test_df["UNITS_MISSING"] = test_df["UNITS"].isna().astype(int)

    constant_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        train_df = train_df.drop(columns=constant_cols, errors="ignore")
        test_df = test_df.drop(columns=constant_cols, errors="ignore")
        cat_cols = [c for c in cat_cols if c not in constant_cols]

    if train_df.shape[1] == 0:
        majority = y.mode().iloc[0]
        save_predictions(np.full(len(test_df), majority, dtype=object))
        return

    use_validation = len(train_df) >= 200
    if use_validation:
        class_counts = y.value_counts()
        stratify = y if (class_counts >= 2).all() else None

        X_fit, X_val, y_fit, y_val = train_test_split(
            train_df,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            cat_features=cat_cols,
            one_hot_max_size=4,
        )
        model.fit(
            X_fit,
            y_fit,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=0,
        )

        best_iter = model.get_best_iteration()
        if best_iter is None or best_iter <= 0:
            best_iter = 2000

        final_model = CatBoostClassifier(
            iterations=int(best_iter),
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            cat_features=cat_cols,
            one_hot_max_size=4,
        )
        final_model.fit(train_df, y, verbose=0)
    else:
        final_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            cat_features=cat_cols,
            one_hot_max_size=4,
        )
        final_model.fit(train_df, y, verbose=0)

    predictions = final_model.predict(test_df)
    save_predictions(predictions)


if __name__ == "__main__":
    main()