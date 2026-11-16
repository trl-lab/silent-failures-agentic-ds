import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/us-school-bus-fleet/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/us-school-bus-fleet/data/test.parquet"
TARGET = "Bus age"
MISSING = "MISSING"

DROP_COLUMNS = [
    "WRI Bus ID",
    "VIN",
    "Name of school district or primary user",
    "LEAID",
    "State District ID",
    "WRI Entity ID",
    "Third party involved",
]


def basic_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_col = "Date updated or last inspected"
    if date_col in df.columns:
        raw = df[date_col]
        text = raw.astype(str)

        year_start = text.str.extract(r"(19|20)\d{2}")[0]
        year_end = text.str.extract(r"(?:19|20)\d{2}\s*-\s*(19|20)\d{2}")[0]

        df["Date_year"] = pd.to_numeric(year_start, errors="coerce")
        df["Date_year_end"] = pd.to_numeric(year_end, errors="coerce")

        parsed = pd.to_datetime(raw, errors="coerce")
        df["Date_month"] = parsed.dt.month
        df["Date_day"] = parsed.dt.day
        df["Date_year_missing"] = df["Date_year"].isna().astype(np.float32)

        df = df.drop(columns=[date_col])

    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def clean_categorical(frame: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in cat_cols:
        if col in frame.columns:
            s = frame[col].astype(str)
        else:
            s = pd.Series(MISSING, index=frame.index)

        s = s.replace(
            {
                "nan": MISSING,
                "None": MISSING,
                "": MISSING,
                "NaN": MISSING,
            }
        )
        out[col] = s
    return out


def get_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
    else:
        values = pd.Series(np.nan, index=df.index, dtype=np.float64)

    values = values.replace([np.inf, -np.inf], np.nan)
    return values.astype(np.float32)


def make_features(df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in reference_df.columns
        if pd.api.types.is_numeric_dtype(reference_df[col])
    ]

    categorical_candidates = [
        col
        for col in reference_df.columns
        if not pd.api.types.is_numeric_dtype(reference_df[col])
    ]

    out = pd.DataFrame(index=df.index)

    for col in numeric_cols:
        values = get_numeric_series(df, col)
        out[col] = values
        out[f"{col}_missing"] = values.isna().astype(np.float32)

    for col in categorical_candidates:
        if col in df.columns:
            out[f"{col}_missing"] = df[col].isna().astype(np.float32)
        else:
            out[f"{col}_missing"] = 0.0

    categorical_cols = []
    for col in categorical_candidates:
        if col in reference_df.columns:
            nunique = reference_df[col].nunique(dropna=True)
        else:
            nunique = 0
        if 0 < nunique <= 100:
            categorical_cols.append(col)

    if categorical_cols:
        ref_cat = clean_categorical(reference_df, categorical_cols)
        test_cat = clean_categorical(df, categorical_cols)

        ref_dummies = pd.get_dummies(ref_cat, prefix=categorical_cols, drop_first=False)
        test_dummies = pd.get_dummies(test_cat, prefix=categorical_cols, drop_first=False)

        for col in categorical_cols:
            missing_col = f"{col}_{MISSING}"
            if missing_col not in ref_dummies.columns:
                ref_dummies[missing_col] = 0

        test_dummies = test_dummies.reindex(columns=ref_dummies.columns, fill_value=0)
        test_dummies = test_dummies.astype(np.float32)

        out = pd.concat([out, test_dummies], axis=1)

    return out


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    def clean(name: str) -> str:
        s = str(name)
        s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        if not s:
            s = "feature"
        if s[0].isdigit():
            s = f"f_{s}"
        return s[:50]

    seen = {}
    new_cols = []
    for col in df.columns:
        base = clean(col)
        if base in seen:
            seen[base] += 1
            new_col = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            new_col = base
        new_cols.append(new_col)

    df = df.copy()
    df.columns = new_cols
    return df


def build_models() -> dict:
    return {
        "lightgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            thread_count=-1,
        ),
    }


def train_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_predict: pd.DataFrame,
) -> np.ndarray:
    if len(X_predict) == 0:
        return np.empty(0, dtype=np.float64)

    if X_train.shape[1] == 0:
        return np.full(len(X_predict), float(np.mean(y_train)), dtype=np.float64)

    models = build_models()
    predictions = []

    for model in models.values():
        model.fit(X_train, y_train)
        pred = model.predict(X_predict)
        pred = np.asarray(pred, dtype=np.float64)
        pred = np.where(np.isfinite(pred), pred, float(np.mean(y_train)))
        predictions.append(pred)

    return np.mean(predictions, axis=0)


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    y = pd.to_numeric(train_df[TARGET], errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan)

    train_df = train_df[y.notna()].copy()
    y = y[y.notna()].astype(np.float64)

    if len(y) == 0:
        pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                "Bus age": np.zeros(len(test_df), dtype=np.float64),
            }
        ).to_csv("predictions.csv", index=False)
        return

    X_raw = train_df.drop(columns=[TARGET])

    if len(X_raw) > 20:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_raw,
            y,
            test_size=0.2,
            random_state=42,
        )
    else:
        X_train_raw = X_raw
        X_val_raw = X_raw
        y_train = y
        y_val = y

    train_ref = basic_transform(X_train_raw)

    X_train = make_features(train_ref, train_ref)
    X_train = sanitize_feature_names(X_train)
    X_train = X_train.dropna(axis=1, how="all")

    X_val = make_features(basic_transform(X_val_raw), train_ref)
    X_val = sanitize_feature_names(X_val)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    val_pred = train_and_predict(X_train, y_train, X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
    print(f"Validation RMSE: {val_rmse:.6f}")

    full_ref = basic_transform(X_raw)

    X_full = make_features(full_ref, full_ref)
    X_full = sanitize_feature_names(X_full)
    X_full = X_full.dropna(axis=1, how="all")

    X_test = make_features(basic_transform(test_df), full_ref)
    X_test = sanitize_feature_names(X_test)
    X_test = X_test.reindex(columns=X_full.columns, fill_value=0)

    final_pred = train_and_predict(X_full, y, X_test)
    final_pred = np.where(np.isfinite(final_pred), final_pred, float(np.mean(y)))

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "Bus age": final_pred.astype(np.float64),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()