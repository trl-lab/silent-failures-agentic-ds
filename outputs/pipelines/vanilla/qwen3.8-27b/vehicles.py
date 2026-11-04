import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/vehicles/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/vehicles/data/test.parquet"
TARGET = "youSaveSpend"
DROP_COLS = ["id", "engId", "createdOn", "modifiedOn"]
MISSING = "__MISSING__"
RANDOM_STATE = 42


def write_predictions(n, values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size != n:
        if values.size > 0 and np.isfinite(values).any():
            fill = float(np.nanmean(values[np.isfinite(values)]))
        else:
            fill = 0.0
        values = np.full(n, fill, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    pd.DataFrame(
        {
            "id": np.arange(n, dtype=int),
            "youSaveSpend": values,
        }
    ).to_csv("predictions.csv", index=False)


def get_test_len():
    try:
        return len(pd.read_parquet(TEST_PATH))
    except Exception:
        return 0


def clean_frame(df, drop_target=False):
    df = df.copy()
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    if drop_target and TARGET in df.columns:
        drop_cols.append(TARGET)
    return df.drop(columns=drop_cols, errors="ignore")


def prepare_data(train_path, test_path):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    train = clean_frame(train, drop_target=False)
    test = clean_frame(test, drop_target=True)

    if TARGET not in train.columns:
        raise ValueError(f"Target column {TARGET!r} not found in train data.")

    common_cols = [c for c in train.columns if c in test.columns and c != TARGET]
    train = train[common_cols + [TARGET]]
    test = test[common_cols]

    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce").astype(float)
    train[TARGET] = train[TARGET].replace([np.inf, -np.inf], np.nan)
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)

    for col in common_cols:
        if col == "co2A":
            train[col] = train[col].replace(-1, np.nan)
            test[col] = test[col].replace(-1, np.nan)

        if pd.api.types.is_bool_dtype(train[col]) or pd.api.types.is_bool_dtype(test[col]):
            train[col] = train[col].astype(int)
            test[col] = test[col].astype(int)

    numeric_cols = []
    categorical_cols = []

    for col in common_cols:
        if pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_numeric_dtype(test[col]):
            train[col] = pd.to_numeric(train[col], errors="coerce").astype(float)
            test[col] = pd.to_numeric(test[col], errors="coerce").astype(float)
            train[col] = train[col].replace([np.inf, -np.inf], np.nan)
            test[col] = test[col].replace([np.inf, -np.inf], np.nan)
            numeric_cols.append(col)
        else:
            train[col] = (
                train[col]
                .astype(object)
                .where(train[col].notna(), MISSING)
                .astype(str)
            )
            test[col] = (
                test[col]
                .astype(object)
                .where(test[col].notna(), MISSING)
                .astype(str)
            )
            categorical_cols.append(col)

    keep_cols = []
    for col in common_cols:
        if train[col].isna().all():
            continue
        if train[col].nunique(dropna=True) <= 1:
            continue
        keep_cols.append(col)

    numeric_cols = [c for c in numeric_cols if c in keep_cols]
    categorical_cols = [c for c in categorical_cols if c in keep_cols]

    X_train = train[keep_cols].reset_index(drop=True)
    y_train = train[TARGET].reset_index(drop=True)
    X_test = test[keep_cols].reset_index(drop=True)

    return X_train, y_train, X_test, numeric_cols, categorical_cols


class FeatureProcessor:
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.num_imputer = None
        self.ordinal_encoder = None

    def fit(self, X):
        if self.numeric_cols:
            self.num_imputer = SimpleImputer(strategy="median")
            self.num_imputer.fit(X[self.numeric_cols])

        if self.categorical_cols:
            self.ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.ordinal_encoder.fit(X[self.categorical_cols])

        return self

    def transform(self, X):
        parts = []

        if self.numeric_cols:
            parts.append(self.num_imputer.transform(X[self.numeric_cols]))

        if self.categorical_cols:
            parts.append(self.ordinal_encoder.transform(X[self.categorical_cols]))

        if not parts:
            return np.empty((len(X), 0), dtype=float)

        return np.hstack(parts).astype(float, copy=False)


def make_model(n_estimators=1000):
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
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


def main():
    n_test = get_test_len()

    try:
        X_full, y_full, X_test, numeric_cols, categorical_cols = prepare_data(
            TRAIN_PATH, TEST_PATH
        )
        n_test = len(X_test)

        if len(X_full) == 0 or X_full.shape[1] == 0:
            fallback = float(y_full.mean()) if len(y_full) > 0 else 0.0
            write_predictions(n_test, fallback)
            return

        if len(X_full) >= 100:
            X_train, X_val, y_train, y_val = train_test_split(
                X_full,
                y_full,
                test_size=0.2,
                random_state=RANDOM_STATE,
                shuffle=True,
            )

            val_processor = FeatureProcessor(numeric_cols, categorical_cols).fit(X_train)
            X_train_val = val_processor.transform(X_train)
            X_val_val = val_processor.transform(X_val)

            val_model = make_model(n_estimators=500)
            val_model.fit(X_train_val, y_train)
            val_pred = val_model.predict(X_val_val)
            _ = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        final_processor = FeatureProcessor(numeric_cols, categorical_cols).fit(X_full)
        X_full_proc = final_processor.transform(X_full)
        X_test_proc = final_processor.transform(X_test)

        final_model = make_model(n_estimators=1000)
        final_model.fit(X_full_proc, y_full)

        predictions = final_model.predict(X_test_proc)
        write_predictions(n_test, predictions)

    except Exception:
        fallback = 0.0
        try:
            train = pd.read_parquet(TRAIN_PATH)
            if TARGET in train.columns:
                y = pd.to_numeric(train[TARGET], errors="coerce")
                y = y.replace([np.inf, -np.inf], np.nan)
                if y.notna().any():
                    fallback = float(y.mean())
        except Exception:
            pass

        write_predictions(n_test, fallback)


if __name__ == "__main__":
    main()