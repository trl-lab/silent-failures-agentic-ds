import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/beijing-pm25/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/beijing-pm25/data/test.parquet"
TARGET = "pm2.5"
ID_COLUMNS = {"no", "id", "index", "passengerid", "row_id"}


def clean_dataframe(df, drop_target=False):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    drop_cols = [c for c in df.columns if c.lower() in ID_COLUMNS]
    if drop_target and TARGET in df.columns:
        drop_cols.append(TARGET)
    df = df.drop(columns=drop_cols, errors="ignore")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-1, np.nan)
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    if all(c in df.columns for c in ("year", "month", "day", "hour")):
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        try:
            dt = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
            df["dayofweek"] = dt.dt.dayofweek
        except Exception:
            df["dayofweek"] = np.nan

    return df


class TabularPreprocessor:
    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_fill = {}
        self.categorical_categories = {}
        self.feature_names = []

    def fit(self, X):
        self.numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

        for col in self.numeric_cols:
            fill = X[col].median()
            self.numeric_fill[col] = float(fill) if pd.notna(fill) else 0.0

        for col in self.categorical_cols:
            vals = X[col].dropna().astype(str)
            cats = sorted(set(vals.unique()) | {"missing"})
            self.categorical_categories[col] = cats

        self.feature_names = list(self.numeric_cols)
        for col in self.categorical_cols:
            self.feature_names.extend([f"{col}={cat}" for cat in self.categorical_categories[col]])

        return self

    def transform(self, X):
        X = X.reindex(columns=self.numeric_cols + self.categorical_cols, fill_value=np.nan)
        out = {}

        for col in self.numeric_cols:
            if col in X.columns:
                vals = pd.to_numeric(X[col], errors="coerce")
            else:
                vals = pd.Series(np.nan, index=X.index)
            out[col] = vals.fillna(self.numeric_fill.get(col, 0.0))

        for col in self.categorical_cols:
            if col in X.columns:
                vals = X[col].astype(str)
                vals = vals.replace({"nan": "missing", "None": "missing", "": "missing"})
            else:
                vals = pd.Series("missing", index=X.index)

            for cat in self.categorical_categories.get(col, []):
                out[f"{col}={cat}"] = (vals == cat).astype(int)

        if not self.feature_names:
            return pd.DataFrame(index=X.index)

        return pd.DataFrame(out, index=X.index)[self.feature_names]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def make_validation_split(X, y):
    if len(X) < 2:
        return X, X.iloc[0:0], y, y.iloc[0:0]

    if len(X) < 20:
        n_val = max(1, int(len(X) * 0.2))
        n_val = min(n_val, len(X) // 2)
        if n_val == 0:
            n_val = 1
        return train_test_split(X, y, test_size=n_val, random_state=42)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="rmse",
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
    }


def sanitize_predictions(pred, y_ref):
    pred = np.asarray(pred, dtype=float)
    finite = y_ref[np.isfinite(y_ref)]
    fallback = float(np.mean(finite)) if len(finite) else 0.0
    pred = np.nan_to_num(pred, nan=fallback, posinf=fallback, neginf=fallback)
    return np.maximum(pred, 0.0)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = clean_dataframe(train_df, drop_target=False)
    test_df = clean_dataframe(test_df, drop_target=True)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    y_train = pd.to_numeric(train_df[TARGET], errors="coerce")
    y_train = y_train.replace([np.inf, -np.inf], np.nan)
    X_train = train_df.drop(columns=[TARGET])

    valid = y_train.notna()
    X_train = X_train.loc[valid]
    y_train = y_train.loc[valid]

    X_test = test_df.reindex(columns=X_train.columns, fill_value=np.nan)

    if len(X_train) == 0:
        fallback = 0.0
        pd.DataFrame({"id": np.arange(len(X_test), dtype=int), "pm2.5": fallback}).to_csv(
            "predictions.csv", index=False
        )
        return

    X_tr, X_val, y_tr, y_val = make_validation_split(X_train, y_train)

    preprocessor = TabularPreprocessor().fit(X_tr)
    X_tr_p = preprocessor.transform(X_tr)
    X_val_p = preprocessor.transform(X_val)

    scores = {}
    if len(X_val) > 0 and X_tr_p.shape[1] > 0:
        for name, model in make_models().items():
            try:
                model.fit(X_tr_p, y_tr)
                pred = sanitize_predictions(model.predict(X_val_p), y_tr)
                scores[name] = rmse(y_val, pred)
            except Exception:
                pass

    best_name = min(scores, key=scores.get) if scores else None

    final_preprocessor = TabularPreprocessor().fit(X_train)
    X_train_p = final_preprocessor.transform(X_train)
    X_test_p = final_preprocessor.transform(X_test)

    if best_name is not None and X_train_p.shape[1] > 0:
        final_model = make_models()[best_name]
        final_model.fit(X_train_p, y_train)
        test_pred = sanitize_predictions(final_model.predict(X_test_p), y_train)
    else:
        test_pred = np.full(len(X_test), float(np.nanmean(y_train)))

    pd.DataFrame({"id": np.arange(len(test_pred), dtype=int), "pm2.5": test_pred}).to_csv(
        "predictions.csv", index=False
    )


if __name__ == "__main__":
    main()