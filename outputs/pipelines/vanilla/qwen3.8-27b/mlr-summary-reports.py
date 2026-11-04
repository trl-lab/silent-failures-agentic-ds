import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/mlr-summary-reports/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/mlr-summary-reports/data/test.parquet"
TARGET = "Adjusted MLR"
RANDOM_STATE = 42


def _clean_numeric_value(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if not pd.isna(value) else np.nan

    s = str(value).strip()
    if s == "" or s.lower() in {"none", "nan", "null", "na", "n/a", "--", "-"}:
        return np.nan

    s = s.replace("$", "").replace(",", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    if s.endswith("%"):
        s = s[:-1]
    s = s.replace("USD", "").replace("usd", "")

    try:
        return float(s)
    except ValueError:
        return np.nan


def _is_numeric_like(series, threshold=0.8):
    if pd.api.types.is_numeric_dtype(series):
        return True

    s = series.dropna()
    if len(s) == 0:
        return False

    sample = s.sample(min(len(s), 1000), random_state=RANDOM_STATE) if len(s) > 1000 else s
    parsed = sample.map(_clean_numeric_value)
    return parsed.notna().mean() >= threshold


def _is_date_like(series, threshold=0.8):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    s = series.dropna()
    if len(s) == 0:
        return False

    sample = s.sample(min(len(s), 1000), random_state=RANDOM_STATE) if len(s) > 1000 else s
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= threshold


class FeatureProcessor:
    def __init__(self):
        self.feature_cols = []
        self.numeric_cols = []
        self.date_cols = []
        self.cat_cols = []
        self.derived_cols = []
        self.period_start = None
        self.period_end = None
        self.encoder = None
        self.target_mean = np.nan

    def _clean_categorical_frame(self, df_cat):
        out = df_cat.copy()
        for col in out.columns:
            out[col] = out[col].astype(str)
            out[col] = out[col].str.strip()
            out[col] = out[col].replace({"": "missing", "nan": "missing", "None": "missing", "<NA>": "missing"})
        return out

    def fit(self, train_df):
        df = train_df.copy()
        if TARGET in df.columns:
            self.target_mean = float(df[TARGET].mean())
            df = df.drop(columns=[TARGET])

        self.feature_cols = list(df.columns)

        for col in self.feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                self.date_cols.append(col)
            elif _is_numeric_like(df[col]):
                self.numeric_cols.append(col)
            elif _is_date_like(df[col]):
                self.date_cols.append(col)
            else:
                self.cat_cols.append(col)

        for col in self.date_cols:
            for part in ["year", "month", "day", "quarter", "dayofweek"]:
                self.derived_cols.append(f"{col}_{part}")

        start_cols = [c for c in self.date_cols if "start" in c.lower()]
        end_cols = [c for c in self.date_cols if "end" in c.lower()]
        if start_cols and end_cols:
            self.period_start = start_cols[0]
            self.period_end = end_cols[0]
            self.derived_cols.append("MLR_Period_Days")

        if self.cat_cols:
            X_cat = self._clean_categorical_frame(df[self.cat_cols])
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.encoder.fit(X_cat)

        return self

    def transform(self, df):
        df = df.copy()
        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])

        for col in self.feature_cols:
            if col not in df.columns:
                if col in self.numeric_cols or col in self.date_cols:
                    df[col] = np.nan
                else:
                    df[col] = "missing"

        for col in self.numeric_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].map(_clean_numeric_value)

        date_dts = {}
        for col in self.date_cols:
            dt = pd.to_datetime(df[col], errors="coerce")
            date_dts[col] = dt

            df[f"{col}_year"] = dt.dt.year
            df[f"{col}_month"] = dt.dt.month
            df[f"{col}_day"] = dt.dt.day
            df[f"{col}_quarter"] = dt.dt.quarter
            df[f"{col}_dayofweek"] = dt.dt.dayofweek

            df[col] = dt.map(lambda x: x.toordinal() if pd.notnull(x) else np.nan).astype(float)

        if self.period_start is not None and self.period_end is not None:
            delta = date_dts[self.period_end] - date_dts[self.period_start]
            df["MLR_Period_Days"] = delta.dt.days

        if self.cat_cols:
            X_cat = self._clean_categorical_frame(df[self.cat_cols])
            encoded = self.encoder.transform(X_cat)
            df[self.cat_cols] = encoded

        cols = list(dict.fromkeys(self.feature_cols + self.derived_cols))
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan

        return df[cols]


def _make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
    }


def _fit_with_early_stopping(model, X_train, y_train, X_val, y_val):
    if model.__class__.__name__ == "LGBMRegressor":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=False,
        )
    elif model.__class__.__name__ == "XGBRegressor":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=50,
            verbose=False,
        )
    elif model.__class__.__name__ == "CatBoostRegressor":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            metric="RMSE",
            early_stopping_rounds=50,
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    return model


def _best_iteration(model, default):
    best = getattr(model, "best_iteration_", None)
    if best is None:
        return default
    try:
        best = int(best)
    except Exception:
        return default
    return max(1, best + 1)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train_df = train_df.dropna(subset=[TARGET])
    y_train = train_df[TARGET].astype(float)

    processor = FeatureProcessor().fit(train_df)
    X_train = processor.transform(train_df)
    X_test = processor.transform(test_df)

    keep_cols = []
    for col in X_train.columns:
        if X_train[col].notna().sum() == 0:
            continue
        if X_train[col].nunique(dropna=True) <= 1:
            continue
        keep_cols.append(col)

    if len(keep_cols) == 0:
        predictions = np.full(len(X_test), processor.target_mean if np.isfinite(processor.target_mean) else 0.0)
    else:
        X_train = X_train[keep_cols]
        X_test = X_test[keep_cols]

        if len(X_train) > 200:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=RANDOM_STATE,
            )
        else:
            X_fit, X_val, y_fit, y_val = X_train, X_train, y_train, y_train

        val_predictions = []
        val_weights = []

        for name, model in _make_models().items():
            try:
                model = _fit_with_early_stopping(model, X_fit, y_fit, X_val, y_val)
                pred_val = model.predict(X_val)
                if not np.all(np.isfinite(pred_val)):
                    continue

                rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
                if not np.isfinite(rmse):
                    continue

                best_iter = _best_iteration(model, 1500)

                final_model = model.__class__(**{
                    k: v for k, v in model.get_params().items()
                    if k not in {"n_estimators", "iterations"}
                })
                if final_model.__class__.__name__ == "LGBMRegressor":
                    final_model.set_params(n_estimators=best_iter)
                elif final_model.__class__.__name__ == "XGBRegressor":
                    final_model.set_params(n_estimators=best_iter)
                elif final_model.__class__.__name__ == "CatBoostRegressor":
                    final_model.set_params(iterations=best_iter)

                final_model.fit(X_train, y_train)
                pred_test = final_model.predict(X_test)

                if not np.all(np.isfinite(pred_test)):
                    continue

                val_predictions.append(pred_test)
                val_weights.append(1.0 / max(rmse, 1e-8))
            except Exception:
                continue

        if len(val_predictions) == 0:
            predictions = np.full(len(X_test), processor.target_mean if np.isfinite(processor.target_mean) else 0.0)
        else:
            weights = np.array(val_weights, dtype=float)
            weights = weights / weights.sum()
            predictions = np.average(np.vstack(val_predictions), axis=0, weights=weights)

    predictions = np.asarray(predictions, dtype=float)
    predictions[~np.isfinite(predictions)] = processor.target_mean if np.isfinite(processor.target_mean) else 0.0

    output = pd.DataFrame({
        "id": np.arange(len(predictions), dtype=int),
        "Adjusted MLR": predictions,
    })
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()