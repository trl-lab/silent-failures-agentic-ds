import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/recipient-executed-grants-commitments-disbursements/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/recipient-executed-grants-commitments-disbursements/data/test.parquet"
TARGET = "Disbursements USD (Trust funds)"
RANDOM_STATE = 42
MODEL_NAMES = ["lightgbm", "xgboost", "catboost"]


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class FeatureBuilder:
    def __init__(self, drop_cols, date_col):
        self.drop_cols = list(drop_cols)
        self.date_col = date_col
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.frequency_maps = {}
        self.feature_columns = []

    def _prepare(self, df):
        x = df.copy()

        for col in self.drop_cols:
            if col in x.columns:
                x = x.drop(columns=[col])

        if self.date_col in x.columns:
            parsed_date = pd.to_datetime(x[self.date_col], errors="coerce")
            x["Agreement_Year"] = parsed_date.dt.year
            x["Agreement_Month"] = parsed_date.dt.month
            x["Agreement_Day"] = parsed_date.dt.day
            x["Agreement_Date_Missing"] = parsed_date.isna().astype(int)
            x = x.drop(columns=[self.date_col])

        if "Fiscal Year" in x.columns and "Agreement_Year" in x.columns:
            x["Fiscal_Year_Minus_Agreement_Year"] = (
                pd.to_numeric(x["Fiscal Year"], errors="coerce")
                - pd.to_numeric(x["Agreement_Year"], errors="coerce")
            )

        if "Grant Commitments USD" in x.columns:
            commitments = pd.to_numeric(x["Grant Commitments USD"], errors="coerce")
            x["Grant_Commitments_Log"] = np.log1p(commitments.clip(lower=0.0))
            x["Grant_Commitments_Missing"] = commitments.isna().astype(int)

        return x

    def fit(self, raw_train):
        x = self._prepare(raw_train)

        self.numeric_cols = [
            col for col in x.columns if pd.api.types.is_numeric_dtype(x[col])
        ]
        self.categorical_cols = [
            col for col in x.columns if not pd.api.types.is_numeric_dtype(x[col])
        ]

        if not self.numeric_cols and not self.categorical_cols:
            self.numeric_cols = ["__constant__"]
            self.medians["__constant__"] = 1.0
            self.feature_columns = ["__constant__"]
            return self

        for col in self.numeric_cols:
            values = pd.to_numeric(x[col], errors="coerce")
            median = values.median()
            self.medians[col] = 0.0 if pd.isna(median) else float(median)

        for col in self.categorical_cols:
            values = x[col].astype("string").fillna("missing")
            self.frequency_maps[col] = values.value_counts(dropna=False).to_dict()

        self.feature_columns = self.numeric_cols + self.categorical_cols
        return self

    def transform(self, raw_df):
        x = self._prepare(raw_df)

        for col in self.numeric_cols:
            if col not in x.columns:
                x[col] = np.nan
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[col] = x[col].replace([np.inf, -np.inf], np.nan)
            x[col] = x[col].fillna(self.medians[col])

        for col in self.categorical_cols:
            if col not in x.columns:
                x[col] = "missing"
            x[col] = x[col].astype("string").fillna("missing")
            x[col] = x[col].map(self.frequency_maps[col]).fillna(0.0).astype(float)

        x = x[self.feature_columns]
        x = x.apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0.0)
        return x


def make_models():
    return {
        "lightgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def safe_predict(model, x, fallback):
    try:
        pred = model.predict(x)
        pred = np.asarray(pred, dtype=float)
        return np.nan_to_num(pred, nan=fallback, posinf=fallback, neginf=fallback)
    except Exception:
        return np.full(len(x), fallback, dtype=float)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df[TARGET] = train_df[TARGET].replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)

    if train_df.empty:
        pd.DataFrame(
            {"id": np.arange(len(test_df), dtype=int), TARGET: 0.0}
        ).to_csv("predictions.csv", index=False)
        return

    drop_cols = ["Project ID", "Project Description"]
    date_col = "Grant Agreement Date"

    x_raw_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(float)
    fallback = float(y_train.mean()) if len(y_train) > 0 else 0.0

    weights = {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES}

    if len(train_df) >= 20:
        x_train_raw, x_val_raw, y_train_split, y_val_split = train_test_split(
            x_raw_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        validation_builder = FeatureBuilder(drop_cols=drop_cols, date_col=date_col)
        validation_builder.fit(x_train_raw)

        x_train = validation_builder.transform(x_train_raw)
        x_val = validation_builder.transform(x_val_raw)

        val_rmses = {}
        for name in MODEL_NAMES:
            try:
                model = make_models()[name]
                model.fit(x_train, y_train_split)
                pred = safe_predict(model, x_val, fallback)
                val_rmses[name] = rmse(y_val_split, pred)
            except Exception:
                val_rmses[name] = np.inf

        finite_scores = {
            name: score for name, score in val_rmses.items() if np.isfinite(score)
        }

        if finite_scores:
            weights = {}
            for name, score in val_rmses.items():
                if np.isfinite(score):
                    weights[name] = 1.0 / max(float(score), 1e-12)
                else:
                    weights[name] = 0.0

            total_weight = sum(weights.values())
            if total_weight > 0 and np.isfinite(total_weight):
                weights = {name: weight / total_weight for name, weight in weights.items()}
            else:
                weights = {name: 1.0 / len(MODEL_NAMES) for name in MODEL_NAMES}

    final_builder = FeatureBuilder(drop_cols=drop_cols, date_col=date_col)
    final_builder.fit(x_raw_train)

    x_full_train = final_builder.transform(x_raw_train)
    x_test = final_builder.transform(test_df)

    final_predictions = np.zeros(len(x_test), dtype=float)

    for name in MODEL_NAMES:
        try:
            model = make_models()[name]
            model.fit(x_full_train, y_train)
            pred = safe_predict(model, x_test, fallback)
        except Exception:
            pred = np.full(len(x_test), fallback, dtype=float)

        final_predictions += weights[name] * pred

    if np.isfinite(fallback) and y_train.min() >= 0:
        final_predictions = np.clip(final_predictions, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            TARGET: final_predictions,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()