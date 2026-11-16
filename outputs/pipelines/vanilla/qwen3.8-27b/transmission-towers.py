import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/transmission-towers/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/transmission-towers/data/test.parquet"
TARGET = "SupStruc"
RANDOM_STATE = 42
DROP_COLS = {"FID", "UniqSysID", "Callsign", "url", "LocAdd"}


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clean_frame(df, drop_cols):
    df = df.copy()
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_feature_columns(train_df):
    numeric_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns if c != TARGET
    ]
    categorical_cols = [
        c
        for c in train_df.select_dtypes(include=["object", "category"]).columns
        if c != TARGET
    ]
    return numeric_cols, categorical_cols


class FeatureBuilder:
    def __init__(self, numeric_cols, categorical_cols, onehot_threshold=100):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.onehot_threshold = onehot_threshold
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.onehot_columns = {}
        self.frequency_maps = {}
        self.feature_columns = []

    def _clean_categorical_series(self, df, col):
        if col in df.columns:
            s = df[col]
        else:
            s = pd.Series(index=df.index, dtype=object)
        s = s.fillna("__MISSING__").astype(str)
        return s

    def fit(self, df):
        if self.numeric_cols:
            numeric_df = df.reindex(columns=self.numeric_cols).copy()
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
            self.numeric_imputer.fit(numeric_df)

        self.feature_columns = list(self.numeric_cols)

        for col in self.categorical_cols:
            s = self._clean_categorical_series(df, col)
            nunique = s.nunique()

            if nunique <= self.onehot_threshold:
                dummies = pd.get_dummies(s, prefix=col, dtype=float)
                self.onehot_columns[col] = list(dummies.columns)
                self.feature_columns.extend(dummies.columns)
            else:
                freq = s.value_counts()
                self.frequency_maps[col] = freq
                feat_name = f"{col}__freq"
                self.feature_columns.append(feat_name)

        self.feature_columns = list(dict.fromkeys(self.feature_columns))
        return self

    def transform(self, df):
        parts = []

        if self.numeric_cols:
            numeric_df = df.reindex(columns=self.numeric_cols).copy()
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
            numeric_transformed = self.numeric_imputer.transform(numeric_df)
            parts.append(
                pd.DataFrame(
                    numeric_transformed,
                    columns=self.numeric_cols,
                    index=df.index,
                )
            )
        else:
            parts.append(pd.DataFrame(index=df.index))

        for col in self.categorical_cols:
            s = self._clean_categorical_series(df, col)

            if col in self.onehot_columns:
                dummies = pd.get_dummies(s, prefix=col, dtype=float)
                dummies = dummies.reindex(columns=self.onehot_columns[col], fill_value=0)
                parts.append(dummies)
            elif col in self.frequency_maps:
                feat_name = f"{col}__freq"
                freq_values = s.map(self.frequency_maps[col]).fillna(0).astype(float)
                parts.append(
                    pd.DataFrame({feat_name: np.log1p(freq_values)}, index=df.index)
                )

        out = pd.concat(parts, axis=1)
        out = out.reindex(columns=self.feature_columns, fill_value=0.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.fillna(0.0)
        return out


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3.0,
            subsample=0.9,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
    }


def safe_predict(model, X, fallback):
    pred = model.predict(X)
    pred = np.asarray(pred, dtype=float).ravel()

    if pred.size != X.shape[0]:
        if pred.size == 1:
            pred = np.full(X.shape[0], float(pred[0]))
        else:
            pred = np.full(X.shape[0], fallback)

    pred = np.where(np.isfinite(pred), pred, fallback)
    return pred


def write_predictions(test_len, values):
    values = np.asarray(values, dtype=float).ravel()

    if values.size != test_len:
        if values.size == 1:
            values = np.full(test_len, float(values[0]))
        else:
            values = np.zeros(test_len, dtype=float)

    values = np.where(np.isfinite(values), values, 0.0)

    pd.DataFrame(
        {
            "id": np.arange(test_len, dtype=int),
            "SupStruc": values,
        }
    ).to_csv("predictions.csv", index=False)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df = clean_frame(train_df, DROP_COLS)
    test_df = clean_frame(test_df, DROP_COLS)

    if len(test_df) == 0:
        write_predictions(0, np.array([]))
        return

    if TARGET not in train_df.columns:
        write_predictions(len(test_df), 0.0)
        return

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=[TARGET])

    if train_df.empty:
        write_predictions(len(test_df), 0.0)
        return

    y_all = train_df[TARGET].astype(float)
    X_all = train_df.drop(columns=[TARGET])

    numeric_cols, categorical_cols = get_feature_columns(train_df)

    feature_builder = FeatureBuilder(
        numeric_cols,
        categorical_cols,
        onehot_threshold=100,
    ).fit(X_all)

    X_all_processed = feature_builder.transform(X_all)
    X_test_processed = feature_builder.transform(test_df)

    if X_all_processed.shape[1] == 0:
        X_all_processed = pd.DataFrame({"const": 1.0}, index=X_all_processed.index)

    if X_test_processed.shape[1] == 0:
        X_test_processed = pd.DataFrame({"const": 1.0}, index=X_test_processed.index)

    if list(X_all_processed.columns) != list(X_test_processed.columns):
        X_test_processed = X_test_processed.reindex(
            columns=X_all_processed.columns,
            fill_value=0.0,
        )

    fallback = float(y_all.median()) if len(y_all) > 0 else 0.0

    if len(train_df) > 50:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all_processed,
            y_all,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    else:
        X_train, X_val, y_train, y_val = (
            X_all_processed,
            X_all_processed,
            y_all,
            y_all,
        )

    models = make_models()
    val_predictions = {}
    val_scores = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = safe_predict(model, X_val, fallback)
        except Exception:
            pred = np.full(len(y_val), fallback)

        val_predictions[name] = pred
        val_scores[name] = rmse(y_val, pred)

    finite_scores = {
        k: v for k, v in val_scores.items() if np.isfinite(v) and v > 0
    }

    if finite_scores:
        inv_scores = {k: 1.0 / v for k, v in finite_scores.items()}
        total = sum(inv_scores.values())
        weights = {k: v / total for k, v in inv_scores.items()}
    else:
        weights = {name: 1.0 / len(models) for name in models}

    final_models = make_models()
    test_predictions = np.zeros(len(test_df), dtype=float)

    for name, model in final_models.items():
        try:
            model.fit(X_all_processed, y_all)
            pred = safe_predict(model, X_test_processed, fallback)
        except Exception:
            pred = np.full(len(test_df), fallback)

        test_predictions += weights.get(name, 0.0) * pred

    test_predictions = np.where(np.isfinite(test_predictions), test_predictions, fallback)
    write_predictions(len(test_df), test_predictions)


if __name__ == "__main__":
    main()