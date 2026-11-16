import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import xgboost as xgb

try:
    from scipy.optimize import nnls
except ImportError:
    nnls = None

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:

    def root_mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
SUBMISSION_DIR = "./submission"
TARGET = "Recovery"
RANDOM_STATE = 42
VAL_FRACTION = 0.2
SMOOTHING = 10.0
MAX_TFIDF_FEATURES_PER_TEXT_COL = 30

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def safe_read_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame from {path}, got {type(df)}")
    return df.copy()


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def to_datetime_mixed(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, errors="coerce")


def tfidf_feature_names(vec: TfidfVectorizer):
    if hasattr(vec, "get_feature_names_out"):
        return vec.get_feature_names_out()
    return vec.get_feature_names()


def clean_basic(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()

    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=int)
    else:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype("int64")
        if (df["id"] < 0).any():
            df["id"] = np.arange(len(df), dtype=int)

    if target in df.columns:
        df[target] = coerce_numeric_series(df[target])

    for col in df.columns:
        if col in {"id", target}:
            continue

        if df[col].dtype == object or str(df[col].dtype) == "string":
            s = df[col].astype("string").str.strip()
            s = s.replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "NaN": pd.NA,
                    "null": pd.NA,
                    "None": pd.NA,
                }
            )
            df[col] = s

        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = coerce_numeric_series(df[col])

        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("object")

    empty_cols = [
        c for c in df.columns if c not in {"id", target} and df[c].isna().all()
    ]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    return df


def is_date_like(s: pd.Series) -> bool:
    if s.isna().all():
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = to_datetime_mixed(sample)
    return parsed.notna().mean() >= 0.8


def is_text_like(s: pd.Series, n_rows: int) -> bool:
    if s.isna().all():
        return False
    sample = s.dropna().astype(str)
    if sample.empty:
        return False
    mean_len = sample.str.len().mean()
    unique_ratio = sample.nunique() / max(1, n_rows)
    return unique_ratio >= 0.6 and mean_len >= 20


def classify_columns(train: pd.DataFrame, target: str) -> dict:
    numeric_cols, categorical_cols, date_cols, text_cols = [], [], [], []

    for col in train.columns:
        if col in {"id", target}:
            continue
        s = train[col]

        if pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) > 1:
                numeric_cols.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(col)
            continue

        if s.dtype == object or str(s.dtype) == "string":
            if is_date_like(s):
                date_cols.append(col)
            elif is_text_like(s, len(train)):
                text_cols.append(col)
            else:
                categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols,
        "text": text_cols,
    }


class FeatureBuilder:
    def __init__(self):
        self.feature_names = []
        self.col_types = {}
        self.numeric_stats = {}
        self.numeric_top_interaction_cols = []
        self.cat_mappings = {}
        self.date_stats = {}
        self.text_vectorizers = {}
        self.global_target_mean = 0.0
        self.smoothing = SMOOTHING

    def _fit_numeric(self, train: pd.DataFrame, cols):
        out = pd.DataFrame(index=train.index)
        self.numeric_stats = {}
        num_missing_count = 0

        for col in cols:
            s = coerce_numeric_series(train[col])
            missing = s.isna().astype(int)
            median = float(s.median()) if s.notna().any() else 0.0
            q01 = (
                float(s.quantile(0.001))
                if s.notna().sum() >= 100
                else (float(s.min()) if s.notna().any() else 0.0)
            )
            q99 = (
                float(s.quantile(0.999))
                if s.notna().sum() >= 100
                else (float(s.max()) if s.notna().any() else 0.0)
            )
            if not np.isfinite(q01):
                q01 = median
            if not np.isfinite(q99):
                q99 = median
            if q01 > q99:
                q01, q99 = q99, q01

            self.numeric_stats[col] = {"median": median, "q01": q01, "q99": q99}

            imp = s.fillna(median).clip(q01, q99)
            out[col] = imp
            out[f"{col}__missing"] = missing
            out[f"{col}__signed_log"] = np.sign(imp) * np.log1p(np.abs(imp))
            out[f"{col}__sqrt"] = np.sign(imp) * np.sqrt(np.abs(imp))

        if len(cols) > 0:
            num_matrix = out[cols].to_numpy(dtype=float)
            out["num__row_mean"] = np.nanmean(num_matrix, axis=1)
            out["num__row_std"] = np.nanstd(num_matrix, axis=1)
            out["num__row_max"] = np.nanmax(num_matrix, axis=1)
            out["num__row_min"] = np.nanmin(num_matrix, axis=1)
            out["num__row_range"] = out["num__row_max"] - out["num__row_min"]
            out["num__row_median"] = np.nanmedian(num_matrix, axis=1)
            num_missing_count = out[[f"{c}__missing" for c in cols]].sum(axis=1)

            variances = out[cols].var(axis=0, skipna=True).fillna(0.0)
            top_interaction_cols = list(variances.sort_values(ascending=False).head(5).index)
            self.numeric_top_interaction_cols = top_interaction_cols
            eps = 1e-6
            for i in range(len(top_interaction_cols)):
                for j in range(i + 1, len(top_interaction_cols)):
                    a = top_interaction_cols[i]
                    b = top_interaction_cols[j]
                    out[f"{a}__x__{b}"] = out[a] * out[b]
                    out[f"{a}__plus__{b}"] = out[a] + out[b]
                    out[f"{a}__minus__{b}"] = (out[a] - out[b]).abs()
                    out[f"{a}__div__{b}"] = out[a] / (out[b] + eps)
        else:
            self.numeric_top_interaction_cols = []

        out["num__missing_count"] = num_missing_count
        return out

    def _transform_numeric(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in cols:
            if col not in df.columns:
                s = pd.Series(np.nan, index=df.index, dtype=float)
            else:
                s = coerce_numeric_series(df[col])

            stats = self.numeric_stats.get(col, {"median": 0.0, "q01": 0.0, "q99": 0.0})
            missing = s.isna().astype(int)
            imp = s.fillna(stats["median"]).clip(stats["q01"], stats["q99"])

            out[col] = imp
            out[f"{col}__missing"] = missing
            out[f"{col}__signed_log"] = np.sign(imp) * np.log1p(np.abs(imp))
            out[f"{col}__sqrt"] = np.sign(imp) * np.sqrt(np.abs(imp))

        if len(cols) > 0:
            num_matrix = out[cols].to_numpy(dtype=float)
            out["num__row_mean"] = np.nanmean(num_matrix, axis=1)
            out["num__row_std"] = np.nanstd(num_matrix, axis=1)
            out["num__row_max"] = np.nanmax(num_matrix, axis=1)
            out["num__row_min"] = np.nanmin(num_matrix, axis=1)
            out["num__row_range"] = out["num__row_max"] - out["num__row_min"]
            out["num__row_median"] = np.nanmedian(num_matrix, axis=1)
            out["num__missing_count"] = out[[f"{c}__missing" for c in cols]].sum(axis=1)

            top_interaction_cols = getattr(self, "numeric_top_interaction_cols", [])
            eps = 1e-6
            for i in range(len(top_interaction_cols)):
                for j in range(i + 1, len(top_interaction_cols)):
                    a = top_interaction_cols[i]
                    b = top_interaction_cols[j]
                    if a in out.columns and b in out.columns:
                        out[f"{a}__x__{b}"] = out[a] * out[b]
                        out[f"{a}__plus__{b}"] = out[a] + out[b]
                        out[f"{a}__minus__{b}"] = (out[a] - out[b]).abs()
                        out[f"{a}__div__{b}"] = out[a] / (out[b] + eps)
        else:
            out["num__missing_count"] = 0.0
        return out

    def _fit_categorical(self, train: pd.DataFrame, y: pd.Series, cols):
        out = pd.DataFrame(index=train.index)
        cat_missing_count = 0

        for col in cols:
            s = train[col].astype("string").fillna("__MISSING__")
            missing = (
                train[col].isna().astype(int)
                if col in train.columns
                else pd.Series(0, index=train.index, dtype=int)
            )

            value_counts = s.value_counts(dropna=False)
            n = max(1, len(s))
            freq = value_counts / n

            cat_series = s.copy()
            cat_series.name = "__cat__"
            sums = (
                pd.DataFrame({"__cat__": cat_series, "__y__": y.to_numpy(dtype=float)})
                .groupby("__cat__")["__y__"]
                .agg(["sum", "count", "mean"])
            )
            global_mean = float(y.mean()) if y.notna().any() else 0.0
            self.global_target_mean = global_mean

            target_enc = (sums["sum"] + self.smoothing * global_mean) / (
                sums["count"] + self.smoothing
            )

            self.cat_mappings[col] = {
                "categories": sorted([str(x) for x in s.unique()]),
                "freq": freq.to_dict(),
                "target_enc": target_enc.to_dict(),
                "global_mean": global_mean,
            }

            out[f"{col}__missing"] = missing
            out[f"{col}__freq"] = s.map(freq).fillna(0.0).astype(float)
            out[f"{col}__target_enc"] = (
                s.map(target_enc).fillna(global_mean).astype(float)
            )

            cat_codes = pd.Categorical(
                s, categories=sorted([str(x) for x in s.unique()])
            )
            out[f"{col}__ord"] = cat_codes.codes.astype(float)

            if s.nunique(dropna=True) <= 12:
                dummies = pd.get_dummies(s, prefix=col, dummy_na=False).astype(float)
                dummies.columns = [f"{c}__oh" for c in dummies.columns]
                out = pd.concat([out, dummies], axis=1)

            cat_missing_count += missing

        out["cat__missing_count"] = cat_missing_count
        return out

    def _transform_categorical(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        cat_missing_count = 0

        for col in cols:
            mapping = self.cat_mappings[col]
            if col in df.columns:
                s = df[col].astype("string").fillna("__MISSING__")
                missing = df[col].isna().astype(int)
            else:
                s = pd.Series("__MISSING__", index=df.index, dtype="string")
                missing = pd.Series(1, index=df.index, dtype=int)

            freq = pd.Series(mapping["freq"])
            target_enc = pd.Series(mapping["target_enc"])
            global_mean = mapping["global_mean"]
            categories = mapping["categories"]

            out[f"{col}__missing"] = missing
            out[f"{col}__freq"] = s.map(freq).fillna(0.0).astype(float)
            out[f"{col}__target_enc"] = (
                s.map(target_enc).fillna(global_mean).astype(float)
            )

            cat_codes = pd.Categorical(s, categories=categories)
            out[f"{col}__ord"] = cat_codes.codes.astype(float)

            if len(categories) <= 12:
                dummies = pd.get_dummies(s, prefix=col, dummy_na=False).astype(float)
                dummies.columns = [f"{c}__oh" for c in dummies.columns]

                expected = [
                    f"{c}__oh"
                    for c in pd.get_dummies(
                        pd.Series(categories, dtype="string"),
                        prefix=col,
                        dummy_na=False,
                    ).columns
                ]
                for c in expected:
                    if c not in dummies.columns:
                        dummies[c] = 0.0
                out = pd.concat([out, dummies[expected]], axis=1)

            cat_missing_count += missing

        out["cat__missing_count"] = cat_missing_count
        return out

    def _fit_date(self, train: pd.DataFrame, cols):
        out = pd.DataFrame(index=train.index)
        date_missing_count = 0

        for col in cols:
            s = to_datetime_mixed(train[col])
            missing = s.isna().astype(int)
            min_date = s.min() if s.notna().any() else pd.Timestamp("1970-01-01")
            max_date = s.max() if s.notna().any() else pd.Timestamp("1970-01-01")
            median_date = s.median() if s.notna().any() else min_date

            self.date_stats[col] = {
                "min": str(min_date),
                "max": str(max_date),
                "median": str(median_date),
            }

            imp = s.fillna(median_date)
            out[f"{col}__missing"] = missing
            out[f"{col}__year"] = imp.dt.year.astype(float)
            out[f"{col}__month"] = imp.dt.month.astype(float)
            out[f"{col}__day"] = imp.dt.day.astype(float)
            out[f"{col}__dayofweek"] = imp.dt.dayofweek.astype(float)
            out[f"{col}__quarter"] = imp.dt.quarter.astype(float)
            out[f"{col}__dayofyear"] = imp.dt.dayofyear.astype(float)
            out[f"{col}__is_weekend"] = (imp.dt.dayofweek >= 5).astype(float)
            out[f"{col}__days_since_min"] = (imp - min_date).dt.days.astype(float)
            out[f"{col}__days_to_max"] = (max_date - imp).dt.days.astype(float)
            out[f"{col}__days_from_median"] = (imp - median_date).dt.days.astype(float)

            date_missing_count += missing

        out["date__missing_count"] = date_missing_count
        return out

    def _transform_date(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        date_missing_count = 0

        for col in cols:
            stats = self.date_stats[col]
            min_date = pd.Timestamp(stats["min"])
            max_date = pd.Timestamp(stats["max"])
            median_date = pd.Timestamp(stats["median"])

            if col in df.columns:
                s = to_datetime_mixed(df[col])
                missing = s.isna().astype(int)
            else:
                s = pd.Series(pd.NaT, index=df.index)
                missing = pd.Series(1, index=df.index, dtype=int)

            imp = s.fillna(median_date)
            out[f"{col}__missing"] = missing
            out[f"{col}__year"] = imp.dt.year.astype(float)
            out[f"{col}__month"] = imp.dt.month.astype(float)
            out[f"{col}__day"] = imp.dt.day.astype(float)
            out[f"{col}__dayofweek"] = imp.dt.dayofweek.astype(float)
            out[f"{col}__quarter"] = imp.dt.quarter.astype(float)
            out[f"{col}__dayofyear"] = imp.dt.dayofyear.astype(float)
            out[f"{col}__is_weekend"] = (imp.dt.dayofweek >= 5).astype(float)
            out[f"{col}__days_since_min"] = (imp - min_date).dt.days.astype(float)
            out[f"{col}__days_to_max"] = (max_date - imp).dt.days.astype(float)
            out[f"{col}__days_from_median"] = (imp - median_date).dt.days.astype(float)

            date_missing_count += missing

        out["date__missing_count"] = date_missing_count
        return out

    def _fit_text(self, train: pd.DataFrame, cols):
        out = pd.DataFrame(index=train.index)
        text_missing_count = 0

        for col in cols:
            s = train[col].astype("string").fillna("")
            missing = (
                train[col].isna().astype(int)
                if col in train.columns
                else pd.Series(1, index=train.index, dtype=int)
            )

            out[f"{col}__missing"] = missing
            out[f"{col}__len"] = s.str.len().astype(float)
            out[f"{col}__word_count"] = s.str.split().str.len().fillna(0).astype(float)
            out[f"{col}__avg_word_len"] = (
                out[f"{col}__len"] / (out[f"{col}__word_count"] + 1.0)
            ).astype(float)
            out[f"{col}__digit_count"] = s.str.count(r"\d").astype(float)
            out[f"{col}__upper_count"] = s.str.count(r"[A-Z]").astype(float)
            out[f"{col}__punct_count"] = s.str.count(r"[^\w\s]").astype(float)
            out[f"{col}__unique_char_ratio"] = s.apply(
                lambda x: len(set(x)) / max(1, len(x))
            ).astype(float)

            try:
                vec = TfidfVectorizer(
                    max_features=MAX_TFIDF_FEATURES_PER_TEXT_COL,
                    ngram_range=(1, 2),
                    min_df=2,
                    strip_accents="unicode",
                    lowercase=True,
                    sublinear_tf=True,
                )
                X = vec.fit_transform(s.astype(str))
                feature_names = tfidf_feature_names(vec)

                if len(feature_names) > 0:
                    X_dense = X.toarray().astype(float)
                    for j, fname in enumerate(feature_names):
                        out[f"{col}__tfidf__{j}"] = X_dense[:, j]
                    self.text_vectorizers[col] = vec
                else:
                    self.text_vectorizers[col] = None
            except Exception:
                self.text_vectorizers[col] = None

            text_missing_count += missing

        out["text__missing_count"] = text_missing_count
        return out

    def _transform_text(self, df: pd.DataFrame, cols) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        text_missing_count = 0

        for col in cols:
            if col in df.columns:
                s = df[col].astype("string").fillna("")
                missing = df[col].isna().astype(int)
            else:
                s = pd.Series("", index=df.index, dtype="string")
                missing = pd.Series(1, index=df.index, dtype=int)

            out[f"{col}__missing"] = missing
            out[f"{col}__len"] = s.str.len().astype(float)
            out[f"{col}__word_count"] = s.str.split().str.len().fillna(0).astype(float)
            out[f"{col}__avg_word_len"] = (
                out[f"{col}__len"] / (out[f"{col}__word_count"] + 1.0)
            ).astype(float)
            out[f"{col}__digit_count"] = s.str.count(r"\d").astype(float)
            out[f"{col}__upper_count"] = s.str.count(r"[A-Z]").astype(float)
            out[f"{col}__punct_count"] = s.str.count(r"[^\w\s]").astype(float)
            out[f"{col}__unique_char_ratio"] = s.apply(
                lambda x: len(set(x)) / max(1, len(x))
            ).astype(float)

            vec = self.text_vectorizers.get(col)
            if vec is not None:
                try:
                    feature_names = tfidf_feature_names(vec)
                    if len(feature_names) > 0:
                        X = vec.transform(s.astype(str))
                        X_dense = X.toarray().astype(float)
                        for j, fname in enumerate(feature_names):
                            out[f"{col}__tfidf__{j}"] = X_dense[:, j]
                except Exception:
                    pass

            text_missing_count += missing

        out["text__missing_count"] = text_missing_count
        return out

    def fit(self, train: pd.DataFrame, y: pd.Series, col_types: dict):
        self.col_types = col_types
        parts = []

        if col_types["numeric"]:
            parts.append(self._fit_numeric(train, col_types["numeric"]))
        if col_types["categorical"]:
            parts.append(self._fit_categorical(train, y, col_types["categorical"]))
        if col_types["date"]:
            parts.append(self._fit_date(train, col_types["date"]))
        if col_types["text"]:
            parts.append(self._fit_text(train, col_types["text"]))

        if not parts:
            X = pd.DataFrame(index=train.index)
            X["no_features"] = 0.0
        else:
            X = pd.concat(parts, axis=1)

        X = X.loc[:, X.std(axis=0, skipna=True).fillna(0) > 0]
        X = X.fillna(0.0)
        X = X.replace([np.inf, -np.inf], 0.0)

        self.feature_names = list(X.columns)
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if self.col_types["numeric"]:
            parts.append(self._transform_numeric(df, self.col_types["numeric"]))
        if self.col_types["categorical"]:
            parts.append(self._transform_categorical(df, self.col_types["categorical"]))
        if self.col_types["date"]:
            parts.append(self._transform_date(df, self.col_types["date"]))
        if self.col_types["text"]:
            parts.append(self._transform_text(df, self.col_types["text"]))

        if not parts:
            X = pd.DataFrame(index=df.index)
            X["no_features"] = 0.0
        else:
            X = pd.concat(parts, axis=1)

        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names].copy()
        X = X.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return X


def build_lgbm_model_a() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=1e-3,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def build_lgbm_model_b() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=10,
        min_child_weight=1e-3,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgb_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_estimators=3000,
        early_stopping_rounds=60,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def build_criterion():
    return root_mean_squared_error


def build_optimizer():
    return {
        "optimizer": "lightgbm_histogram_gradient_descent",
        "learning_rate": 0.03,
        "n_estimators": 3000,
        "early_stopping_rounds": 60,
        "row_sampling": 0.9,
        "feature_sampling": 0.8,
        "l2_regularization": 1.0,
        "l1_regularization": 0.05,
    }


def sanitize_predictions(pred: np.ndarray, y_train: pd.Series) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(
        pred,
        nan=float(y_train.mean()),
        posinf=float(y_train.max()),
        neginf=float(y_train.min()),
    )
    low = float(y_train.min())
    high = float(y_train.max())
    if low > high:
        low, high = high, low
    return np.clip(pred, low, high)


class RMSETracker:
    def __init__(self):
        self.best_rmse = np.inf
        self.best_iteration = 0

    def __call__(self, env):
        if not env.evaluation_result_list:
            return
        rmse_value = float(env.evaluation_result_list[0][2])
        if np.isfinite(rmse_value) and rmse_value < self.best_rmse:
            self.best_rmse = rmse_value
            self.best_iteration = int(env.iteration + 1)


def main():
    train_raw = safe_read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
    test_raw = safe_read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

    train = clean_basic(train_raw, TARGET)
    test = clean_basic(test_raw, TARGET)

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    if len(train) == 0:
        raise ValueError("No valid training rows with non-null target.")

    # Submission id is specified as 0-based row index.
    test["id"] = np.arange(len(test), dtype=int)

    train_idx, val_idx = train_test_split(
        np.arange(len(train)),
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_split = train.loc[train_idx].reset_index(drop=True)
    val_split = train.loc[val_idx].reset_index(drop=True)

    y_train = train_split[TARGET].astype(float).reset_index(drop=True)
    y_val = val_split[TARGET].astype(float).reset_index(drop=True)

    col_types = classify_columns(train_split, TARGET)

    builder = FeatureBuilder()
    X_train = builder.fit(train_split, y_train, col_types).reset_index(drop=True)
    X_val = builder.transform(val_split).reset_index(drop=True)
    X_test = builder.transform(test).reset_index(drop=True)

    X_train = X_train.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    X_val = X_val.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    X_test = X_test.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)

    if len(y_train) != len(X_train):
        raise ValueError(
            "Training target length does not match training feature matrix."
        )
    if len(y_val) != len(X_val):
        raise ValueError(
            "Validation target length does not match validation feature matrix."
        )
    if len(X_test) != len(test):
        raise ValueError("Test feature matrix length does not match test frame length.")

    schema_ok = list(X_train.columns) == list(X_val.columns) == list(X_test.columns)
    finite_ok = bool(
        np.isfinite(X_train.to_numpy(dtype=float)).all()
        and np.isfinite(X_val.to_numpy(dtype=float)).all()
        and np.isfinite(X_test.to_numpy(dtype=float)).all()
    )
    target_ok = bool(y_train.notna().all() and y_val.notna().all())

    if not (schema_ok and target_ok):
        raise ValueError("Feature schema or target validation failed.")

    X_train.to_parquet(os.path.join(WORK_DIR, "train_X.parquet"))
    X_val.to_parquet(os.path.join(WORK_DIR, "val_X.parquet"))
    X_test.to_parquet(os.path.join(WORK_DIR, "test_X.parquet"))

    train_split[["id", TARGET]].to_parquet(
        os.path.join(WORK_DIR, "train_ids_target.parquet")
    )
    val_split[["id", TARGET]].to_parquet(
        os.path.join(WORK_DIR, "val_ids_target.parquet")
    )

    test_ids = pd.Series(np.arange(len(X_test), dtype=int), name="id")
    test_ids.to_frame().to_parquet(os.path.join(WORK_DIR, "test_ids.parquet"))

    feature_metadata = {
        "target": TARGET,
        "metric": "rmse",
        "split": {
            "method": "random",
            "val_fraction": VAL_FRACTION,
            "random_state": RANDOM_STATE,
        },
        "n_train": int(len(train_split)),
        "n_val": int(len(val_split)),
        "n_test": int(len(test)),
        "feature_names": builder.feature_names,
        "column_types": col_types,
        "postprocessing": "none",
        "data_readiness": {
            "schema_ok": bool(schema_ok),
            "finite_ok": bool(finite_ok),
            "target_ok": bool(target_ok),
        },
        "notes": "All scalers/encoders/statistics are fitted only on the train split. Validation and test use transform only.",
    }
    with open(
        os.path.join(WORK_DIR, "feature_metadata.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(feature_metadata, f, indent=2, default=str)

    if X_train.shape[1] == 0:
        X_train["constant_feature"] = 1.0
        X_val["constant_feature"] = 1.0
        X_test["constant_feature"] = 1.0

    criterion = build_criterion()
    optimizer_config = build_optimizer()

    model_a = build_lgbm_model_a()
    model_b = build_lgbm_model_b()
    xgb_model = build_xgb_model()

    model_design = {
        "model_type": "weighted_ensemble",
        "objective": "regression",
        "metric": "rmse",
        "optimizer": optimizer_config,
        "weights": {
            "lgbm_a": 0.40,
            "lgbm_b": 0.35,
            "xgb": 0.25,
        },
        "models": [
            {
                "name": "lgbm_a",
                "model": "lightgbm.LGBMRegressor",
                "weight": 0.40,
            },
            {
                "name": "lgbm_b",
                "model": "lightgbm.LGBMRegressor",
                "weight": 0.35,
            },
            {
                "name": "xgb",
                "model": "xgboost.XGBRegressor",
                "weight": 0.25,
            },
        ],
        "validation_split": {
            "source": "in-memory train/val artifacts",
            "target": TARGET,
            "metric": "root_mean_squared_error",
        },
        "postprocessing": "nan_to_num + clip to train target range",
    }
    with open(os.path.join(WORK_DIR, "model_design.json"), "w", encoding="utf-8") as f:
        json.dump(model_design, f, indent=2, default=str)

    early_stopping_rounds = int(optimizer_config["early_stopping_rounds"])

    tracker_a = RMSETracker()
    model_a.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            tracker_a,
        ],
    )

    best_iteration_a = getattr(model_a, "best_iteration_", None)
    if not isinstance(best_iteration_a, int) or best_iteration_a <= 0:
        best_iteration_a = (
            tracker_a.best_iteration
            if tracker_a.best_iteration > 0
            else int(model_a.n_estimators)
        )

    model_a_path = os.path.join(WORK_DIR, "best_lgbm_model_a.txt")
    try:
        model_a.booster_.set_num_iterations(best_iteration_a)
    except Exception:
        pass
    model_a.booster_.save_model(model_a_path)

    best_booster_a = lgb.Booster(model_file=model_a_path)

    val_pred_a = best_booster_a.predict(X_val, num_iteration=best_iteration_a)
    test_pred_a = best_booster_a.predict(X_test, num_iteration=best_iteration_a)

    val_pred_a = sanitize_predictions(val_pred_a, y_train)
    test_pred_a = sanitize_predictions(test_pred_a, y_train)

    tracker_b = RMSETracker()
    model_b.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            tracker_b,
        ],
    )

    best_iteration_b = getattr(model_b, "best_iteration_", None)
    if not isinstance(best_iteration_b, int) or best_iteration_b <= 0:
        best_iteration_b = (
            tracker_b.best_iteration
            if tracker_b.best_iteration > 0
            else int(model_b.n_estimators)
        )

    model_b_path = os.path.join(WORK_DIR, "best_lgbm_model_b.txt")
    try:
        model_b.booster_.set_num_iterations(best_iteration_b)
    except Exception:
        pass
    model_b.booster_.save_model(model_b_path)

    best_booster_b = lgb.Booster(model_file=model_b_path)

    val_pred_b = best_booster_b.predict(X_val, num_iteration=best_iteration_b)
    test_pred_b = best_booster_b.predict(X_test, num_iteration=best_iteration_b)

    val_pred_b = sanitize_predictions(val_pred_b, y_train)
    test_pred_b = sanitize_predictions(test_pred_b, y_train)

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    xgb_best_iteration = getattr(xgb_model, "best_iteration", None)
    if not isinstance(xgb_best_iteration, int) or xgb_best_iteration < 0:
        xgb_best_iteration = 0

    val_pred_xgb = xgb_model.predict(X_val)
    test_pred_xgb = xgb_model.predict(X_test)

    val_pred_xgb = sanitize_predictions(val_pred_xgb, y_train)
    test_pred_xgb = sanitize_predictions(test_pred_xgb, y_train)

    P_val = np.column_stack([val_pred_a, val_pred_b, val_pred_xgb])
    if nnls is not None and P_val.shape[0] > 0:
        try:
            w, _ = nnls(P_val, y_val.to_numpy(dtype=float))
            w = np.asarray(w, dtype=float)
            if w.shape[0] != 3 or not np.all(np.isfinite(w)):
                w = np.array([0.40, 0.35, 0.25])
            else:
                w_sum = float(w.sum())
                if w_sum <= 0 or not np.isfinite(w_sum):
                    w = np.array([0.40, 0.35, 0.25])
                else:
                    w = w / w_sum
        except Exception:
            w = np.array([0.40, 0.35, 0.25])
    else:
        w = np.array([0.40, 0.35, 0.25])

    val_pred = w[0] * val_pred_a + w[1] * val_pred_b + w[2] * val_pred_xgb
    test_pred = w[0] * test_pred_a + w[1] * test_pred_b + w[2] * test_pred_xgb
    print(f"Learned ensemble weights: lgbm_a={w[0]:.6f}, lgbm_b={w[1]:.6f}, xgb={w[2]:.6f}")

    val_pred = sanitize_predictions(val_pred, y_train)
    test_pred = sanitize_predictions(test_pred, y_train)

    score = float(criterion(y_val, val_pred))

    submission = pd.DataFrame(
        {
            "id": test_ids.astype("int64"),
            "Recovery": test_pred.astype(float),
        }
    )
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    training_metadata = {
        "step": "training_evaluation",
        "model_type": "weighted_ensemble",
        "objective": "regression",
        "metric": "rmse",
        "optimizer": optimizer_config,
        "weights": {
            "lgbm_a": 0.40,
            "lgbm_b": 0.35,
            "xgb": 0.25,
        },
        "best_iterations": {
            "lgbm_a": int(best_iteration_a),
            "lgbm_b": int(best_iteration_b),
            "xgb": int(xgb_best_iteration),
        },
        "early_stopping_rounds": early_stopping_rounds,
        "tracker_best_rmse": {
            "lgbm_a": (
                float(tracker_a.best_rmse)
                if np.isfinite(tracker_a.best_rmse)
                else None
            ),
            "lgbm_b": (
                float(tracker_b.best_rmse)
                if np.isfinite(tracker_b.best_rmse)
                else None
            ),
        },
        "validation_rmse_after_postprocessing": score,
        "postprocessing": "nan_to_num + clip to train target range",
        "model_paths": [model_a_path, model_b_path],
        "submission_path": os.path.join(SUBMISSION_DIR, "submission.csv"),
    }
    with open(
        os.path.join(WORK_DIR, "training_evaluation.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(training_metadata, f, indent=2, default=str)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
