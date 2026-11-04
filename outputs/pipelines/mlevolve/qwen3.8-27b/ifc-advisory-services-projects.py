import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Estimated Total Budget ($)"
SEED = 42
WORK = "./working"
SUB_DIR = "./submission"

os.makedirs(WORK, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)


def clean_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).str.strip()
    t = t.str.replace(r"[$,%\s]", "", regex=True)
    t = t.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "null": np.nan,
            "N/A": np.nan,
        }
    )
    return pd.to_numeric(t, errors="coerce")


def looks_like_numeric(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    x = s.dropna().astype(str).str.strip()
    if x.empty:
        return False
    pattern = r"^[-+]?[\d,]*\.?\d+([eE][-+]?\d+)?$|^[-+]?[\d,]*\.?\d+[%]?$"
    return x.str.replace(r"[$,\s%]", "", regex=True).str.match(pattern).mean() > 0.85


def looks_like_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    x = s.dropna().astype(str).str.strip()
    if x.empty:
        return False
    try:
        parsed = pd.to_datetime(x, errors="coerce", format="mixed", dayfirst=False)
    except TypeError:
        parsed = pd.to_datetime(x, errors="coerce", dayfirst=False)
    return parsed.notna().mean() > 0.85


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
    except TypeError:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)


class LinearBaseline:
    """Regularized linear baseline with train-only imputation/scaling."""

    def __init__(self, alpha: float = 25.0):
        self.alpha = alpha
        self.imputer = None
        self.scaler = None
        self.model = None

    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        if fit:
            self.imputer = SimpleImputer(strategy="median").fit(X)
            self.scaler = StandardScaler().fit(self.imputer.transform(X))
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("LinearBaseline must be fitted before transform.")
        return self.scaler.transform(self.imputer.transform(X))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearBaseline":
        Xp = self._preprocess(X, fit=True)
        y = np.asarray(y, dtype=float)
        self.model = Ridge(alpha=self.alpha).fit(Xp, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._preprocess(X, fit=False))

    def predict_oof(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            return np.array([], dtype=float)

        n_splits = int(max(2, min(n_splits, n))) if n >= 2 else 1
        if n_splits < 2:
            return self.predict(X)

        oof = np.full(n, np.nan, dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for tr_idx, va_idx in kf.split(X):
            fold_model = LinearBaseline(alpha=self.alpha).fit(X.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = fold_model.predict(X.iloc[va_idx])
        return np.nan_to_num(oof, nan=float(np.mean(y)))


class FeatureBuilder:
    def __init__(self, target: str):
        self.target = target
        self.feature_names = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.text_cols = []
        self.freq_maps = {}
        self.target_stats = {}
        self.quantile_edges = {}
        self.numeric_median = {}
        self.combo_cols = []
        self.combo_freq_maps = {}
        self.combo_target_stats = {}
        self.interaction_top_cols = []
        self.n_train = 0

    def _reset_state(self):
        self.feature_names = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.text_cols = []
        self.freq_maps = {}
        self.target_stats = {}
        self.quantile_edges = {}
        self.numeric_median = {}
        self.combo_cols = []
        self.combo_freq_maps = {}
        self.combo_target_stats = {}
        self.interaction_top_cols = []

    @staticmethod
    def _category_series(df: pd.DataFrame, col: str) -> pd.Series:
        if col.endswith("__as_cat"):
            base = col.replace("__as_cat", "")
            s = (
                clean_numeric_series(df[base])
                .astype("Int64")
                .astype(str)
                .replace("<NA>", "__MISSING__")
            )
        else:
            s = df[col].astype("object")
            s = s.where(s.notna(), "__MISSING__")
            s = s.astype(str)
            s = s.replace(
                {
                    "nan": "__MISSING__",
                    "None": "__MISSING__",
                    "": "__MISSING__",
                }
            )
        return s.fillna("__MISSING__")

    def _classify(self, df: pd.DataFrame):
        self._reset_state()

        for col in df.columns:
            if col == self.target:
                continue
            s = df[col]
            if looks_like_datetime(s):
                self.datetime_cols.append(col)
            elif looks_like_numeric(s):
                self.numeric_cols.append(col)
            else:
                nuniq = s.nunique(dropna=True)
                if nuniq <= 250:
                    self.categorical_cols.append(col)
                else:
                    self.text_cols.append(col)

        for col in list(self.numeric_cols):
            s = clean_numeric_series(df[col])
            if s.nunique(dropna=True) <= 20:
                self.categorical_cols.append(f"{col}__as_cat")

        cat_candidates = []
        for col in self.categorical_cols:
            s = self._category_series(df, col)
            nuniq = s.nunique(dropna=True)
            if 1 <= nuniq <= 25:
                cat_candidates.append(col)
        self.combo_cols = cat_candidates[:5]

    def _fit_categorical_maps(self, df: pd.DataFrame, y: pd.Series):
        y = y.astype(float).reset_index(drop=True)
        global_mean = float(y.mean()) if len(y) else 0.0
        m = 20.0

        for col in self.categorical_cols:
            s = self._category_series(df, col).reset_index(drop=True)
            counts = s.value_counts(normalize=True)
            self.freq_maps[col] = counts.to_dict()

            tmp = pd.DataFrame({"cat": s.values, "y": y.values})
            agg = tmp.groupby("cat")["y"].agg(["sum", "size", "mean"])
            smoothed = (agg["sum"] + m * global_mean) / (agg["size"] + m)
            self.target_stats[col] = smoothed.to_dict()

        if len(self.combo_cols) >= 2:
            combos = []
            for i in range(len(self.combo_cols)):
                for j in range(i + 1, min(i + 3, len(self.combo_cols))):
                    combos.append((self.combo_cols[i], self.combo_cols[j]))
            self.combo_cols = combos

            for c1, c2 in self.combo_cols:
                name = f"combo__{c1}__{c2}"
                s1 = self._category_series(df, c1).reset_index(drop=True)
                s2 = self._category_series(df, c2).reset_index(drop=True)
                combo = s1 + "|" + s2

                counts = combo.value_counts(normalize=True)
                self.combo_freq_maps[name] = counts.to_dict()

                tmp = pd.DataFrame({"cat": combo.values, "y": y.values})
                agg = tmp.groupby("cat")["y"].agg(["sum", "size"])
                smoothed = (agg["sum"] + m * global_mean) / (agg["size"] + m)
                self.combo_target_stats[name] = smoothed.to_dict()

    def _oof_target_encoding(
        self, df: pd.DataFrame, y: pd.Series, col: str, k: int = 5, m: float = 20.0
    ) -> pd.Series:
        s = self._category_series(df, col).reset_index(drop=True)
        y = y.reset_index(drop=True).astype(float)
        out = pd.Series(np.nan, index=s.index)

        if len(s) <= 2:
            return pd.Series(float(y.mean()) if len(y) else 0.0, index=s.index)

        k = int(max(2, min(k, len(s))))
        folds = np.array([i % k for i in range(len(s))])
        for fold in range(k):
            train_mask = folds != fold
            val_mask = folds == fold
            if not train_mask.any() or not val_mask.any():
                continue
            tmp = pd.DataFrame({"cat": s[train_mask].values, "y": y[train_mask].values})
            agg = tmp.groupby("cat")["y"].agg(["sum", "size"])
            global_mean = float(y[train_mask].mean()) if train_mask.any() else 0.0
            smoothed = (agg["sum"] + m * global_mean) / (agg["size"] + m)
            out[val_mask] = s[val_mask].map(smoothed).fillna(global_mean).values
        return out.fillna(float(y.mean()) if len(y) else 0.0)

    def _oof_target_encoding_series(
        self, s: pd.Series, y: pd.Series, k: int = 5, m: float = 20.0
    ) -> pd.Series:
        s = s.reset_index(drop=True)
        y = y.reset_index(drop=True).astype(float)
        out = pd.Series(np.nan, index=s.index)

        if len(s) <= 2:
            return pd.Series(float(y.mean()) if len(y) else 0.0, index=s.index)

        k = int(max(2, min(k, len(s))))
        folds = np.array([i % k for i in range(len(s))])
        for fold in range(k):
            train_mask = folds != fold
            val_mask = folds == fold
            if not train_mask.any() or not val_mask.any():
                continue
            tmp = pd.DataFrame({"cat": s[train_mask].values, "y": y[train_mask].values})
            agg = tmp.groupby("cat")["y"].agg(["sum", "size"])
            global_mean = float(y[train_mask].mean()) if train_mask.any() else 0.0
            smoothed = (agg["sum"] + m * global_mean) / (agg["size"] + m)
            out[val_mask] = s[val_mask].map(smoothed).fillna(global_mean).values
        return out.fillna(float(y.mean()) if len(y) else 0.0)

    def fit(self, df: pd.DataFrame, y: pd.Series):
        self.n_train = len(df)
        self._classify(df)
        self._fit_categorical_maps(df, y)

        variances = {}
        for col in self.numeric_cols:
            s = clean_numeric_series(df[col])
            self.numeric_median[col] = float(s.median()) if s.notna().any() else 0.0
            if s.notna().any():
                self.quantile_edges[col] = np.unique(
                    np.quantile(s.dropna().to_numpy(), np.linspace(0, 1, 11))
                )
            else:
                self.quantile_edges[col] = np.array([])
            variances[col] = float(s.var()) if s.notna().any() else 0.0

        self.interaction_top_cols = sorted(
            variances, key=lambda c: variances[c], reverse=True
        )[:5]

        return self

    def _transform_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        for col in self.numeric_cols:
            s = clean_numeric_series(df[col])
            out[f"{col}"] = s

            abs_s = s.abs().fillna(0.0)
            out[f"{col}__log"] = np.sign(s.fillna(0.0)) * np.log1p(abs_s)

            edges = self.quantile_edges.get(col)
            if edges is not None and len(edges) > 1:
                try:
                    out[f"{col}__qbin"] = pd.cut(
                        s, bins=edges, labels=False, include_lowest=True
                    ).astype(float)
                except Exception:
                    out[f"{col}__qbin"] = -1.0
            else:
                out[f"{col}__qbin"] = -1.0

            out[f"{col}__missing"] = s.isna().astype(int)

        if len(self.interaction_top_cols) > 1:
            top = self.interaction_top_cols
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    a = clean_numeric_series(df[top[i]]).fillna(0.0)
                    b = clean_numeric_series(df[top[j]]).fillna(0.0)
                    out[f"{top[i]}__x__{top[j]}"] = a * b
                    denom = np.where(np.abs(b) > 1e-8, b, np.nan)
                    out[f"{top[i]}__div__{top[j]}"] = (a / denom).fillna(0.0)

        return out

    def _transform_categorical(
        self, df: pd.DataFrame, y: pd.Series | None, is_train: bool
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        for col in self.categorical_cols:
            s = self._category_series(df, col)
            base = col.replace("__as_cat", "") if col.endswith("__as_cat") else col

            freq_map = self.freq_maps.get(col, {})
            out[f"{col}__freq"] = s.map(freq_map).fillna(0.0)
            out[f"{col}__missing"] = (
                clean_numeric_series(df[base]).isna().astype(int)
                if col.endswith("__as_cat")
                else df[base].isna().astype(int)
            )

            if is_train and y is not None:
                out[f"{col}__tenc"] = self._oof_target_encoding(df, y, col)
            else:
                stats = self.target_stats.get(col, {})
                fallback = float(np.mean(list(stats.values()))) if stats else 0.0
                out[f"{col}__tenc"] = s.map(stats).fillna(fallback)

        for name, (c1, c2) in zip(self.combo_freq_maps.keys(), self.combo_cols):
            s1 = self._category_series(df, c1)
            s2 = self._category_series(df, c2)
            combo = s1 + "|" + s2

            out[f"{name}__freq"] = combo.map(self.combo_freq_maps.get(name, {})).fillna(
                0.0
            )
            if is_train and y is not None:
                out[f"{name}__tenc"] = self._oof_target_encoding_series(combo, y)
            else:
                stats = self.combo_target_stats.get(name, {})
                fallback = float(np.mean(list(stats.values()))) if stats else 0.0
                out[f"{name}__tenc"] = combo.map(stats).fillna(fallback)

        return out

    def _transform_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.datetime_cols:
            s = _safe_to_datetime(df[col])
            out[f"{col}__year"] = s.dt.year
            out[f"{col}__month"] = s.dt.month
            out[f"{col}__day"] = s.dt.day
            out[f"{col}__dow"] = s.dt.dayofweek
            out[f"{col}__is_weekend"] = (s.dt.dayofweek >= 5).astype(int)
            out[f"{col}__quarter"] = s.dt.quarter
            out[f"{col}__dayofyear"] = s.dt.dayofyear
            out[f"{col}__missing"] = s.isna().astype(int)
        return out

    def _transform_text(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.text_cols:
            s = df[col].astype(str)
            out[f"{col}__len"] = s.str.len()
            out[f"{col}__words"] = s.str.split().str.len()
            out[f"{col}__digits"] = s.str.count(r"\d")
            out[f"{col}__upper_ratio"] = s.str.count(r"[A-Z]") / s.str.len().replace(
                0, np.nan
            )
            out[f"{col}__missing"] = df[col].isna().astype(int)
        return out

    def transform(
        self, df: pd.DataFrame, y: pd.Series | None = None, is_train: bool = False
    ) -> pd.DataFrame:
        parts = [
            self._transform_numeric(df),
            self._transform_categorical(df, y, is_train),
            self._transform_datetime(df),
            self._transform_text(df),
        ]
        X = pd.concat(parts, axis=1)
        X = X.replace([np.inf, -np.inf], np.nan)
        X.columns = [str(c) for c in X.columns]
        return X


def clean_features(X_train: pd.DataFrame) -> list:
    keep = []
    for c in X_train.columns:
        s = X_train[c]
        if s.isna().all():
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return keep


def postprocess_predictions(pred: np.ndarray, fallback: float) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=fallback, posinf=fallback, neginf=fallback)
    return np.maximum(pred, 0.0)


def train_models_and_select_blend(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
):
    train_mean = float(np.mean(y_train)) if len(y_train) else 0.0

    linear = LinearBaseline(alpha=25.0).fit(X_train, y_train)
    linear_oof_train = linear.predict_oof(X_train, y_train, n_splits=5)
    linear_val = linear.predict(X_val)
    linear_test = linear.predict(X_test)

    residual_train = y_train.values - linear_oof_train
    residual_val = y_val.values - linear_val

    direct_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 5,
        "subsample": 0.9,
        "bagging_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }

    residual_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 8,
        "subsample": 0.9,
        "bagging_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }

    dtrain_direct = lgb.Dataset(X_train, label=y_train.values, free_raw_data=False)
    dval_direct = lgb.Dataset(
        X_val,
        label=y_val.values,
        reference=dtrain_direct,
        free_raw_data=False,
    )
    direct_model = lgb.train(
        direct_params,
        dtrain_direct,
        num_boost_round=3000,
        valid_sets=[dval_direct],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    dtrain_residual = lgb.Dataset(X_train, label=residual_train, free_raw_data=False)
    dval_residual = lgb.Dataset(
        X_val,
        label=residual_val,
        reference=dtrain_residual,
        free_raw_data=False,
    )
    residual_model = lgb.train(
        residual_params,
        dtrain_residual,
        num_boost_round=3000,
        valid_sets=[dval_residual],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    direct_val_raw = direct_model.predict(
        X_val, num_iteration=direct_model.best_iteration
    )
    residual_val_raw = residual_model.predict(
        X_val, num_iteration=residual_model.best_iteration
    )
    hybrid_val_raw = linear_val + residual_val_raw

    best_weight = 0.5
    best_score = np.inf
    for w in np.linspace(0.0, 1.0, 51):
        val_pred = postprocess_predictions(
            w * direct_val_raw + (1.0 - w) * hybrid_val_raw, train_mean
        )
        score = rmse(y_val.values, val_pred)
        if score < best_score:
            best_score = score
            best_weight = float(w)

    direct_test_raw = direct_model.predict(
        X_test, num_iteration=direct_model.best_iteration
    )
    residual_test_raw = residual_model.predict(
        X_test, num_iteration=residual_model.best_iteration
    )
    hybrid_test_raw = linear_test + residual_test_raw

    val_pred = postprocess_predictions(
        best_weight * direct_val_raw + (1.0 - best_weight) * hybrid_val_raw,
        train_mean,
    )
    test_pred = postprocess_predictions(
        best_weight * direct_test_raw + (1.0 - best_weight) * hybrid_test_raw,
        train_mean,
    )

    final_score = rmse(y_val.values, val_pred)

    meta = {
        "architecture": "convex blend of direct LightGBM and linear+LightGBM residual hybrid",
        "loss": "L2 regression",
        "optimizers": ["ridge closed-form", "lightgbm gradient boosting"],
        "blend_weight_direct": best_weight,
        "clip_lower": 0.0,
        "train_mean": train_mean,
        "direct_best_iteration": int(direct_model.best_iteration),
        "residual_best_iteration": int(residual_model.best_iteration),
        "validation_rmse": final_score,
    }

    with open(os.path.join(WORK, "training_evaluation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return test_pred, final_score, meta


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET not in train.columns:
        raise ValueError(f"Target column {TARGET!r} not found in train.parquet")

    train[TARGET] = clean_numeric_series(train[TARGET])
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test = test.reset_index(drop=True)
    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    if len(train) >= 2:
        n_val = max(1, min(len(train) - 1, int(round(0.2 * len(train)))))
        stratify = None
        try:
            if len(train) >= 20:
                stratify = pd.qcut(train[TARGET], q=10, duplicates="drop")
        except Exception:
            stratify = None

        try:
            train_full, val = train_test_split(
                train,
                test_size=n_val,
                random_state=SEED,
                stratify=stratify,
                shuffle=True,
            )
        except Exception:
            train_full, val = train_test_split(
                train,
                test_size=n_val,
                random_state=SEED,
                shuffle=True,
            )
    else:
        train_full = train
        val = train

    train_full = train_full.reset_index(drop=True)
    val = val.reset_index(drop=True)

    y_train = train_full[TARGET].astype(float)
    y_val = val[TARGET].astype(float)

    builder = FeatureBuilder(TARGET).fit(train_full, y_train)

    X_train = builder.transform(train_full, y=y_train, is_train=True)
    X_val = builder.transform(val, y=None, is_train=False)
    X_test = builder.transform(test, y=None, is_train=False)

    keep = clean_features(X_train)
    if not keep:
        keep = list(X_train.columns)

    X_train = X_train.loc[:, keep].reset_index(drop=True)
    X_val = X_val.loc[:, keep].reset_index(drop=True)
    X_test = X_test.loc[:, keep].reset_index(drop=True)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    X_train.to_parquet(os.path.join(WORK, "X_train.parquet"))
    X_val.to_parquet(os.path.join(WORK, "X_val.parquet"))
    X_test.to_parquet(os.path.join(WORK, "X_test.parquet"))
    pd.DataFrame({TARGET: y_train.values}).to_csv(
        os.path.join(WORK, "y_train.csv"), index=False
    )
    pd.DataFrame({TARGET: y_val.values}).to_csv(
        os.path.join(WORK, "y_val.csv"), index=False
    )

    with open(os.path.join(WORK, "feature_meta.json"), "w") as f:
        json.dump(
            {
                "target": TARGET,
                "seed": SEED,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "n_test": int(len(X_test)),
                "n_features": int(X_train.shape[1]),
                "numeric_cols": builder.numeric_cols,
                "categorical_cols": builder.categorical_cols,
                "datetime_cols": builder.datetime_cols,
                "text_cols": builder.text_cols,
                "feature_names": list(X_train.columns),
            },
            f,
            indent=2,
        )

    test_pred, score, meta = train_models_and_select_blend(
        X_train, y_train, X_val, y_val, X_test
    )

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            TARGET: test_pred,
        }
    )
    submission.to_csv(os.path.join(SUB_DIR, "submission.csv"), index=False)

    print(
        f"Blend direct={meta['blend_weight_direct']:.3f}, "
        f"hybrid={1.0 - meta['blend_weight_direct']:.3f}, "
        f"direct_iter={meta['direct_best_iteration']}, "
        f"residual_iter={meta['residual_best_iteration']}"
    )
    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()