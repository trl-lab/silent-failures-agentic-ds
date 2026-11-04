import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

TARGET = "Task Assigned"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
SEED = 42

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)
np.random.seed(SEED)


def _clean_str(s: str) -> str:
    return str(s).strip().lower()


def _is_id_column(col: str) -> bool:
    c = _clean_str(col)
    if c in {"id", "index", "row_id", "sample_id", "uuid", "key", "idx", "row_index"}:
        return True
    if c.endswith("_id"):
        return True
    return False


def _looks_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(
        series
    ):
        return False
    sample = series.dropna().astype(str).head(120)
    if len(sample) == 0:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.8


def _looks_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(
        series
    ):
        return False
    sample = series.dropna().astype(str).head(120)
    if len(sample) == 0:
        return False
    parsed = pd.to_numeric(sample, errors="coerce")
    return parsed.notna().mean() >= 0.8


def _looks_text(series: pd.Series) -> bool:
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(
        series
    ):
        return False
    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    avg_len = float(sample.str.len().mean())
    avg_words = float(sample.str.split().str.len().mean())
    return avg_len > 25 or avg_words > 4


def _encode_binary_target(y: pd.Series):
    y = y.dropna()
    if len(y) == 0:
        return pd.Series(dtype=int), {"mode": "empty", "positive": None, "classes": []}

    if pd.api.types.is_numeric_dtype(y):
        uniq = sorted(y.unique().tolist())
        if len(uniq) == 1:
            return pd.Series(0, index=y.index, dtype=int), {
                "mode": "single",
                "positive": uniq[0],
                "classes": uniq,
            }
        if len(uniq) == 2:
            positive = 1 if 1 in uniq else max(uniq)
            negative = [x for x in uniq if x != positive][0]
            y_bin = (y == positive).astype(int)
            return y_bin, {
                "mode": "binary",
                "positive": positive,
                "negative": negative,
                "classes": uniq,
            }

        # Fallback for numeric multiclass: choose the largest numeric label as the positive class.
        positive = max(uniq)
        y_bin = (y == positive).astype(int)
        return y_bin, {
            "mode": "multiclass_collapsed",
            "positive": positive,
            "classes": uniq,
        }

    y_str = y.astype(str).str.strip().str.lower()
    uniq = list(y_str.unique())
    if len(uniq) == 1:
        return pd.Series(0, index=y.index, dtype=int), {
            "mode": "single",
            "positive": uniq[0],
            "classes": uniq,
        }
    if len(uniq) == 2:
        pos_terms = {
            "yes",
            "y",
            "true",
            "t",
            "1",
            "assigned",
            "accident",
            "accidental",
            "unsafe",
            "injury",
            "incident",
            "fatal",
            "lost time",
            "lti",
            "recordable",
        }
        neg_terms = {
            "no",
            "n",
            "false",
            "f",
            "0",
            "not assigned",
            "unassigned",
            "safe",
            "no accident",
            "none",
            "na",
            "n/a",
        }
        positive = None
        for c in uniq:
            if c in pos_terms:
                positive = c
                break
        if positive is None:
            for c in uniq:
                if c in neg_terms:
                    positive = [x for x in uniq if x != c][0]
                    break
        if positive is None:
            not_like = [
                c
                for c in uniq
                if c.startswith("not")
                or c.startswith("no ")
                or c in {"no", "n", "false", "f", "0"}
            ]
            if len(not_like) == 1:
                positive = [x for x in uniq if x != not_like[0]][0]
        if positive is None:
            counts = y_str.value_counts()
            positive = counts.idxmin()
        negative = [x for x in uniq if x != positive][0]
        y_bin = (y_str == positive).astype(int)
        return y_bin, {
            "mode": "binary",
            "positive": positive,
            "negative": negative,
            "classes": uniq,
        }

    # Fallback for string multiclass: choose the least frequent class as positive.
    counts = y_str.value_counts()
    positive = counts.idxmin()
    y_bin = (y_str == positive).astype(int)
    return y_bin, {
        "mode": "multiclass_collapsed",
        "positive": positive,
        "classes": uniq,
    }


class TabularFeatureProcessor:
    def __init__(self):
        self.numeric_cols = []
        self.datetime_cols = []
        self.categorical_cols = []
        self.text_cols = []
        self.dropped_cols = []
        self.medians = {}
        self.mins = {}
        self.skews = {}
        self.cat_mappings = {}
        self.cat_freqs = {}
        self.cat_target_enc = {}
        self.global_target_mean = 0.5
        self.scaler = StandardScaler()
        self.scale_cols = []
        self.feature_order = []

    def _classify(self, X: pd.DataFrame):
        self.numeric_cols = []
        self.datetime_cols = []
        self.categorical_cols = []
        self.text_cols = []
        for c in X.columns:
            if _is_id_column(c):
                self.dropped_cols.append(c)
                continue
            s = X[c]
            if _looks_datetime(s):
                self.datetime_cols.append(c)
            elif _looks_numeric(s):
                self.numeric_cols.append(c)
            else:
                self.categorical_cols.append(c)
                if _looks_text(s):
                    self.text_cols.append(c)

    def _unique_ratio(self, x) -> float:
        x = str(x)
        return len(set(x)) / max(len(x), 1)

    def _datetime_parts(self, s: pd.Series) -> dict:
        parts = {
            "year": s.dt.year,
            "month": s.dt.month,
            "day": s.dt.day,
            "dayofweek": s.dt.dayofweek,
            "hour": s.dt.hour,
            "minute": s.dt.minute,
            "quarter": s.dt.quarter,
            "is_weekend": (s.dt.dayofweek >= 5).astype(float),
        }
        for name, vals in parts.items():
            vals = vals.astype(float)
            vals[s.isna()] = np.nan
        return parts

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        X = X.copy()
        self._classify(X)
        out = pd.DataFrame(index=X.index)

        if y is not None:
            y_arr = np.asarray(y).reshape(-1)
            if len(y_arr) == len(X):
                self.global_target_mean = float(np.mean(y_arr)) if len(y_arr) else 0.5
            else:
                self.global_target_mean = 0.5
        else:
            self.global_target_mean = 0.5

        for c in self.numeric_cols:
            s = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = s.median()
            if pd.isna(med):
                med = 0.0
            self.medians[c] = float(med)
            s_imp = s.fillna(med)
            self.mins[c] = float(s_imp.min()) if len(s_imp) else 0.0
            self.skews[c] = float(s_imp.skew()) if len(s_imp) else 0.0
            out[c] = s_imp.astype(float)
            out[c + "_missing"] = s.isna().astype(float)
            if abs(self.skews[c]) > 1.0:
                if self.mins[c] >= 0:
                    out[c + "_log"] = np.log1p(s_imp)
                else:
                    out[c + "_log"] = np.log1p(s_imp - self.mins[c] + 1.0)

        for c in self.datetime_cols:
            s = pd.to_datetime(X[c], errors="coerce")
            parts = self._datetime_parts(s)
            for name, vals in parts.items():
                med = vals.median()
                if pd.isna(med):
                    med = 0.0
                self.medians[(c, name)] = float(med)
                out[f"{c}_{name}"] = vals.fillna(med).astype(float)
            out[c + "_missing"] = s.isna().astype(float)

        binary_target = False
        if y is not None:
            y_arr = np.asarray(y).reshape(-1)
            binary_target = len(np.unique(y_arr)) == 2

        for c in self.categorical_cols:
            raw = X[c]
            s = raw.astype("string").str.strip().fillna("__MISSING__")
            mapping = {cat: i for i, cat in enumerate(s.unique())}
            self.cat_mappings[c] = mapping
            freq = s.value_counts(normalize=True)
            self.cat_freqs[c] = freq
            out[c + "_code"] = s.map(mapping).fillna(-1).astype(int)
            out[c + "_freq"] = s.map(freq).fillna(0.0).astype(float)
            out[c + "_missing"] = raw.isna().astype(float)

            if binary_target and y is not None:
                y_arr = np.asarray(y).reshape(-1)
                if len(y_arr) == len(s):
                    tmp = pd.DataFrame({"cat": s, "y": y_arr})
                    agg = tmp.groupby("cat")["y"].agg(["count", "mean"])
                    smoothing = 10.0
                    te = (
                        agg["count"] * agg["mean"] + smoothing * self.global_target_mean
                    ) / (agg["count"] + smoothing)
                    self.cat_target_enc[c] = te.to_dict()
                    out[c + "_target_enc"] = (
                        s.map(te).fillna(self.global_target_mean).astype(float)
                    )

        for c in self.text_cols:
            raw = X[c]
            s = raw.fillna("").astype(str)
            out[c + "_len"] = s.str.len().astype(float)
            out[c + "_words"] = s.str.split().str.len().astype(float)
            out[c + "_digits"] = s.str.count(r"\d").astype(float)
            out[c + "_upper"] = (s.str.count(r"[A-Z]") / (s.str.len() + 1)).astype(
                float
            )
            out[c + "_unique_char"] = s.apply(self._unique_ratio).astype(float)
            out[c + "_missing"] = raw.isna().astype(float)

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.scale_cols = [
            c for c in out.columns if not c.endswith(("_code", "_missing"))
        ]
        if self.scale_cols:
            self.scaler = StandardScaler()
            out[self.scale_cols] = self.scaler.fit_transform(out[self.scale_cols])
        if out.shape[1] == 0:
            out["constant_feature"] = 0.0
        self.feature_order = list(out.columns)
        return out.astype(np.float32)

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        out = pd.DataFrame(index=X.index)

        for c in self.numeric_cols:
            s = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = self.medians.get(c, 0.0)
            s_imp = s.fillna(med)
            out[c] = s_imp.astype(float)
            out[c + "_missing"] = s.isna().astype(float)
            if abs(self.skews.get(c, 0.0)) > 1.0:
                if self.mins.get(c, 0.0) >= 0:
                    out[c + "_log"] = np.log1p(s_imp)
                else:
                    out[c + "_log"] = np.log1p(s_imp - self.mins.get(c, 0.0) + 1.0)

        for c in self.datetime_cols:
            s = pd.to_datetime(X[c], errors="coerce")
            parts = self._datetime_parts(s)
            for name, vals in parts.items():
                med = self.medians.get((c, name), 0.0)
                out[f"{c}_{name}"] = vals.fillna(med).astype(float)
            out[c + "_missing"] = s.isna().astype(float)

        for c in self.categorical_cols:
            raw = X[c]
            s = raw.astype("string").str.strip().fillna("__MISSING__")
            mapping = self.cat_mappings.get(c, {})
            freq = self.cat_freqs.get(c, pd.Series(dtype=float))
            out[c + "_code"] = s.map(mapping).fillna(-1).astype(int)
            out[c + "_freq"] = s.map(freq).fillna(0.0).astype(float)
            out[c + "_missing"] = raw.isna().astype(float)
            if c in self.cat_target_enc:
                out[c + "_target_enc"] = (
                    s.map(self.cat_target_enc[c])
                    .fillna(self.global_target_mean)
                    .astype(float)
                )

        for c in self.text_cols:
            raw = X[c]
            s = raw.fillna("").astype(str)
            out[c + "_len"] = s.str.len().astype(float)
            out[c + "_words"] = s.str.split().str.len().astype(float)
            out[c + "_digits"] = s.str.count(r"\d").astype(float)
            out[c + "_upper"] = (s.str.count(r"[A-Z]") / (s.str.len() + 1)).astype(
                float
            )
            out[c + "_unique_char"] = s.apply(self._unique_ratio).astype(float)
            out[c + "_missing"] = raw.isna().astype(float)

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cols = [c for c in self.scale_cols if c in out.columns]
        if cols:
            out[cols] = self.scaler.transform(out[cols])
        if out.shape[1] == 0:
            out["constant_feature"] = 0.0
        return out.astype(np.float32)


def _safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    if np.all(y_score == y_score[0]):
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _postprocess_scores(scores: np.ndarray, mode: str) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if mode == "lambdarank":
        return _sigmoid(scores)
    return np.clip(scores, 1e-6, 1.0 - 1e-6)


def _make_lgbm_ranker_params(n_train: int):
    n_estimators = 1200 if n_train < 20000 else 1800
    min_child = max(20, int(0.02 * n_train))
    return {
        "objective": "lambdarank",
        "boosting_type": "gbdt",
        "n_estimators": n_estimators,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": min_child,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "force_row_wise": True,
    }


def _make_lgbm_binary_params(n_train: int):
    n_estimators = 1200 if n_train < 20000 else 1800
    min_child = max(20, int(0.02 * n_train))
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": n_estimators,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": min_child,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "force_row_wise": True,
    }


def _train_rank_model(X_train, y_train, X_val, y_val):
    import lightgbm as lgb

    group_train = np.array([len(y_train)], dtype=np.int32)
    group_val = np.array([len(y_val)], dtype=np.int32)

    params = _make_lgbm_ranker_params(len(y_train))

    def auc_eval(y_true, y_pred):
        return ("auc", _safe_auc(y_true, y_pred), True)

    model = lgb.LGBMRanker(**params)

    if y_val.nunique() > 1:
        model.fit(
            X_train,
            y_train.astype(int),
            group=group_train,
            eval_set=[(X_val, y_val.astype(int))],
            eval_group=[group_val],
            eval_metric=auc_eval,
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )
        best_iter = int(getattr(model, "best_iteration_", 0) or 0)
    else:
        model.fit(X_train, y_train.astype(int), group=group_train)
        best_iter = int(getattr(model, "n_estimators", 600) or 600)

    val_score = _safe_auc(
        y_val, _postprocess_scores(model.predict(X_val), "lambdarank")
    )
    return model, val_score, best_iter, "lambdarank"


def _train_binary_model(X_train, y_train, X_val, y_val):
    import lightgbm as lgb

    params = _make_lgbm_binary_params(len(y_train))
    pos_count = int(np.sum(np.asarray(y_train) == 1))
    neg_count = int(np.sum(np.asarray(y_train) == 0))
    if pos_count > 0 and neg_count > 0 and pos_count < neg_count:
        params["scale_pos_weight"] = float(neg_count) / float(pos_count)

    model = lgb.LGBMClassifier(**params)

    if y_val.nunique() > 1:
        model.fit(
            X_train,
            y_train.astype(int),
            eval_set=[(X_val, y_val.astype(int))],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )
        best_iter = int(getattr(model, "best_iteration_", 0) or 0)
    else:
        model.fit(X_train, y_train.astype(int))
        best_iter = int(getattr(model, "n_estimators", 600) or 600)

    val_score = _safe_auc(
        y_val, _postprocess_scores(model.predict_proba(X_val)[:, 1], "binary")
    )
    return model, val_score, best_iter, "binary"


def _train_logistic_fallback(X_train, y_train):
    X_fit = (
        X_train
        if X_train.shape[1] > 0
        else pd.DataFrame({"constant_feature": np.zeros(len(X_train))})
    )
    y_fit = y_train.astype(int)

    model = LogisticRegression(max_iter=2000, random_state=SEED)
    try:
        model.fit(X_fit, y_fit)
    except Exception:
        # If even logistic regression cannot fit because of a single class,
        # use a constant learned baseline derived from the training label mean.
        mean_y = float(np.mean(y_fit)) if len(y_fit) else 0.0
        model = None
        model_mean = mean_y

    return model, model_mean if model is None else None


def _predict_model(model, X, mode, fallback_mean=None):
    if mode == "lambdarank":
        return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)

    if mode == "logistic_fallback":
        if model is None:
            return np.full(
                len(X),
                float(fallback_mean if fallback_mean is not None else 0.5),
                dtype=np.float64,
            )
        proba = np.asarray(model.predict_proba(X), dtype=np.float64)
        if proba.shape[1] == 1:
            return proba[:, 0].reshape(-1)
        return proba[:, 1].reshape(-1)

    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64).reshape(-1)


def _retrain_final_model(X_full, y_full, best_iter, mode):
    import lightgbm as lgb

    if best_iter <= 0:
        best_iter = 600

    if mode == "lambdarank":
        params = _make_lgbm_ranker_params(len(y_full))
        params["n_estimators"] = best_iter
        model = lgb.LGBMRanker(**params)
        model.fit(
            X_full, y_full.astype(int), group=np.array([len(y_full)], dtype=np.int32)
        )
        return model, "lambdarank"

    if mode == "binary":
        params = _make_lgbm_binary_params(len(y_full))
        params["n_estimators"] = best_iter
        pos_count = int(np.sum(np.asarray(y_full) == 1))
        neg_count = int(np.sum(np.asarray(y_full) == 0))
        if pos_count > 0 and neg_count > 0 and pos_count < neg_count:
            params["scale_pos_weight"] = float(neg_count) / float(pos_count)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_full, y_full.astype(int))
        return model, "binary"

    return None, mode


def _make_stratified_split(X, y, test_size=0.2, seed=SEED):
    if len(y) == 0:
        return X.iloc[0:0], X.iloc[0:0], y.iloc[0:0], y.iloc[0:0]

    counts = y.value_counts()
    if y.nunique() > 1 and min(counts) >= 2:
        return train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

    if y.nunique() == 2 and min(counts) == 1 and len(y) >= 3:
        rare_class = counts.idxmin()
        rare_idx = y.index[y == rare_class]
        other_idx = y.index[y != rare_class]
        n_val = max(2, int(round(len(y) * test_size)))
        n_val_other = max(1, n_val - 1)
        rng = np.random.default_rng(seed)
        chosen_other = rng.choice(
            other_idx.to_numpy(), size=min(n_val_other, len(other_idx)), replace=False
        )
        val_idx = pd.Index(list(rare_idx) + list(chosen_other))
        train_idx = y.index.difference(val_idx)
        return X.loc[train_idx], X.loc[val_idx], y.loc[train_idx], y.loc[val_idx]

    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=None)


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    target_col = TARGET if TARGET in train.columns else train.columns[-1]

    train = train.dropna(subset=[target_col]).reset_index(drop=True)
    test = test.drop(
        columns=[c for c in [target_col] if c in test.columns], errors="ignore"
    )

    y_raw = train[target_col].copy()
    X_full_raw = train.drop(columns=[target_col]).copy()

    feature_cols = [c for c in X_full_raw.columns if not _is_id_column(c)]
    X_full_raw = X_full_raw.reindex(columns=feature_cols)
    test = test.reindex(columns=feature_cols)

    y_full, target_info = _encode_binary_target(y_raw)
    y_full = y_full.reset_index(drop=True)
    X_full_raw = X_full_raw.reset_index(drop=True)

    if len(X_full_raw) == 0:
        X_full_raw = pd.DataFrame(
            {"constant_feature": np.zeros(len(train), dtype=np.float32)}
        )
        test = pd.DataFrame({"constant_feature": np.zeros(len(test), dtype=np.float32)})
        y_full = pd.Series(0, index=X_full_raw.index, dtype=int)
        target_info = {"mode": "empty", "positive": None, "classes": []}

    X_train_raw, X_val_raw, y_train, y_val = _make_stratified_split(
        X_full_raw, y_full, test_size=0.2, seed=SEED
    )

    processor = TabularFeatureProcessor()
    X_train = processor.fit_transform(X_train_raw, y_train)
    X_val = processor.transform(X_val_raw)
    X_test = processor.transform(test)
    X_full = processor.transform(X_full_raw)

    X_train.to_parquet(os.path.join(WORKING_DIR, "X_train.parquet"))
    X_val.to_parquet(os.path.join(WORKING_DIR, "X_val.parquet"))
    X_test.to_parquet(os.path.join(WORKING_DIR, "X_test.parquet"))
    y_train.to_csv(os.path.join(WORKING_DIR, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(WORKING_DIR, "y_val.csv"), index=False)
    with open(os.path.join(WORKING_DIR, "target_info.json"), "w") as f:
        json.dump(target_info, f, indent=2, default=str)

    model = None
    mode = "logistic_fallback"
    val_score = 0.5
    best_iter = 0
    fallback_mean = None

    if y_train.nunique() > 1 and y_val.nunique() > 1:
        try:
            model, val_score, best_iter, mode = _train_rank_model(
                X_train, y_train, X_val, y_val
            )
            if not np.isfinite(val_score) or val_score <= 0.5001:
                raise RuntimeError("rank model validation AUC is not informative")
        except Exception:
            try:
                model, val_score, best_iter, mode = _train_binary_model(
                    X_train, y_train, X_val, y_val
                )
            except Exception:
                model, fallback_mean = _train_logistic_fallback(X_train, y_train)
                mode = "logistic_fallback"
                val_score = _safe_auc(
                    y_val,
                    _postprocess_scores(
                        _predict_model(model, X_val, mode, fallback_mean), mode
                    ),
                )
                best_iter = 1
    elif y_train.nunique() > 1:
        try:
            model, val_score, best_iter, mode = _train_binary_model(
                X_train, y_train, X_val, y_val
            )
        except Exception:
            model, fallback_mean = _train_logistic_fallback(X_train, y_train)
            mode = "logistic_fallback"
            val_score = _safe_auc(
                y_val,
                _postprocess_scores(
                    _predict_model(model, X_val, mode, fallback_mean), mode
                ),
            )
            best_iter = 1
    else:
        model, fallback_mean = _train_logistic_fallback(X_train, y_train)
        mode = "logistic_fallback"
        val_score = 0.5
        best_iter = 1

    if mode in {"lambdarank", "binary"}:
        final_model, final_mode = _retrain_final_model(X_full, y_full, best_iter, mode)
    else:
        final_model, final_mode = model, mode

    if final_mode in {"lambdarank", "binary"}:
        val_pred = _postprocess_scores(_predict_model(model, X_val, mode), mode)
        test_pred = _postprocess_scores(
            _predict_model(final_model, X_test, final_mode), final_mode
        )
    else:
        val_pred = _postprocess_scores(
            _predict_model(model, X_val, mode, fallback_mean), mode
        )
        test_pred = _postprocess_scores(
            _predict_model(final_model, X_test, final_mode, fallback_mean), final_mode
        )

    final_val_score = _safe_auc(y_val, val_pred)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            TARGET: np.asarray(test_pred, dtype=float),
        }
    )
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()
