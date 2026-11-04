import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb

SEED = 42
TARGET = "Amount in USD"
WORK_DIR = "./working"
SUB_DIR = "./submission"

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

np.random.seed(SEED)


def is_id_col(name: str) -> bool:
    n = str(name).strip().lower()
    id_names = {
        "id",
        "index",
        "idx",
        "row_id",
        "rowid",
        "unique_id",
        "uuid",
        "identifier",
        "submission_id",
        "sample_id",
        "record_id",
    }
    return n in id_names or n.startswith("id_") or n.endswith("_id")


def is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = s.dropna().head(200)
        if len(sample) == 0:
            return False
        parsed = pd.to_numeric(sample, errors="coerce")
        return parsed.notna().mean() > 0.8
    return False


def is_datetime_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().head(200)
    if len(sample) == 0:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() > 0.8


def sign_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official RMSE in original target space, with finite-value masking."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))


class FeatureBuilder:
    def __init__(self, target_col: str = TARGET, smoothing: float = 10.0):
        self.target_col = target_col
        self.smoothing = smoothing
        self.feature_names = []
        self.global_target_mean = 0.0
        self._meta = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureBuilder":
        X = X.copy()
        y = pd.Series(y).astype(float)
        self.global_target_mean = float(y.mean()) if len(y) else 0.0

        drop_cols = [c for c in X.columns if is_id_col(c) or c == self.target_col]
        X = X.drop(columns=drop_cols, errors="ignore")

        self._meta = {}

        for col in X.columns:
            s = X[col]
            if is_numeric_series(s):
                self._fit_numeric(col, s)
            elif is_datetime_series(s):
                self._fit_datetime(col, s)
            else:
                self._fit_categorical(col, s, y)

        train_out = self._transform_raw(X)
        train_out = self._clean_frame(train_out)

        keep = []
        for c in train_out.columns:
            v = train_out[c]
            if v.isna().mean() > 0.95:
                continue
            if not np.isfinite(v.to_numpy(dtype=float, na_value=np.nan)).any():
                continue
            if v.nunique(dropna=True) <= 1:
                continue
            keep.append(c)

        if len(keep) == 0 and train_out.shape[1] > 0:
            keep = [train_out.columns[0]]

        self.feature_names = keep
        return self

    def _fit_numeric(self, col: str, s: pd.Series):
        x = pd.to_numeric(s, errors="coerce")
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            return

        median = float(finite.median()) if finite.notna().any() else 0.0
        if len(finite) > 20:
            q_low = float(finite.quantile(0.001))
            q_high = float(finite.quantile(0.999))
        else:
            q_low = float(finite.min())
            q_high = float(finite.max())

        if not np.isfinite(q_low):
            q_low = median
        if not np.isfinite(q_high):
            q_high = median
        if q_low > q_high:
            q_low, q_high = q_high, q_low
        if q_low == q_high:
            q_low = q_high = median

        clipped = finite.clip(q_low, q_high)
        mean = float(clipped.mean()) if len(clipped) else median
        std = float(clipped.std()) if len(clipped) else 0.0
        if not np.isfinite(std) or std == 0:
            std = 1.0

        self._meta[col] = {
            "type": "numeric",
            "median": median,
            "q_low": q_low,
            "q_high": q_high,
            "mean": mean,
            "std": std,
            "nonnegative": bool((finite >= 0).all()),
        }

    def _fit_datetime(self, col: str, s: pd.Series):
        dt = pd.to_datetime(s, errors="coerce")
        valid = dt.dropna()
        if len(valid) == 0:
            return
        self._meta[col] = {
            "type": "datetime",
            "min": valid.min(),
            "has_time": bool(
                (valid.dt.hour != 0).any() or (valid.dt.minute != 0).any()
            ),
        }

    def _fit_categorical(self, col: str, s: pd.Series, y: pd.Series):
        s_clean = s.where(s.notna(), "missing").astype(str)
        if s_clean.nunique() == 0:
            return

        counts = s_clean.value_counts()
        freq = counts / max(len(s_clean), 1)

        global_mean = float(y.mean()) if len(y) else 0.0
        tmp = pd.DataFrame({"cat": s_clean.to_numpy(), "y": y.to_numpy(dtype=float)})
        tmp = tmp[np.isfinite(tmp["y"])]
        if len(tmp) == 0:
            target_enc = pd.Series(dtype=float)
        else:
            sums = tmp.groupby("cat")["y"].sum()
            cnts = tmp.groupby("cat")["y"].count()
            target_enc = (sums + global_mean * self.smoothing) / (cnts + self.smoothing)

        onehot_categories = None
        if s_clean.nunique() <= 10:
            onehot_categories = sorted(s_clean.unique().tolist())

        self._meta[col] = {
            "type": "categorical",
            "freq": freq,
            "target_enc": target_enc,
            "onehot_categories": onehot_categories,
            "global_mean": global_mean,
        }

    def _transform_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        out = pd.DataFrame(index=X.index)

        for col in X.columns:
            meta = self._meta.get(col)
            if meta is None:
                continue
            t = meta["type"]

            if t == "numeric":
                x = pd.to_numeric(X[col], errors="coerce")
                missing = x.isna().astype(float)
                x = x.fillna(meta["median"]).clip(meta["q_low"], meta["q_high"])
                out[col] = x.astype(float)
                out[f"{col}__missing"] = missing.astype(float)

                if meta["nonnegative"]:
                    out[f"{col}__log"] = np.log1p(np.maximum(x, 0.0)).astype(float)
                else:
                    out[f"{col}__log"] = sign_log(x.to_numpy(dtype=float)).astype(float)

                scaled = (x - meta["mean"]) / meta["std"]
                out[f"{col}__scaled"] = scaled.astype(float)

            elif t == "datetime":
                dt = pd.to_datetime(X[col], errors="coerce")
                missing = dt.isna().astype(float)
                out[f"{col}__missing"] = missing.astype(float)

                min_dt = meta["min"]
                days = (dt - min_dt).dt.total_seconds() / 86400.0
                out[f"{col}__days"] = days.fillna(0.0).astype(float)

                out[f"{col}__year"] = dt.dt.year.fillna(0).astype(float)
                out[f"{col}__month"] = dt.dt.month.fillna(0).astype(float)
                out[f"{col}__day"] = dt.dt.day.fillna(0).astype(float)
                out[f"{col}__dayofweek"] = dt.dt.dayofweek.fillna(0).astype(float)
                out[f"{col}__quarter"] = dt.dt.quarter.fillna(0).astype(float)
                out[f"{col}__is_weekend"] = (
                    (dt.dt.dayofweek >= 5).astype(float).fillna(0.0)
                )

                month = dt.dt.month.fillna(0).astype(float)
                dow = dt.dt.dayofweek.fillna(0).astype(float)
                out[f"{col}__month_sin"] = np.sin(2 * np.pi * month / 12.0)
                out[f"{col}__month_cos"] = np.cos(2 * np.pi * month / 12.0)
                out[f"{col}__dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
                out[f"{col}__dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

                if meta["has_time"]:
                    out[f"{col}__hour"] = dt.dt.hour.fillna(0).astype(float)
                    out[f"{col}__minute"] = dt.dt.minute.fillna(0).astype(float)

            elif t == "categorical":
                s_clean = X[col].where(X[col].notna(), "missing").astype(str)
                missing = X[col].isna().astype(float)
                out[f"{col}__missing"] = missing.astype(float)

                out[f"{col}__freq"] = (
                    s_clean.map(meta["freq"]).fillna(0.0).astype(float)
                )
                out[f"{col}__target_enc"] = (
                    s_clean.map(meta["target_enc"])
                    .fillna(meta["global_mean"])
                    .astype(float)
                )

                out[f"{col}__len"] = s_clean.str.len().fillna(0).astype(float)
                out[f"{col}__words"] = (
                    s_clean.str.split().str.len().fillna(0).astype(float)
                )
                out[f"{col}__digits"] = s_clean.str.count(r"\d").fillna(0).astype(float)

                if meta["onehot_categories"] is not None:
                    for i, cat in enumerate(meta["onehot_categories"]):
                        out[f"{col}__oh_{i}"] = (s_clean == cat).astype(float)

        if out.shape[1] == 0:
            out["bias"] = 1.0

        return out

    def _clean_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)
        return df.astype(np.float32)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.reindex(columns=list(self._meta.keys()), fill_value=np.nan)
        out = self._transform_raw(X)
        out = self._clean_frame(out)
        if len(self.feature_names) > 0:
            out = out.reindex(columns=self.feature_names, fill_value=0.0)
        return out.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


class EnsembleRegressor:
    def __init__(self):
        self.models = []
        self._fitted = False
        self._use_log = False
        self._n_models_total = 0

    def _make_lgbm(self, seed: int):
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=3000,
            learning_rate=0.01,
            num_leaves=15,
            max_depth=4,
            min_child_samples=5,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    def _make_xgb(self, seed: int):
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=3000,
            learning_rate=0.01,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=50,
            tree_method="hist",
            eval_metric="rmse",
        )

    def fit(
        self,
        X_trn: np.ndarray,
        y_trn: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "EnsembleRegressor":
        X_trn = np.asarray(X_trn, dtype=np.float32)
        y_trn = np.asarray(y_trn, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)

        finite_y = y_trn[np.isfinite(y_trn)]
        self._use_log = bool(finite_y.size > 0 and np.min(finite_y) >= 0.0)
        if self._use_log:
            y_trn_log = np.log1p(np.maximum(y_trn, 0.0)).astype(np.float32)
            y_val_log = np.log1p(np.maximum(y_val, 0.0)).astype(np.float32)
        else:
            y_trn_log = y_trn
            y_val_log = y_val

        self.models = []
        for seed in (42, 43, 44):
            lgbm_raw = self._make_lgbm(seed)
            lgbm_raw.fit(
                X_trn,
                y_trn,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            self.models.append((lgbm_raw, False))

            xgb_raw = self._make_xgb(seed)
            xgb_raw.fit(
                X_trn,
                y_trn,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.models.append((xgb_raw, False))

            if self._use_log:
                lgbm_log = self._make_lgbm(seed)
                lgbm_log.fit(
                    X_trn,
                    y_trn_log,
                    eval_set=[(X_val, y_val_log)],
                    eval_metric="rmse",
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                self.models.append((lgbm_log, True))

                xgb_log = self._make_xgb(seed)
                xgb_log.fit(
                    X_trn,
                    y_trn_log,
                    eval_set=[(X_val, y_val_log)],
                    verbose=False,
                )
                self.models.append((xgb_log, True))

        self._n_models_total = len(self.models)

        raw_preds = []
        log_preds = []
        for model, is_log in self.models:
            pred = model.predict(X_val)
            if is_log:
                pred = np.expm1(pred)
            if is_log:
                log_preds.append(pred)
            else:
                raw_preds.append(pred)

        raw_pred = np.mean(raw_preds, axis=0) if raw_preds else np.zeros_like(y_val)
        log_pred = np.mean(log_preds, axis=0) if log_preds else raw_pred
        ens_pred = (raw_pred + log_pred) / 2.0 if self._use_log else raw_pred

        print(f"Raw Models Validation RMSE: {safe_rmse(y_val, raw_pred):.6f}")
        if self._use_log:
            print(f"Log Models Validation RMSE: {safe_rmse(y_val, log_pred):.6f}")
        print(f"Full Ensemble Validation RMSE: {safe_rmse(y_val, ens_pred):.6f}")

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict.")
        X = np.asarray(X, dtype=np.float32)
        all_preds = []
        for model, is_log in self.models:
            pred = model.predict(X)
            if is_log:
                pred = np.expm1(pred)
            all_preds.append(pred)
        return np.mean(all_preds, axis=0)


def _postprocess_predictions(pred: np.ndarray, nonnegative: bool) -> np.ndarray:
    """Validation and test post-processing are identical."""
    pred = np.asarray(pred, dtype=float)
    if nonnegative:
        pred = np.maximum(pred, 0.0)
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)


# PyTorch-specific prediction helper removed.


# PyTorch-specific training loop removed.


def load_data():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train.parquet")

    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train = train.replace([np.inf, -np.inf], np.nan)
    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test


def make_train_valid_split(train: pd.DataFrame):
    """Prefer chronological split if a reliable datetime column exists."""
    date_col = None
    for col in train.columns:
        if col == TARGET:
            continue
        if is_datetime_series(train[col]):
            valid_frac = pd.to_datetime(train[col], errors="coerce").notna().mean()
            if valid_frac > 0.8:
                date_col = col
                break

    if date_col is not None:
        tmp = train.copy()
        tmp["_split_date"] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.sort_values("_split_date", kind="mergesort")
        n_valid = max(1, int(0.2 * len(tmp)))
        n_valid = min(n_valid, max(0, len(tmp) - 1))
        if n_valid == 0:
            n_valid = 1
        valid = tmp.iloc[-n_valid:].drop(columns=["_split_date"]).reset_index(drop=True)
        trn = tmp.iloc[:-n_valid].drop(columns=["_split_date"]).reset_index(drop=True)
        split_type = "chronological"
    else:
        try:
            tmp = train.copy()
            tmp["_target_bin"] = pd.qcut(tmp[TARGET], q=5, duplicates="drop")
            trn, valid = train_test_split(
                tmp,
                test_size=0.2,
                random_state=SEED,
                stratify=tmp["_target_bin"],
            )
            trn = trn.drop(columns=["_target_bin"]).reset_index(drop=True)
            valid = valid.drop(columns=["_target_bin"]).reset_index(drop=True)
        except Exception:
            trn, valid = train_test_split(train, test_size=0.2, random_state=SEED)
            trn = trn.reset_index(drop=True)
            valid = valid.reset_index(drop=True)
        split_type = "random_stratified"

    return trn, valid, split_type


def main():
    train_raw, test_raw = load_data()

    trn_raw, val_raw, split_type = make_train_valid_split(train_raw)

    X_trn_raw = trn_raw.drop(columns=[TARGET], errors="ignore")
    X_val_raw = val_raw.drop(columns=[TARGET], errors="ignore")
    X_test_raw = test_raw.copy()

    y_trn = trn_raw[TARGET].astype(float)
    y_val = val_raw[TARGET].astype(float)

    fb = FeatureBuilder(target_col=TARGET, smoothing=10.0)
    X_trn = fb.fit_transform(X_trn_raw, y_trn)
    X_val = fb.transform(X_val_raw)
    X_test = fb.transform(X_test_raw)

    X_trn_np = np.nan_to_num(
        X_trn.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_val_np = np.nan_to_num(
        X_val.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    X_test_np = np.nan_to_num(
        X_test.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    y_trn_np = np.nan_to_num(
        y_trn.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    y_val_np = np.nan_to_num(
        y_val.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    if len(X_trn_np) != len(y_trn_np):
        raise ValueError("X_train and y_train length mismatch.")
    if len(X_val_np) != len(y_val_np):
        raise ValueError("X_valid and y_valid length mismatch.")

    n_features = X_trn_np.shape[1]
    if n_features == 0:
        X_trn_np = np.ones((len(y_trn_np), 1), dtype=np.float32)
        X_val_np = np.ones((len(y_val_np), 1), dtype=np.float32)
        X_test_np = np.ones((len(X_test_np), 1), dtype=np.float32)
        n_features = 1

    nonnegative = bool(np.nanmin(y_trn_np) >= 0.0) if len(y_trn_np) > 0 else True
    zero_frac = float(np.mean(y_trn_np <= 1e-8)) if nonnegative else 0.0
    zero_inflated = bool(nonnegative and zero_frac > 0.02)

    model = EnsembleRegressor()
    model.fit(X_trn_np, y_trn_np, X_val_np, y_val_np)

    val_pred = _postprocess_predictions(model.predict(X_val_np), nonnegative=nonnegative)
    test_pred = _postprocess_predictions(model.predict(X_test_np), nonnegative=nonnegative)

    score = safe_rmse(y_val_np, val_pred)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            TARGET: test_pred.astype(float),
        }
    )
    submission.to_csv(os.path.join(SUB_DIR, "submission.csv"), index=False)

    design_meta = {
        "model_class": "EnsembleRegressor",
        "models": ["LGBM_raw", "XGB_raw", "LGBM_log", "XGB_log"],
        "n_seeds": 3,
        "transform_ensemble": True,
        "n_models_total": 12,
        "target_transform": "log1p raw+log blend when nonnegative",
        "n_features": int(n_features),
        "nonnegative": nonnegative,
        "zero_inflated": zero_inflated,
        "zero_fraction_train": zero_frac,
        "loss": "regression_rmse_early_stopping",
        "lightgbm": {
            "objective": "regression",
            "n_estimators": 3000,
            "learning_rate": 0.01,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 5,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
        "xgboost": {
            "objective": "reg:squarederror",
            "n_estimators": 3000,
            "learning_rate": 0.01,
            "max_depth": 4,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "early_stopping_rounds": 50,
        },
        "split_type": split_type,
        "n_train": int(len(X_trn_np)),
        "n_valid": int(len(X_val_np)),
        "n_test": int(len(X_test_np)),
        "validation_metric": "rmse_original_target_space",
        "final_validation_score": float(score),
    }
    with open(os.path.join(WORK_DIR, "model_design_meta.json"), "w") as f:
        json.dump(design_meta, f, indent=2)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
