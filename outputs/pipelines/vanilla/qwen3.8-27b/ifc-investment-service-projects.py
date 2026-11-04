import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-investment-service-projects/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-investment-service-projects/data/test.parquet"
TARGET = "Total IFC investment as approved by Board(Million - USD)"

DROP_COLS = [
    "Project Name",
    "Company Name",
    "Project Url",
    "Project Number",
]

DATE_COLS = [
    "Date Disclosed",
    "Projected Board Date",
    "IFC Approval Date",
    "IFC Signed Date",
    "IFC Invested Date",
]

DATE_DIFF_PAIRS = [
    ("Date Disclosed", "Projected Board Date"),
    ("Date Disclosed", "IFC Approval Date"),
    ("Date Disclosed", "IFC Signed Date"),
    ("Date Disclosed", "IFC Invested Date"),
    ("Projected Board Date", "IFC Approval Date"),
    ("IFC Approval Date", "IFC Signed Date"),
    ("IFC Signed Date", "IFC Invested Date"),
    ("IFC Approval Date", "IFC Invested Date"),
]


def _clean_date_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    s = s.replace({"None": np.nan, "": np.nan, "nan": np.nan, "NaT": np.nan})
    return pd.to_datetime(s, errors="coerce")


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = {}

    for col in DATE_COLS:
        if col not in df.columns:
            continue

        dt = _clean_date_series(df[col])
        parsed[col] = dt
        base = col.replace(" ", "_")

        df[f"{base}_year"] = dt.dt.year
        df[f"{base}_month"] = dt.dt.month
        df[f"{base}_day"] = dt.dt.day
        df[f"{base}_dayofweek"] = dt.dt.dayofweek
        df[f"{base}_quarter"] = dt.dt.quarter

        df = df.drop(columns=[col])

    for start_col, end_col in DATE_DIFF_PAIRS:
        if start_col in parsed and end_col in parsed:
            start_base = start_col.replace(" ", "_")
            end_base = end_col.replace(" ", "_")
            diff = (parsed[end_col] - parsed[start_col]).dt.days
            df[f"{start_base}_to_{end_base}_days"] = diff

    return df


class Preprocessor:
    def __init__(self):
        self.numeric_cols = []
        self.cat_cols = []
        self.medians = {}
        self.encoder = None

    def _make_encoder(self):
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _prepare(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        df = df.copy()

        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])

        for col in DROP_COLS:
            if col in df.columns:
                df = df.drop(columns=[col])

        df = add_date_features(df)

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(object)

        if fit:
            self.numeric_cols = [
                col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
            ]
            self.cat_cols = [col for col in df.columns if col not in self.numeric_cols]

            for col in self.numeric_cols:
                med = df[col].median()
                if pd.isna(med):
                    med = 0.0
                self.medians[col] = float(med)

            if self.cat_cols:
                cat_df = df[self.cat_cols].fillna("missing").astype(str)
                self.encoder = self._make_encoder()
                self.encoder.fit(cat_df)
            else:
                self.encoder = None
        else:
            for col in self.numeric_cols:
                if col not in df.columns:
                    df[col] = np.nan
            for col in self.cat_cols:
                if col not in df.columns:
                    df[col] = np.nan

            df = df[self.numeric_cols + self.cat_cols]

        for col in self.numeric_cols:
            df[col] = df[col].fillna(self.medians.get(col, 0.0)).astype(float)

        if self.cat_cols and self.encoder is not None:
            cat_df = df[self.cat_cols].fillna("missing").astype(str)
            cat_enc = self.encoder.transform(cat_df)
            X = np.hstack([df[self.numeric_cols].to_numpy(dtype=float), cat_enc])
        elif self.numeric_cols:
            X = df[self.numeric_cols].to_numpy(dtype=float)
        else:
            X = np.zeros((len(df), 1), dtype=float)

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._prepare(df, fit=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._prepare(df, fit=False)


def make_model(name: str):
    if name == "lgbm":
        return LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgb":
        return XGBRegressor(
            n_estimators=5000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse",
        )
    if name == "catboost":
        return CatBoostRegressor(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        )
    raise ValueError(f"Unknown model: {name}")


def train_candidate(name: str, X_fit: np.ndarray, y_fit: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray):
    model = make_model(name)

    if name == "lgbm":
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best_iter = getattr(model, "best_iteration_", None)
    elif name == "xgb":
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )
        best_iter = getattr(model, "best_iteration", None)
    elif name == "catboost":
        model.fit(
            X_fit,
            y_fit,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0,
        )
        best_iter = model.get_best_iteration()
    else:
        raise ValueError(f"Unknown model: {name}")

    val_pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

    if best_iter is None:
        best_iter = 1000
    else:
        best_iter = int(best_iter) + 1
        best_iter = max(100, min(5000, best_iter))

    return model, rmse, best_iter


def refit_model(name: str, X_full: np.ndarray, y_full: np.ndarray, best_iter: int):
    model = make_model(name)
    model.set_params(n_estimators=best_iter)
    model.fit(X_full, y_full)
    return model


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)

    if len(train_df) == 0:
        pred = np.zeros(len(test_df), dtype=float)
    elif len(train_df) < 20:
        pred = np.full(len(test_df), float(train_df[TARGET].mean()), dtype=float)
    else:
        X_all = train_df.drop(columns=[TARGET])
        y_all = train_df[TARGET].astype(float)

        X_fit, X_val, y_fit, y_val = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=42,
        )

        val_preprocessor = Preprocessor()
        X_fit_proc = val_preprocessor.fit_transform(X_fit)
        X_val_proc = val_preprocessor.transform(X_val)

        candidates = []
        for name in ["lgbm", "xgb", "catboost"]:
            try:
                model, rmse, best_iter = train_candidate(
                    name, X_fit_proc, y_fit, X_val_proc, y_val
                )
                candidates.append((name, rmse, best_iter))
            except Exception:
                continue

        if not candidates:
            final_preprocessor = Preprocessor()
            X_full_proc = final_preprocessor.fit_transform(X_all)
            X_test_proc = final_preprocessor.transform(test_df)

            model = LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_full_proc, y_all)
            pred = model.predict(X_test_proc)
        else:
            candidates.sort(key=lambda item: item[1])
            best_name, _, best_iter = candidates[0]

            final_preprocessor = Preprocessor()
            X_full_proc = final_preprocessor.fit_transform(X_all)
            X_test_proc = final_preprocessor.transform(test_df)

            try:
                final_model = refit_model(best_name, X_full_proc, y_all, best_iter)
                pred = final_model.predict(X_test_proc)
            except Exception:
                final_model = LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
                final_model.fit(X_full_proc, y_all)
                pred = final_model.predict(X_test_proc)

    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=float(train_df[TARGET].mean()), posinf=0.0, neginf=0.0)

    out = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        TARGET: pred,
    })
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()