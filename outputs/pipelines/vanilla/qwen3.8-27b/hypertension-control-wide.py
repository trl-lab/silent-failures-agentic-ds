import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/hypertension-control-wide/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/hypertension-control-wide/data/test.parquet"
TARGET = "Estimated Percentage of Patients with Controlled Blood Pressure"
RANDOM_STATE = 42


class FeatureTransformer:
    def __init__(self, onehot_threshold=20):
        self.onehot_threshold = onehot_threshold

    def _add_date_features(self, df):
        df = df.copy()
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        self.date_cols_ = date_cols

        for col in date_cols:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_quarter"] = df[col].dt.quarter

        start_col = "Project Period Start Date"
        end_col = "Grant Project Period End Date"
        self.duration_pair_ = None
        if start_col in df.columns and end_col in df.columns:
            df["grant_duration_days"] = (df[end_col] - df[start_col]).dt.days
            self.duration_pair_ = (start_col, end_col)

        df = df.drop(columns=date_cols, errors="ignore")
        return df

    def fit(self, X):
        X = self._add_date_features(X)

        drop_cols = []
        for col in X.columns:
            if X[col].isna().all() or X[col].nunique(dropna=True) <= 1:
                drop_cols.append(col)
        X = X.drop(columns=drop_cols, errors="ignore")

        self.numeric_cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat_cols_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

        self.numeric_medians_ = {}
        self.missing_indicators_ = []
        keep_numeric = []
        for col in self.numeric_cols_:
            median = X[col].median()
            if pd.isna(median):
                median = 0.0
            self.numeric_medians_[col] = float(median)
            keep_numeric.append(col)
            if X[col].isna().any():
                self.missing_indicators_.append(f"{col}_missing")
        self.numeric_cols_ = keep_numeric

        self.onehot_cols_ = []
        self.freq_cols_ = []
        self.freq_maps_ = {}
        keep_cat = []

        for col in self.cat_cols_:
            series = X[col].fillna("missing").astype(str)
            if series.nunique() <= 1:
                continue
            keep_cat.append(col)
            if series.nunique() <= self.onehot_threshold:
                self.onehot_cols_.append(col)
            else:
                self.freq_cols_.append(col)
                self.freq_maps_[col] = series.value_counts().to_dict()

        self.cat_cols_ = keep_cat

        if self.onehot_cols_:
            self.onehot_encoder_ = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=float,
            )
            self.onehot_encoder_.fit(X[self.onehot_cols_].fillna("missing").astype(str))
            self.onehot_feature_names_ = self.onehot_encoder_.get_feature_names_out().tolist()
        else:
            self.onehot_encoder_ = None
            self.onehot_feature_names_ = []

        self.feature_names_ = (
            self.numeric_cols_
            + self.missing_indicators_
            + self.onehot_feature_names_
            + self.freq_cols_
        )
        return self

    def transform(self, X):
        X = self._add_date_features(X)

        out = pd.DataFrame(index=X.index)

        for col in self.numeric_cols_:
            if col in X.columns:
                out[col] = X[col].fillna(self.numeric_medians_[col])
            else:
                out[col] = self.numeric_medians_[col]

        for col in self.missing_indicators_:
            base = col[:-9]
            if base in X.columns:
                out[col] = X[base].isna().astype(float)
            else:
                out[col] = 0.0

        if self.onehot_cols_ and self.onehot_encoder_ is not None:
            cat_df = X.reindex(columns=self.onehot_cols_, fill_value="missing").astype(str)
            onehot = self.onehot_encoder_.transform(cat_df)
            out[self.onehot_feature_names_] = onehot

        for col in self.freq_cols_:
            if col in X.columns:
                values = X[col].fillna("missing").astype(str)
                out[col] = values.map(self.freq_maps_[col]).fillna(0.0).astype(float)
            else:
                out[col] = 0.0

        return out[self.feature_names_]


def make_binary_target(y_raw):
    y = pd.to_numeric(y_raw, errors="coerce").astype(float)
    unique = y.dropna().unique()

    if len(unique) == 0:
        return np.zeros(len(y), dtype=int), 1.0

    if len(unique) == 1:
        return (y == unique[0]).astype(int), unique[0]

    if len(unique) == 2:
        positive = 1.0 if 1.0 in unique else float(np.max(unique))
        return (y == positive).astype(int), positive

    threshold = float(y.median())
    return (y >= threshold).astype(int), threshold


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def train_candidate(name, X_train, y_train, X_val, y_val):
    if name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[],
        )
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None or best_iter <= 0:
            best_iter = 1000
        return model, int(best_iter)

    if name == "xgboost":
        model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None or best_iter <= 0:
            best_iter = 1000
        return model, int(best_iter)

    if name == "catboost":
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
            eval_metric="AUC",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0,
        )
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None or best_iter < 0:
            best_iter = 1000
        return model, int(best_iter + 1)

    raise ValueError(f"Unknown model: {name}")


def refit_candidate(name, X_train, y_train, n_estimators):
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ).fit(X_train, y_train)

    if name == "xgboost":
        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ).fit(X_train, y_train)

    if name == "catboost":
        return CatBoostClassifier(
            iterations=n_estimators,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        ).fit(X_train, y_train)

    raise ValueError(f"Unknown model: {name}")


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_raw = train_df[TARGET]
    y, _ = make_binary_target(y_raw)

    X_train_full = train_df.drop(columns=[TARGET], errors="ignore")
    X_test = test_df.drop(columns=[TARGET], errors="ignore")

    common_cols = [c for c in X_train_full.columns if c in X_test.columns]
    X_train_full = X_train_full[common_cols]
    X_test = X_test[common_cols]

    if len(np.unique(y)) < 2:
        predictions = np.full(len(test_df), 0.5, dtype=float)
        pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                TARGET: predictions,
            }
        ).to_csv("predictions.csv", index=False)
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    transformer = FeatureTransformer(onehot_threshold=20).fit(X_train)
    X_train_enc = transformer.transform(X_train)
    X_val_enc = transformer.transform(X_val)
    X_test_enc = transformer.transform(X_test)

    candidates = ["lightgbm", "xgboost", "catboost"]
    best_name = None
    best_auc = -np.inf
    best_n_estimators = 1000

    for name in candidates:
        try:
            model, best_iter = train_candidate(name, X_train_enc, y_train, X_val_enc, y_val)
            val_pred = model.predict_proba(X_val_enc)[:, 1]
            auc = safe_auc(y_val, val_pred)
            if np.isfinite(auc) and auc > best_auc:
                best_auc = auc
                best_name = name
                best_n_estimators = best_iter
        except Exception:
            continue

    if best_name is None:
        best_name = "lightgbm"
        best_n_estimators = 1000

    final_transformer = FeatureTransformer(onehot_threshold=20).fit(X_train_full)
    X_train_final = final_transformer.transform(X_train_full)
    X_test_final = final_transformer.transform(X_test)

    final_model = refit_candidate(best_name, X_train_final, y, best_n_estimators)
    test_pred = final_model.predict_proba(X_test_final)[:, 1]

    pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            TARGET: test_pred,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()