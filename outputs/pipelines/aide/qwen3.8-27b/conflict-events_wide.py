import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

TARGET = "fatalities"
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = (
    pd.to_numeric(train[TARGET], errors="coerce")
    .fillna(0.0)
    .clip(lower=0.0)
    .astype(float)
)
X_train = train.drop(columns=[TARGET], errors="ignore").copy()
X_test = test.drop(columns=[TARGET], errors="ignore").copy()

X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]


def is_id_column(name):
    s = str(name).lower()
    return s == "id" or s.endswith("_id") or s.startswith("id_") or "identifier" in s


def to_datetime_naive(series):
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        try:
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_localize(None)
        except AttributeError:
            pass
    return parsed


class Preprocessor:
    def __init__(self):
        self.source_cols = []
        self.datetime_cols = []
        self.datetime_origins = {}
        self.category_cols = []
        self.category_maps = {}
        self.feature_names = []
        self.has_constant = False

    def _detect_datetime_columns(self, X):
        self.datetime_cols = []
        for col in self.source_cols:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.datetime_cols.append(col)
                continue

            if X[col].dtype == object or str(X[col].dtype).startswith("string"):
                sample = X[col].dropna().astype(str).head(200)
                if sample.empty:
                    continue

                if (
                    sample.str.contains(
                        r"\d{2,4}[-/]\d{1,2}[-/]\d{1,4}", na=False
                    ).mean()
                    < 0.8
                ):
                    continue

                parsed = to_datetime_naive(sample)
                if parsed.notna().mean() >= 0.8:
                    self.datetime_cols.append(col)

    def fit(self, X):
        X = X.copy()
        X = X.drop(columns=[c for c in X.columns if is_id_column(c)], errors="ignore")
        self.source_cols = list(X.columns)

        self._detect_datetime_columns(X)

        for col in self.datetime_cols:
            parsed = to_datetime_naive(X[col])
            self.datetime_origins[col] = (
                parsed.min() if parsed.notna().any() else pd.Timestamp("1970-01-01")
            )

        self.category_cols = [
            col
            for col in self.source_cols
            if col not in self.datetime_cols
            and (
                X[col].dtype == object or isinstance(X[col].dtype, pd.CategoricalDtype)
            )
        ]

        self.category_maps = {}
        for col in self.category_cols:
            vals = X[col].dropna()
            cats = sorted(vals.astype(str).unique().tolist()) if len(vals) else []
            self.category_maps[col] = {v: i for i, v in enumerate(cats)}

        Xt = self.transform(X, select=False)

        keep = []
        for col in Xt.columns:
            s = Xt[col]
            if s.notna().sum() == 0:
                continue
            if s.nunique(dropna=True) <= 1:
                continue
            keep.append(col)

        if keep:
            self.feature_names = keep
            self.has_constant = False
        else:
            self.feature_names = ["__constant__"]
            self.has_constant = True

        return self

    def transform(self, X, select=True):
        X = X.copy()
        X = X.drop(columns=[c for c in X.columns if is_id_column(c)], errors="ignore")
        X = X.reindex(columns=self.source_cols, fill_value=np.nan)

        out = pd.DataFrame(index=X.index)

        for col in self.source_cols:
            if col in self.datetime_cols:
                parsed = to_datetime_naive(X[col])
                origin = self.datetime_origins[col]
                out[col] = (parsed - origin).dt.total_seconds() / 86400.0
            elif col in self.category_maps:
                mapped = X[col].astype(str).map(self.category_maps[col])
                mapped[X[col].isna()] = np.nan
                out[col] = mapped
            else:
                out[col] = pd.to_numeric(X[col], errors="coerce")

        if select:
            out = out.reindex(columns=self.feature_names, fill_value=np.nan)
            if self.has_constant:
                out["__constant__"] = 0.0

        return out


params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "alpha": 0.1,
    "lambda": 5.0,
    "tree_method": "hist",
    "nthread": -1,
    "seed": 42,
}


def predict_with_limit(booster, dmat, n_trees):
    n_trees = int(n_trees)
    if n_trees <= 0:
        return booster.predict(dmat)
    try:
        return booster.predict(dmat, iteration_range=(0, n_trees))
    except TypeError:
        return booster.predict(dmat, ntree_limit=n_trees)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y), dtype=float)
best_tree_counts = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
    pre = Preprocessor().fit(X_train.iloc[tr_idx])

    Xtr = pre.transform(X_train.iloc[tr_idx])
    Xva = pre.transform(X_train.iloc[va_idx])
    ytr = y.iloc[tr_idx]
    yva = y.iloc[va_idx]

    dtrain = xgb.DMatrix(Xtr, label=ytr.values)
    dval = xgb.DMatrix(Xva, label=yva.values)

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
        evals_result=evals_result,
    )

    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is None:
        val_rmse = evals_result.get("val", {}).get("rmse", [])
        best_iter = int(np.argmin(val_rmse)) if len(val_rmse) else MAX_ROUNDS - 1

    best_iter = int(best_iter)
    if best_iter < 0:
        best_iter = MAX_ROUNDS - 1

    n_trees = min(best_iter + 1, MAX_ROUNDS)
    best_tree_counts.append(n_trees)

    pred = predict_with_limit(booster, dval, n_trees)
    oof[va_idx] = pred
    print(f"Fold {fold} RMSE: {root_mean_squared_error(yva, pred):.6f}")

cv_rmse = root_mean_squared_error(y, oof)
print(f"CV RMSE: {cv_rmse:.6f}")

pre_final = Preprocessor().fit(X_train)
Xtr_final = pre_final.transform(X_train)

if best_tree_counts:
    final_trees = max(1, int(np.mean(best_tree_counts) + 0.5))
else:
    final_trees = 100

print(f"Final boosting rounds: {final_trees}")

if len(Xtr_final) > 0:
    dtrain_final = xgb.DMatrix(Xtr_final, label=y.values)
    booster_final = xgb.train(params, dtrain_final, num_boost_round=final_trees)
else:
    booster_final = None

Xte = pre_final.transform(X_test)

if len(Xte) > 0:
    if booster_final is not None:
        test_pred = booster_final.predict(xgb.DMatrix(Xte))
    else:
        test_pred = np.full(len(Xte), y.mean() if len(y) else 0.0, dtype=float)
else:
    test_pred = np.array([], dtype=float)

test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)
test_pred = np.clip(test_pred, 0.0, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "fatalities": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
