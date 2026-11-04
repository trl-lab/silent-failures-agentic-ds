import os
import re
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)

TARGET = "time_to_approve"
RANDOM_STATE = 42
N_SPLITS = 5
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Keep only rows with valid target values.
y = train[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
valid_mask = y.notna()
train = train.loc[valid_mask].copy()
y = y.loc[valid_mask].reset_index(drop=True)
train = train.reset_index(drop=True)

X = train.drop(columns=[TARGET])


def is_id_like(name: str) -> bool:
    n = str(name).lower()
    if n in {"id", "index", "uuid", "guid", "identifier"}:
        return True
    if re.search(r"(^|_)id($|_)", n):
        return True
    if n.endswith("_id") or n.startswith("id_"):
        return True
    if re.search(r"(number|reference|case|application_no|permit_no|uuid)", n):
        return True
    return False


def is_target_related(name: str) -> bool:
    n = str(name).lower()
    pattern = (
        r"approv|time_to|duration|elapsed|days_to|processing|status|outcome|"
        r"decision|result|final|closed|completion|expiration|expired|renew|disposition"
    )
    return re.search(pattern, n) is not None


# Initial leakage / ID column removal.
drop_cols = {TARGET}
for col in X.columns:
    if is_id_like(col) or is_target_related(col):
        drop_cols.add(col)

# Identify date-like columns.
date_cols = []
for col in X.columns:
    if col in drop_cols:
        continue
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        date_cols.append(col)
        continue
    n = str(col).lower()
    if any(
        token in n
        for token in ["date", "time", "timestamp", "created", "submitted", "filed"]
    ):
        sample = X[col].dropna().head(200)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.5:
                date_cols.append(col)

# Classify remaining columns.
numeric_cols = []
categorical_cols = []
for col in X.columns:
    if col in drop_cols or col in date_cols:
        continue
    if pd.api.types.is_numeric_dtype(X[col]):
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

# Remove near-constant, highly missing, and very high-cardinality categorical columns.
for col in X.columns:
    if col in drop_cols:
        continue
    s = X[col]
    missing_rate = s.isna().mean()
    nunique = s.nunique(dropna=True)
    if missing_rate > 0.95 or nunique <= 1:
        drop_cols.add(col)
    if col in categorical_cols and nunique > 5000:
        drop_cols.add(col)

raw_cols = [c for c in X.columns if c not in drop_cols]
date_cols = [c for c in date_cols if c in raw_cols]
numeric_cols = [c for c in numeric_cols if c in raw_cols]
categorical_cols = [c for c in categorical_cols if c in raw_cols]

# Build a stable feature-name list after date expansion.
feature_names = []
feature_names.extend(numeric_cols)
feature_names.extend(categorical_cols)
for col in date_cols:
    feature_names.extend(
        [
            f"{col}_year",
            f"{col}_month",
            f"{col}_day",
            f"{col}_dayofweek",
            f"{col}_days_since_epoch",
        ]
    )


def fit_categorical_mappings(X_train: pd.DataFrame, cols: list) -> dict:
    mappings = {}
    for col in cols:
        if col not in X_train.columns:
            continue
        values = X_train[col].dropna().astype("string")
        mappings[col] = {value: idx for idx, value in enumerate(values.unique())}
    return mappings


def preprocess(
    X_raw: pd.DataFrame,
    mappings: dict,
    numeric_cols: list,
    categorical_cols: list,
    date_cols: list,
    feature_names: list,
) -> pd.DataFrame:
    X = X_raw.copy()

    # Numeric conversion.
    for col in numeric_cols:
        if col not in X.columns:
            continue
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Categorical encoding using train-only mappings.
    for col in categorical_cols:
        if col not in X.columns:
            continue
        mapping = mappings.get(col, {})
        X[col] = X[col].astype("string").map(mapping).fillna(-1).astype("int32")

    # Date expansion.
    for col in date_cols:
        if col not in X.columns:
            continue
        dt = pd.to_datetime(X[col], errors="coerce")
        X[f"{col}_year"] = dt.dt.year.astype("float32")
        X[f"{col}_month"] = dt.dt.month.astype("float32")
        X[f"{col}_day"] = dt.dt.day.astype("float32")
        X[f"{col}_dayofweek"] = dt.dt.dayofweek.astype("float32")
        X[f"{col}_days_since_epoch"] = (
            dt.astype("int64", errors="ignore") // 10**9
        ).astype("float64")

    X = X.drop(columns=[c for c in date_cols if c in X.columns], errors="ignore")

    # Ensure all expected feature columns are present.
    for col in feature_names:
        if col not in X.columns:
            X[col] = np.nan

    return X[feature_names]


def make_model(
    n_estimators: int, early_stopping_rounds: int | None = None
) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


# Fallback if feature selection removes everything.
if len(feature_names) == 0:
    fallback = float(y.mean()) if len(y) > 0 else 0.0
    cv_rmse = (
        float(root_mean_squared_error(y, np.full(len(y), fallback)))
        if len(y) > 0
        else float("nan")
    )
    print(f"CV RMSE: {cv_rmse:.6f}")
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "time_to_approve": np.full(len(test), fallback, dtype=float),
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    raise SystemExit(0)

# 5-fold cross-validation with early stopping.
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []
best_ntrees = []

for fold, (train_idx, val_idx) in enumerate(kf.split(y), start=1):
    X_train_raw = train.loc[train_idx, raw_cols]
    X_val_raw = train.loc[val_idx, raw_cols]
    y_train = y.loc[train_idx]
    y_val = y.loc[val_idx]

    mappings = fit_categorical_mappings(X_train_raw, categorical_cols)

    X_train = preprocess(
        X_train_raw,
        mappings,
        numeric_cols,
        categorical_cols,
        date_cols,
        feature_names,
    )
    X_val = preprocess(
        X_val_raw,
        mappings,
        numeric_cols,
        categorical_cols,
        date_cols,
        feature_names,
    )

    model = make_model(MAX_ROUNDS, EARLY_STOPPING_ROUNDS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_ntree = getattr(model, "best_ntree_limit", None)
    if best_ntree is None:
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None or best_iter < 0:
            best_ntree = MAX_ROUNDS
        else:
            best_ntree = int(best_iter) + 1

    best_ntree = int(np.clip(best_ntree, 1, MAX_ROUNDS))
    best_ntrees.append(best_ntree)

    try:
        pred_val = model.predict(X_val, ntree_limit=best_ntree)
    except TypeError:
        pred_val = model.predict(X_val)

    pred_val = np.clip(pred_val, y.min(), None) if y.min() > 0 else pred_val

    fold_rmse = root_mean_squared_error(y_val, pred_val)
    cv_scores.append(fold_rmse)
    print(f"Fold {fold} RMSE: {fold_rmse:.6f} | best trees: {best_ntree}")

cv_rmse = float(np.mean(cv_scores))
print(f"CV RMSE: {cv_rmse:.6f}")

# Final model on all training data using the average best iteration count.
final_rounds = int(np.clip(np.mean(best_ntrees), 1, MAX_ROUNDS))
print(f"Final number of trees: {final_rounds}")

final_mappings = fit_categorical_mappings(train[raw_cols], categorical_cols)
X_train_final = preprocess(
    train[raw_cols],
    final_mappings,
    numeric_cols,
    categorical_cols,
    date_cols,
    feature_names,
)

final_model = make_model(final_rounds, None)
final_model.fit(X_train_final, y)

# Prepare test features.
X_test_raw = test.reindex(columns=raw_cols)
X_test = preprocess(
    X_test_raw,
    final_mappings,
    numeric_cols,
    categorical_cols,
    date_cols,
    feature_names,
)

test_pred = final_model.predict(X_test)
test_pred = np.clip(test_pred, y.min(), None) if y.min() > 0 else test_pred

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "time_to_approve": test_pred.astype(float),
    }
)
submission.to_csv("./working/submission.csv", index=False)
