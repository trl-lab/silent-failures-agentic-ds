import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error

TARGET = "Amount in USD"
SEED = 42
os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Remove explicit id columns if present.
id_cols = [c for c in train.columns if str(c).lower() == "id"]
train = train.drop(columns=id_cols, errors="ignore")
test = test.drop(columns=id_cols, errors="ignore")

# Keep only rows with finite target values.
target_numeric = pd.to_numeric(train[TARGET], errors="coerce")
valid = target_numeric.notna() & np.isfinite(target_numeric.to_numpy())
train = train.loc[valid].copy()
target_numeric = target_numeric.loc[valid]

feature_cols = [c for c in train.columns if c != TARGET]

# Fallback for degenerate data.
if len(train) < 2 or len(feature_cols) == 0:
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "Amount in USD": 0.0,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("CV RMSE: nan")
    raise SystemExit

# Align test features to train features.
test = test.reindex(columns=feature_cols)

X = train[feature_cols].copy()
y = target_numeric.astype(float).to_numpy()

# Identify numeric and categorical features from the training data only.
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in numeric_cols]

# Make categorical columns object dtype so missing values can be filled with a string.
for c in cat_cols:
    X[c] = X[c].astype(object)
    if c in test.columns:
        test[c] = test[c].astype(object)

# Coerce numeric columns and remove non-finite values.
for c in numeric_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    if c in test.columns:
        test[c] = pd.to_numeric(test[c], errors="coerce")

# Use a log1p target transform only when the target is non-negative.
use_log = bool((y.min() >= 0) and (y.max() > 0))
y_model = np.log1p(y) if use_log else y.copy()


def to_original(pred):
    pred = np.asarray(pred, dtype=float)
    if use_log:
        pred = np.expm1(pred)
        pred = np.maximum(pred, 0.0)
    return pred


def make_original_rmse_eval(y_va_orig):
    def eval_fn(preds, dataset):
        pred_orig = to_original(preds)
        true_orig = np.asarray(y_va_orig, dtype=float)

        pred_orig = np.nan_to_num(pred_orig, nan=0.0, posinf=0.0, neginf=0.0)
        true_orig = np.nan_to_num(true_orig, nan=0.0, posinf=0.0, neginf=0.0)

        rmse = root_mean_squared_error(true_orig, pred_orig)
        return ("orig_rmse", float(rmse), False)

    return eval_fn


def make_preprocessor():
    transformers = []

    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if cat_cols:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        transformers.append(("num", SimpleImputer(strategy="median"), feature_cols))

    return ColumnTransformer(transformers, remainder="drop")


def ensure_nonempty(Xm):
    if hasattr(Xm, "shape") and len(Xm.shape) == 2 and Xm.shape[1] == 0:
        return np.ones((Xm.shape[0], 1), dtype=np.float32)
    return Xm


params = {
    "objective": "regression",
    "metric": "None",
    "learning_rate": 0.05,
    "num_leaves": max(4, min(15, len(X) // 10)),
    "max_depth": -1,
    "min_child_samples": max(1, min(10, len(X) // 10)),
    "feature_fraction": 1.0 if len(feature_cols) < 10 else 0.8,
    "bagging_fraction": 1.0 if len(X) < 50 else 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbose": -1,
    "num_threads": -1,
}

cv_scores = []
best_iters = []

# Use 5-fold CV when the dataset is large enough; otherwise use a single hold-out.
if len(X) >= 25:
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
else:
    n_train = max(1, int(0.8 * len(X)))
    if n_train >= len(X):
        n_train = len(X) - 1
    folds = [(np.arange(n_train), np.arange(n_train, len(X)))]

for tr_idx, va_idx in folds:
    if len(va_idx) == 0:
        continue

    X_tr = X.iloc[tr_idx]
    y_tr = y_model[tr_idx]
    X_va = X.iloc[va_idx]
    y_va = y_model[va_idx]
    y_va_orig = y[va_idx]

    pre = make_preprocessor()
    X_tr_t = ensure_nonempty(pre.fit_transform(X_tr))
    X_va_t = ensure_nonempty(pre.transform(X_va))

    train_ds = lgb.Dataset(X_tr_t, label=y_tr)
    valid_ds = lgb.Dataset(X_va_t, label=y_va, reference=train_ds)

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=2000,
        valid_sets=[valid_ds],
        valid_names=["valid"],
        feval=make_original_rmse_eval(y_va_orig),
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    best_iter = max(1, int(getattr(model, "best_iteration", 0) or 0))
    pred = model.predict(X_va_t, num_iteration=best_iter)
    pred_orig = to_original(pred)

    true_orig = y_va_orig
    cv_scores.append(root_mean_squared_error(true_orig, pred_orig))
    best_iters.append(best_iter)

if cv_scores:
    cv_rmse = float(np.mean(cv_scores))
    print(f"CV RMSE: {cv_rmse:.6f}")
else:
    print("CV RMSE: nan")

# If there is no test data, still write a valid empty submission.
if len(test) == 0:
    submission = pd.DataFrame(
        {
            "id": np.array([], dtype=int),
            "Amount in USD": np.array([], dtype=float),
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    raise SystemExit

# Train the final model on all available training data.
pre = make_preprocessor()
X_tr_t = ensure_nonempty(pre.fit_transform(X))
X_te_t = ensure_nonempty(pre.transform(test))

final_ds = lgb.Dataset(X_tr_t, label=y_model)
final_model = lgb.train(
    params,
    final_ds,
    num_boost_round=max(1, int(np.mean(best_iters))) if best_iters else 100,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_te_t)
test_pred = to_original(test_pred)

test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "Amount in USD": test_pred.astype(float),
    }
)
submission.to_csv("./working/submission.csv", index=False)
