import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb

TARGET = "salary_in_usd"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
WORK_DIR = "./working"
SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.csv")

os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# Clean target and drop rows without target.
train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

# Keep only columns present in the training data, excluding target.
feature_cols = [c for c in train.columns if c != TARGET and train[c].notna().any()]

# Identify feature types.
cat_cols = []
num_cols = []
date_cols = []

for col in feature_cols:
    series = train[col]
    if pd.api.types.is_datetime64_any_dtype(series):
        date_cols.append(col)
    elif pd.api.types.is_numeric_dtype(series):
        num_cols.append(col)
    else:
        cat_cols.append(col)


def datetime_to_days(series: pd.Series) -> pd.Series:
    """Convert datetime-like values to days since 2000-01-01."""
    series = pd.to_datetime(series, errors="coerce")
    return (series - pd.Timestamp("2000-01-01")).dt.days


def preprocess(
    X: pd.DataFrame,
    cat_cols: list,
    num_cols: list,
    date_cols: list,
    feature_cols: list,
    encoders=None,
    fit: bool = False,
):
    """
    Convert features to a stable numeric DataFrame.

    If fit=True, fit ordinal encoders on categorical columns.
    If fit=False, transform using supplied encoders.
    """
    X = X.copy()

    # Convert datetime columns to numeric days.
    for col in date_cols:
        if col in X.columns:
            X[col] = datetime_to_days(X[col])
        else:
            X[col] = np.nan

    # Prepare categorical columns.
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("object").fillna("missing").astype(str)
        else:
            X[col] = "missing"

    if fit:
        encoders = {}
        for col in cat_cols:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoder.fit(X[[col]])
            encoders[col] = encoder
    else:
        if encoders is None:
            raise ValueError("encoders must be provided when fit=False")

    # Encode categoricals.
    if cat_cols:
        encoded_parts = []
        for col in cat_cols:
            encoder = encoders[col]
            encoded_parts.append(encoder.transform(X[[col]]))
        X[cat_cols] = np.column_stack(encoded_parts)

    # Convert numeric columns to float, preserving NaNs.
    for col in num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = np.nan

    # Ensure a stable feature order and fill missing columns with NaN.
    out = pd.DataFrame(index=X.index)
    for col in feature_cols:
        out[col] = X[col] if col in X.columns else np.nan

    return out, encoders


# Log-transform target for modelling and evaluation.
y = np.log1p(train[TARGET].to_numpy(dtype=float))

# 5-fold cross-validation with inner early stopping.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.full(len(train), np.nan, dtype=float)
best_iterations = []

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.01,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "seed": 42,
    "tree_method": "hist",
}

for fold, (train_idx, val_idx) in enumerate(kf.split(train), start=1):
    X_train_fold = train.iloc[train_idx]
    X_val_fold = train.iloc[val_idx]

    X_train_proc, encoders = preprocess(
        X_train_fold,
        cat_cols,
        num_cols,
        date_cols,
        feature_cols,
        fit=True,
    )
    X_val_proc, _ = preprocess(
        X_val_fold,
        cat_cols,
        num_cols,
        date_cols,
        feature_cols,
        encoders=encoders,
        fit=False,
    )

    y_train_fold = y[train_idx]
    y_val_fold = y[val_idx]

    # Inner split only for early stopping; outer validation fold is untouched.
    inner_train_idx, inner_val_idx = train_test_split(
        np.arange(len(X_train_proc)),
        test_size=0.2,
        random_state=42,
    )

    dtrain_inner = xgb.DMatrix(
        X_train_proc.iloc[inner_train_idx],
        label=y_train_fold[inner_train_idx],
    )
    dval_inner = xgb.DMatrix(
        X_train_proc.iloc[inner_val_idx],
        label=y_train_fold[inner_val_idx],
    )
    dval_outer = xgb.DMatrix(X_val_proc, label=y_val_fold)

    model = xgb.train(
        params,
        dtrain_inner,
        num_boost_round=8000,
        evals=[(dval_inner, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    best_iter = int(model.best_iteration)
    best_iterations.append(best_iter + 1)

    pred_val = model.predict(dval_outer, iteration_range=(0, best_iter + 1))
    oof_pred[val_idx] = pred_val

    fold_rmse = float(np.sqrt(mean_squared_error(y_val_fold, pred_val)))
    print(f"Fold {fold}: log1p RMSE = {fold_rmse:.6f}, best_iteration = {best_iter}")

cv_rmse_log = float(np.sqrt(mean_squared_error(y, oof_pred)))
print(f"CV RMSE (log1p salary): {cv_rmse_log:.6f}")

# Final model on full training data.
final_rounds = int(np.median(best_iterations)) if best_iterations else 500
X_train_full_proc, final_encoders = preprocess(
    train,
    cat_cols,
    num_cols,
    date_cols,
    feature_cols,
    fit=True,
)

dtrain_full = xgb.DMatrix(X_train_full_proc, label=y)
final_model = xgb.train(
    params, dtrain_full, num_boost_round=final_rounds, verbose_eval=False
)

# Predict test set.
X_test_proc, _ = preprocess(
    test,
    cat_cols,
    num_cols,
    date_cols,
    feature_cols,
    encoders=final_encoders,
    fit=False,
)

dtest = xgb.DMatrix(X_test_proc)
test_pred_log = final_model.predict(dtest)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "salary_in_usd": test_pred,
    }
)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission saved to {SUBMISSION_PATH}")
