import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target = "SALE_PRC"
id_col = "PARCELNO"

# Clean target: keep finite positive sale prices.
train[target] = pd.to_numeric(train[target], errors="coerce")
train = train.dropna(subset=[target]).copy()
train = train[np.isfinite(train[target]) & (train[target] > 0)].copy()

# Use only feature columns that exist in both train and test, excluding id and target.
feature_cols = [
    c for c in train.columns if c not in [id_col, target] and c in test.columns
]

X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

# Force numeric values and replace infinities with NaN.
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

X = X.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

# Drop columns that are entirely missing in both train and test.
keep_cols = [c for c in X.columns if X[c].notna().any() or X_test[c].notna().any()]
X = X[keep_cols]
X_test = X_test[keep_cols]

X = X.astype(np.float32)
X_test = X_test.astype(np.float32)

# Model log1p(price), but evaluate in original price space.
y = np.log1p(train[target].astype(float).to_numpy())
y_original = train[target].astype(float).to_numpy()
max_price = float(np.nanmax(y_original)) if len(y_original) else 0.0


def to_dollar(log_predictions: np.ndarray) -> np.ndarray:
    predictions = np.expm1(log_predictions)
    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=max_price,
        neginf=0.0,
    )
    return np.clip(predictions, 0.0, None)


xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

# Bug fix: LightGBM requires "regression" rather than "reg_l2".
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
    "num_threads": -1,
    "verbosity": -1,
}

max_rounds_xgb = int(xgb_params["n_estimators"])
max_rounds_lgb = 5000

# Group by PARCELNO when possible to avoid repeated parcels leaking across folds.
if id_col in train.columns and train[id_col].nunique() >= 5:
    groups = train[id_col].astype("string")
    if groups.isna().any():
        groups = groups.fillna([f"row_{i}" for i in range(len(groups))])
    groups = groups.to_numpy()
    splitter = GroupKFold(n_splits=5)
    splits = splitter.split(X, y, groups=groups)
else:
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = splitter.split(X, y)

fold_rmses = []
xgb_best_iters = []
lgb_best_iters = []

for tr_idx, va_idx in splits:
    X_tr = X.iloc[tr_idx]
    X_va = X.iloc[va_idx]
    y_tr = y[tr_idx]
    y_va = y[va_idx]
    y_orig_va = y_original[va_idx]

    # XGBoost fold model.
    xgb_model = xgb.XGBRegressor(early_stopping_rounds=50, **xgb_params)
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    best_iter_xgb = getattr(xgb_model, "best_iteration", None)
    if best_iter_xgb is None:
        xgb_best_iters.append(max_rounds_xgb)
        xgb_pred_log = xgb_model.predict(X_va)
    else:
        n_iter_xgb = int(best_iter_xgb) + 1
        xgb_best_iters.append(n_iter_xgb)
        try:
            xgb_pred_log = xgb_model.predict(X_va, iteration_range=(0, n_iter_xgb))
        except TypeError:
            xgb_pred_log = xgb_model.predict(X_va)

    xgb_pred = to_dollar(xgb_pred_log)

    # LightGBM fold model.
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=max_rounds_lgb,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(-1),
        ],
    )

    best_iter_lgb = lgb_model.best_iteration
    if best_iter_lgb is None or best_iter_lgb <= 0:
        lgb_pred_log = lgb_model.predict(X_va)
        lgb_best_iters.append(max_rounds_lgb)
    else:
        lgb_pred_log = lgb_model.predict(X_va, num_iteration=int(best_iter_lgb))
        lgb_best_iters.append(int(best_iter_lgb))

    lgb_pred = to_dollar(lgb_pred_log)

    # Ensemble in dollar space, matching the competition RMSE metric.
    ensemble_pred = 0.5 * (xgb_pred + lgb_pred)
    fold_rmses.append(np.sqrt(mean_squared_error(y_orig_va, ensemble_pred)))

mean_rmse = float(np.mean(fold_rmses))
print(f"Validation RMSE (original price space): {mean_rmse:.2f}")

# Refit on the full training set using average early-stopped depths.
best_n_xgb = max(1, min(max_rounds_xgb, int(round(np.mean(xgb_best_iters)))))
best_n_lgb = max(1, min(max_rounds_lgb, int(round(np.mean(lgb_best_iters)))))

final_xgb_params = {k: v for k, v in xgb_params.items() if k != "n_estimators"}
final_xgb = xgb.XGBRegressor(n_estimators=best_n_xgb, **final_xgb_params)
final_xgb.fit(X, y)
xgb_test_pred = to_dollar(final_xgb.predict(X_test))

final_lgb = lgb.Dataset(X, label=y)
final_lgb_model = lgb.train(
    lgb_params,
    final_lgb,
    num_boost_round=best_n_lgb,
    callbacks=[lgb.log_evaluation(-1)],
)
lgb_test_pred = to_dollar(final_lgb_model.predict(X_test))

test_pred = 0.5 * (xgb_test_pred + lgb_test_pred)
test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=max_price, neginf=0.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "SALE_PRC": test_pred.astype(float),
    }
)
submission.to_csv("./working/submission.csv", index=False)
