import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

TARGET = "Total IFC investment as approved by Board(Million - USD)"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# Clean target and align features.
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
y = train[TARGET].astype(float)
X = train.drop(columns=[TARGET]).copy()
X_test = test.copy()

# Identify categorical columns conservatively.
cat_cols = []
for col in X.columns:
    dtype = X[col].dtype
    if (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_categorical_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    ):
        cat_cols.append(col)

num_cols = [c for c in X.columns if c not in cat_cols]

# Convert numeric columns to numeric, leaving invalid values as NaN for the model.
for col in num_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# Convert categorical columns to strings and replace missing values with a neutral token.
for col in cat_cols:
    X[col] = X[col].astype("object").fillna("missing").astype(str)
    X_test[col] = X_test[col].astype("object").fillna("missing").astype(str)

# Drop features that are entirely missing in both train and test.
drop_cols = [
    col for col in X.columns if X[col].isna().all() and X_test[col].isna().all()
]
X = X.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)
cat_cols = [c for c in cat_cols if c in X.columns]


# Signed log1p transform for skewed nonnegative targets, while remaining safe for negatives.
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def signed_expm1(x):
    return np.sign(x) * np.expm1(np.abs(x))


y_log = signed_log1p(y.values)
min_target = float(y.min())


def inverse_transform(pred_log):
    pred = signed_expm1(np.asarray(pred_log, dtype=float))
    if min_target >= 0:
        pred = np.maximum(pred, 0.0)
    return pred


kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_preds = np.zeros(len(X), dtype=float)
cv_rmse = None
test_pred_log = None

# Prefer CatBoost because it is distinct from the previous LightGBM/XGBoost approaches
# and handles categoricals natively.
use_catboost = False
try:
    from catboost import CatBoostRegressor

    use_catboost = True
except Exception:
    use_catboost = False

if use_catboost:
    try:
        best_iters = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y_log[tr_idx], y_log[va_idx]

            model = CatBoostRegressor(
                iterations=2000,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=42,
                loss_function="RMSE",
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=(X_va, y_va),
                early_stopping_rounds=50,
                verbose=False,
                cat_features=cat_cols if cat_cols else None,
            )

            best_iter = model.get_best_iteration()
            if best_iter is None:
                best_iter = 1999
            best_iters.append(int(best_iter) + 1)

            pred_log = model.predict(X_va)
            cv_preds[va_idx] = inverse_transform(pred_log)
            fold_rmse = root_mean_squared_error(y.iloc[va_idx], cv_preds[va_idx])
            print(f"Fold {fold} RMSE: {fold_rmse:.6f} | best_iter: {best_iters[-1]}")

        cv_rmse = root_mean_squared_error(y, cv_preds)
        final_iters = int(np.median(best_iters))

        final_model = CatBoostRegressor(
            iterations=final_iters,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
        )
        final_model.fit(
            X,
            y_log,
            verbose=False,
            cat_features=cat_cols if cat_cols else None,
        )
        test_pred_log = final_model.predict(X_test)
    except Exception as exc:
        print(
            f"CatBoost failed ({exc}); falling back to HistGradientBoostingRegressor."
        )
        use_catboost = False
        cv_preds = np.zeros(len(X), dtype=float)
        cv_rmse = None
        test_pred_log = None

# Fallback using only scikit-learn.
if not use_catboost:
    from sklearn.ensemble import HistGradientBoostingRegressor

    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y_log[tr_idx], y_log[va_idx]

        if cat_cols:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X_tr[cat_cols] = encoder.fit_transform(X_tr[cat_cols])
            X_va[cat_cols] = encoder.transform(X_va[cat_cols])

        model = HistGradientBoostingRegressor(
            max_iter=2000,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None:
            best_iter = model.n_iter_
        best_iters.append(int(best_iter) + 1)

        pred_log = model.predict(X_va)
        cv_preds[va_idx] = inverse_transform(pred_log)
        fold_rmse = root_mean_squared_error(y.iloc[va_idx], cv_preds[va_idx])
        print(f"Fold {fold} RMSE: {fold_rmse:.6f} | best_iter: {best_iters[-1]}")

    cv_rmse = root_mean_squared_error(y, cv_preds)
    final_iters = int(np.median(best_iters))

    X_full = X.copy()
    X_test_full = X_test.copy()
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_full[cat_cols] = encoder.fit_transform(X_full[cat_cols])
        X_test_full[cat_cols] = encoder.transform(X_test_full[cat_cols])

    final_model = HistGradientBoostingRegressor(
        max_iter=final_iters,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
    final_model.fit(X_full, y_log)
    test_pred_log = final_model.predict(X_test_full)

test_pred = inverse_transform(test_pred_log)
submission = pd.DataFrame({"id": np.arange(len(test), dtype=int), TARGET: test_pred})
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

print(f"CV RMSE: {cv_rmse:.6f}")
