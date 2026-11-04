import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "State Rate"
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def prepare_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()

    if target in df.columns:
        df = df.drop(columns=[target])

    id_like = {"id", "index", "row_id", "rowid", "index_id", "uid", "identifier"}
    df = df.drop(
        columns=[c for c in df.columns if str(c).strip().lower() in id_like],
        errors="ignore",
    )

    df = df.loc[:, ~df.columns.duplicated()]

    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(np.int64) // 10**9
            df[col] = df[col].replace(np.iinfo(np.int64).min, np.nan)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("category")

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(axis=1, how="all")
    return df


def parse_target(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        series = (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    return pd.to_numeric(series, errors="coerce")


def make_params(seed: int) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
    }


train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

if TARGET not in train.columns:
    raise ValueError(f"Target column {TARGET!r} not found in train data.")

y = parse_target(train[TARGET]).replace([np.inf, -np.inf], np.nan)
mask = y.notna()
y = y[mask].astype(float)
train = train.loc[mask]
test = test.drop(columns=[TARGET], errors="ignore")

X_train = prepare_features(train, TARGET)
X_test = prepare_features(test, TARGET)

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

drop_cols = []
for col in common_cols:
    train_is_cat = X_train[col].dtype.name == "category"
    test_is_cat = X_test[col].dtype.name == "category"

    if train_is_cat != test_is_cat:
        drop_cols.append(col)
    elif train_is_cat:
        if len(X_train[col].cat.categories) == 0:
            drop_cols.append(col)
        else:
            X_test[col] = X_test[col].cat.set_categories(X_train[col].cat.categories)

X_train = X_train.drop(columns=drop_cols, errors="ignore")
X_test = X_test.drop(columns=drop_cols, errors="ignore")

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

cat_cols = [c for c in common_cols if X_train[c].dtype.name == "category"]


def save_fallback(value: float) -> None:
    submission = pd.DataFrame(
        {"id": np.arange(len(X_test)), TARGET: np.full(len(X_test), value)}
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)


if len(y) == 0 or X_train.shape[1] == 0:
    save_fallback(0.0)
    print("CV RMSE: 0.000000")
    raise SystemExit(0)

if y.nunique() <= 1:
    save_fallback(float(y.mean()))
    print("CV RMSE: 0.000000")
    raise SystemExit(0)

seeds = [42, 123, 456]
n_splits = 5 if len(y) >= 5 else max(2, len(y)) if len(y) >= 2 else 1
cv_preds = np.zeros(len(y), dtype=float)
best_iters_by_seed = {seed: [] for seed in seeds}

if n_splits >= 2:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(X_train):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y.iloc[va_idx]

        fold_preds = np.zeros(len(va_idx), dtype=float)

        for seed in seeds:
            params = make_params(seed)

            train_ds = lgb.Dataset(
                X_tr,
                label=y_tr,
                categorical_feature=cat_cols,
                free_raw_data=False,
            )
            valid_ds = lgb.Dataset(
                X_va,
                label=y_va,
                reference=train_ds,
                categorical_feature=cat_cols,
                free_raw_data=False,
            )

            model = lgb.train(
                params,
                train_ds,
                num_boost_round=5000,
                valid_sets=[valid_ds],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )

            best_iter = (
                model.best_iteration
                if model.best_iteration and model.best_iteration > 0
                else 1
            )
            best_iters_by_seed[seed].append(best_iter)

            pred_va = model.predict(
                X_va,
                num_iteration=(
                    model.best_iteration if model.best_iteration > 0 else None
                ),
            )
            fold_preds += pred_va

        fold_preds /= len(seeds)
        cv_preds[va_idx] = fold_preds

    cv_preds = np.nan_to_num(
        cv_preds,
        nan=float(y.mean()),
        posinf=float(np.max(y)),
        neginf=float(np.min(y)),
    )
    cv_rmse = float(np.sqrt(mean_squared_error(y, cv_preds)))
else:
    train_ds = lgb.Dataset(
        X_train,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    pred_tr = np.zeros(len(y), dtype=float)
    for seed in seeds:
        model = lgb.train(make_params(seed), train_ds, num_boost_round=100)
        pred_tr += model.predict(X_train)
        best_iters_by_seed[seed].append(100)
    pred_tr /= len(seeds)
    pred_tr = np.nan_to_num(
        pred_tr,
        nan=float(y.mean()),
        posinf=float(np.max(y)),
        neginf=float(np.min(y)),
    )
    cv_rmse = float(np.sqrt(mean_squared_error(y, pred_tr)))

print(f"CV RMSE: {cv_rmse:.6f}")

if len(X_test) == 0:
    pd.DataFrame({"id": np.arange(0), TARGET: np.array([], dtype=float)}).to_csv(
        os.path.join(WORK_DIR, "submission.csv"), index=False
    )
    raise SystemExit(0)

full_ds = lgb.Dataset(
    X_train,
    label=y,
    categorical_feature=cat_cols,
    free_raw_data=False,
)

test_pred = np.zeros(len(X_test), dtype=float)
for seed in seeds:
    best_rounds = int(np.clip(np.mean(best_iters_by_seed[seed]), 1, 5000))
    model = lgb.train(make_params(seed), full_ds, num_boost_round=best_rounds)
    test_pred += model.predict(X_test)

test_pred /= len(seeds)
test_pred = np.nan_to_num(
    test_pred,
    nan=float(y.mean()),
    posinf=float(np.max(y)),
    neginf=float(np.min(y)),
)

submission = pd.DataFrame({"id": np.arange(len(X_test)), TARGET: test_pred})
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
