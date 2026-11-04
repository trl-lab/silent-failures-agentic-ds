import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

os.makedirs("./working", exist_ok=True)

TARGET = "gisacres"
SEED = 42
N_SPLITS = 5

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
valid_target_mask = train[TARGET].notna() & np.isfinite(train[TARGET])
train = train[valid_target_mask].copy()

id_like = [
    c for c in train.columns if c.lower() in {"id", "index", "row_id", "index_id"}
]

X = train.drop(columns=[TARGET] + id_like, errors="ignore").copy()
y = train[TARGET].astype(float).reset_index(drop=True)

X_test = test.drop(columns=[TARGET] + id_like, errors="ignore").copy()

X = X.loc[:, ~X.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]
X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)


def datetime_to_numeric(s):
    s = pd.to_datetime(s, errors="coerce")
    try:
        out = s.astype("int64") / 1e9
        out = out.mask(s.isna(), np.nan)
        return out.astype(float)
    except Exception:
        out = (
            s.dt.year.astype(float) * 10000
            + s.dt.month.astype(float) * 100
            + s.dt.day.astype(float)
        )
        return out


def calendar_features(series, prefix):
    dt = pd.to_datetime(series, errors="coerce")
    feats = {
        f"{prefix}_month_sin": np.sin(2.0 * np.pi * dt.dt.month / 12.0),
        f"{prefix}_month_cos": np.cos(2.0 * np.pi * dt.dt.month / 12.0),
        f"{prefix}_doy_sin": np.sin(2.0 * np.pi * dt.dt.dayofyear / 365.25),
        f"{prefix}_doy_cos": np.cos(2.0 * np.pi * dt.dt.dayofyear / 365.25),
        f"{prefix}_dow_sin": np.sin(2.0 * np.pi * dt.dt.dayofweek / 7.0),
        f"{prefix}_dow_cos": np.cos(2.0 * np.pi * dt.dt.dayofweek / 7.0),
        f"{prefix}_hour_sin": np.sin(2.0 * np.pi * dt.dt.hour / 24.0),
        f"{prefix}_hour_cos": np.cos(2.0 * np.pi * dt.dt.hour / 24.0),
    }
    return pd.DataFrame(feats, index=series.index)


def preprocess_features(X_tr, X_te):
    categorical_cols = []
    extra_tr = pd.DataFrame(index=X_tr.index)
    extra_te = pd.DataFrame(index=X_te.index)

    for col in X_tr.columns:
        if pd.api.types.is_bool_dtype(X_tr[col]):
            X_tr[col] = X_tr[col].astype(int)
            X_te[col] = X_te[col].astype(int)
            continue

        if pd.api.types.is_numeric_dtype(X_tr[col]):
            X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce").astype(float)
            X_te[col] = pd.to_numeric(X_te[col], errors="coerce").astype(float)
            continue

        if pd.api.types.is_datetime64_any_dtype(X_tr[col]):
            cal_tr = calendar_features(X_tr[col], col)
            cal_te = calendar_features(X_te[col], col)
            for c in cal_tr.columns:
                extra_tr[c] = cal_tr[c]
                extra_te[c] = cal_te[c]

            X_tr[col] = datetime_to_numeric(X_tr[col])
            X_te[col] = datetime_to_numeric(X_te[col])
            continue

        num_tr = pd.to_numeric(X_tr[col], errors="coerce")
        non_null_tr = X_tr[col].notna().sum()
        if non_null_tr > 0 and num_tr.notna().sum() >= 0.8 * non_null_tr:
            X_tr[col] = num_tr.astype(float)
            X_te[col] = pd.to_numeric(X_te[col], errors="coerce").astype(float)
            continue

        sample = X_tr[col].dropna().head(200)
        if len(sample) > 0:
            parsed_sample = pd.to_datetime(sample, errors="coerce")
            if parsed_sample.notna().mean() >= 0.8:
                cal_tr = calendar_features(X_tr[col], col)
                cal_te = calendar_features(X_te[col], col)
                for c in cal_tr.columns:
                    extra_tr[c] = cal_tr[c]
                    extra_te[c] = cal_te[c]

                X_tr[col] = datetime_to_numeric(X_tr[col])
                X_te[col] = datetime_to_numeric(X_te[col])
                continue

        if isinstance(X_tr[col].dtype, pd.CategoricalDtype):
            cats = X_tr[col].cat.categories
        else:
            cats = X_tr[col].astype("category").cat.categories

        X_tr[col] = X_tr[col].astype("category")
        X_te[col] = pd.Categorical(X_te[col].astype(object), categories=cats).astype(
            "category"
        )
        categorical_cols.append(col)

    if not extra_tr.empty:
        X_tr = pd.concat([X_tr, extra_tr], axis=1)
        X_te = pd.concat([X_te, extra_te], axis=1)

    drop_cols = []
    for col in X_tr.columns:
        if X_tr[col].isna().all():
            drop_cols.append(col)
        elif X_tr[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)

    X_tr = X_tr.drop(columns=drop_cols, errors="ignore")
    X_te = X_te.drop(columns=drop_cols, errors="ignore")

    categorical_cols = [c for c in categorical_cols if c in X_tr.columns]

    return X_tr, X_te, categorical_cols


X, X_test, categorical_cols = preprocess_features(X, X_test)

use_log = bool((y >= 0).all())
y_target = np.log1p(y) if use_log else y.values


def inverse_transform(pred):
    if use_log:
        pred = np.expm1(pred)
        return np.clip(pred, 0.0, None)
    return pred


params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "seed": SEED,
    "verbose": -1,
    "num_threads": -1,
}

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
best_iterations = []
num_boost_round = 5000

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y_target[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y_target[val_idx]

    dtrain = lgb.Dataset(
        X_train_fold,
        label=y_train_fold,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_val_fold,
        label=y_val_fold,
        categorical_feature=categorical_cols,
        reference=dtrain,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    best_iter = getattr(model, "best_iteration", 0)
    if best_iter <= 0:
        best_iter = num_boost_round
    best_iterations.append(int(best_iter))

    val_pred = model.predict(X_val_fold, num_iteration=best_iter)
    oof_preds[val_idx] = val_pred

oof_preds = inverse_transform(oof_preds)
cv_rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
print(f"5-Fold CV RMSE: {cv_rmse:.6f}")

final_rounds = int(np.median(best_iterations)) if best_iterations else num_boost_round
final_rounds = max(1, final_rounds)

dtrain_full = lgb.Dataset(
    X,
    label=y_target,
    categorical_feature=categorical_cols,
    free_raw_data=False,
)

final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_preds = final_model.predict(X_test)
test_preds = inverse_transform(test_preds)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_preds), dtype=int),
        "gisacres": test_preds,
    }
)

submission.to_csv("./working/submission.csv", index=False)
print(f"Saved submission to ./working/submission.csv with {len(submission)} rows.")
