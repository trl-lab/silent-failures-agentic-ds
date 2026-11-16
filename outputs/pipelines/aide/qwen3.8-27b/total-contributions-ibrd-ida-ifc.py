import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

os.makedirs("./working", exist_ok=True)
TARGET = "Total Contribution (USD)"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = pd.to_numeric(train[TARGET], errors="coerce").astype(float)
train = train.drop(columns=[TARGET])

id_like = {"id", "index", "idx", "row_id", "rowid"}


def drop_id_columns(df):
    cols = [
        c
        for c in df.columns
        if str(c).strip().lower() in id_like
        or str(c).strip().lower().endswith("_id")
        or str(c).strip().lower().startswith("unnamed")
    ]
    return df.drop(columns=cols, errors="ignore")


train = drop_id_columns(train)
test = drop_id_columns(test)

common = [c for c in train.columns if c in test.columns]
train = train[common]
test = test[common]

for c in train.columns:
    if pd.api.types.is_numeric_dtype(train[c]):
        train[c] = pd.to_numeric(train[c], errors="coerce").astype(float)
        test[c] = pd.to_numeric(test[c], errors="coerce").astype(float)
        train[c] = train[c].replace([np.inf, -np.inf], np.nan)
        test[c] = test[c].replace([np.inf, -np.inf], np.nan)
    else:
        train[c] = train[c].astype("string")
        test[c] = test[c].astype("string")

finite_target = np.isfinite(y.to_numpy())
y = y[finite_target].reset_index(drop=True)
train = train[finite_target].reset_index(drop=True)

drop_cols = []
for c in train.columns:
    s = train[c]
    if s.isna().all():
        drop_cols.append(c)
    elif s.nunique(dropna=True) <= 1:
        drop_cols.append(c)
    elif s.dtype.name == "string" and s.nunique(dropna=True) > max(
        50, int(0.5 * len(train))
    ):
        drop_cols.append(c)

train = train.drop(columns=drop_cols, errors="ignore")
test = test.drop(columns=drop_cols, errors="ignore")
feature_cols = [c for c in train.columns if c in test.columns]

if len(y) == 0:
    test_pred = np.full(len(test), 0.0, dtype=float)
    print("No finite target rows; using fallback predictions.")
else:
    string_cols = [c for c in feature_cols if train[c].dtype.name == "string"]
    if string_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_str = train[string_cols].fillna("missing")
        X_test_str = test[string_cols].fillna("missing")
        encoder.fit(X_train_str)
        train[string_cols] = encoder.transform(X_train_str)
        test[string_cols] = encoder.transform(X_test_str)

    for c in feature_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce").astype(float)
        test[c] = pd.to_numeric(test[c], errors="coerce").astype(float)
        train[c] = train[c].replace([np.inf, -np.inf], np.nan)
        test[c] = test[c].replace([np.inf, -np.inf], np.nan)

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    X_test.columns = X.columns
    fallback = float(np.nanmean(y))

    if X.shape[1] == 0 or not X.notna().any().any():
        oof = np.full(len(y), fallback)
        test_pred = np.full(len(test), fallback)
        print(f"5-fold CV RMSE: {root_mean_squared_error(y, oof):.6f}")
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(X), dtype=float)
        test_pred = np.zeros(len(X_test), dtype=float)

        def predict_best(model, X_pred):
            try:
                best_it = int(model.best_iteration)
                if best_it < 0:
                    return model.predict(X_pred)
                return model.predict(X_pred, iteration_range=(0, best_it + 1))
            except (TypeError, AttributeError):
                return model.predict(X_pred)

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=42,
                eval_metric="rmse",
                early_stopping_rounds=50,
                n_jobs=-1,
            )
            model.fit(
                X.iloc[tr_idx],
                y.iloc[tr_idx],
                eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                verbose=False,
            )

            val_pred = predict_best(model, X.iloc[va_idx])
            oof[va_idx] = val_pred
            if X_test.shape[0] > 0:
                test_pred += predict_best(model, X_test) / 5.0

            fold_rmse = root_mean_squared_error(y.iloc[va_idx], val_pred)
            print(f"Fold {fold} validation RMSE: {fold_rmse:.6f}")

        oof = np.nan_to_num(oof, nan=fallback, posinf=fallback, neginf=fallback)
        test_pred = np.nan_to_num(
            test_pred, nan=fallback, posinf=fallback, neginf=fallback
        )

        if y.min() >= 0:
            oof = np.clip(oof, 0.0, None)
            test_pred = np.clip(test_pred, 0.0, None)

        cv_rmse = root_mean_squared_error(y, oof)
        print(f"5-fold CV RMSE: {cv_rmse:.6f}")

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
