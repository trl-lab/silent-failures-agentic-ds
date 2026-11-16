import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ida-statement-credits-grants-guarantees/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ida-statement-credits-grants-guarantees/data/test.parquet"
TARGET = "Original Principal Amount (US$)"

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

train = train.dropna(subset=[TARGET]).reset_index(drop=True)

if train.empty:
    predictions = np.zeros(len(test), dtype=float)
else:
    features = [c for c in train.columns if c != TARGET and c in test.columns]
    features = [c for c in features if train[c].notna().any()]

    X_train = train[features].copy()
    X_test = test[features].copy()

    date_like = [
        c for c in features
        if "date" in c.lower() or "period" in c.lower()
    ]

    for col in date_like:
        parsed = pd.to_datetime(X_train[col], errors="coerce")
        if parsed.notna().sum() > 0:
            try:
                X_train[col] = (parsed - pd.Timestamp("1900-01-01")).dt.days
            except Exception:
                X_train[col] = parsed.astype("int64", errors="ignore") // 10**9

        parsed_test = pd.to_datetime(X_test[col], errors="coerce")
        if parsed_test.notna().sum() > 0:
            try:
                X_test[col] = (parsed_test - pd.Timestamp("1900-01-01")).dt.days
            except Exception:
                X_test[col] = parsed_test.astype("int64", errors="ignore") // 10**9

    keep = []
    for col in features:
        if X_train[col].nunique(dropna=True) > 1:
            keep.append(col)
    features = keep

    if not features:
        predictions = np.full(len(test), float(train[TARGET].mean()), dtype=float)
    else:
        X_train = X_train[features]
        X_test = X_test[features]
        y = train[TARGET].astype(float)

        numeric_cols = [c for c in features if X_train[c].dtype.kind in "iuf"]
        cat_cols = [c for c in features if c not in numeric_cols]

        for col in numeric_cols:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        for col in cat_cols:
            categories = X_train[col].dropna().unique()
            try:
                categories = sorted(categories)
            except TypeError:
                categories = list(categories)

            X_train[col] = pd.Categorical(X_train[col], categories=categories)
            X_test[col] = pd.Categorical(X_test[col], categories=categories)

        lgb_params = dict(
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        if len(X_train) > 20:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y, test_size=0.2, random_state=42
            )

            model = LGBMRegressor(**lgb_params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                categorical_feature=cat_cols,
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )

            best_iter = getattr(model, "best_iteration_", None)
            if best_iter is None or best_iter <= 0:
                best_iter = model.n_estimators

            final_model = LGBMRegressor(**{**lgb_params, "n_estimators": int(best_iter)})
            final_model.fit(X_train, y, categorical_feature=cat_cols)
            predictions = final_model.predict(X_test)
        else:
            model = LGBMRegressor(**{**lgb_params, "n_estimators": 500})
            model.fit(X_train, y, categorical_feature=cat_cols)
            predictions = model.predict(X_test)

        fallback = float(y.mean())
        predictions = np.asarray(predictions, dtype=float)
        predictions = np.where(np.isfinite(predictions), predictions, fallback)

pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: predictions,
    }
).to_csv("predictions.csv", index=False)