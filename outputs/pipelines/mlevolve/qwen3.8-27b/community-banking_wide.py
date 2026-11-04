import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "LOANTOASSET"
SEED = 42
VAL_FRACTION = 0.2


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def detect_date_column(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        col_lower = str(col).lower()
        if col_lower in {"date", "time", "datetime", "timestamp"}:
            return col
        if "date" in col_lower or "time" in col_lower:
            return col
    return None


def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


class LeakageSafeFeatureProcessor:
    def __init__(self, target: str):
        self.target = target
        self.numeric_cols = []
        self.categorical_cols = []
        self.constant_numeric_cols = []
        self.categories = {}
        self.unseen_tokens = {}

    def fit(self, df: pd.DataFrame) -> "LeakageSafeFeatureProcessor":
        self.numeric_cols = []
        self.categorical_cols = []
        self.constant_numeric_cols = []
        self.categories = {}
        self.unseen_tokens = {}

        for col in df.columns:
            if col == self.target:
                continue

            s = df[col]
            if (
                pd.api.types.is_datetime64_any_dtype(s)
                or pd.api.types.is_numeric_dtype(s)
                or pd.api.types.is_bool_dtype(s)
            ):
                self.numeric_cols.append(col)
            else:
                vals = s.dropna().unique()
                if len(vals) >= 2:
                    self.categorical_cols.append(col)
                    cats = list(vals)
                    token = "__UNSEEN__"
                    counter = 0
                    while token in cats:
                        counter += 1
                        token = f"__UNSEEN___{col}_{counter}"
                    cats.append(token)
                    self.categories[col] = cats
                    self.unseen_tokens[col] = token
                else:
                    self.constant_numeric_cols.append(col)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        for col in df.columns:
            if col == self.target:
                continue

            s = df[col]
            if col in self.numeric_cols:
                if pd.api.types.is_datetime64_any_dtype(s):
                    s = pd.to_datetime(s, errors="coerce")
                    s = s.astype("int64") // 10**9
                    s = s.astype("float64")
                    s = s.mask(s == np.iinfo(np.int64).min, np.nan)
                elif pd.api.types.is_bool_dtype(s):
                    s = s.astype("float64")
                else:
                    s = pd.to_numeric(s, errors="coerce")
                out[col] = s.astype(np.float32)
            elif col in self.categorical_cols:
                s = s.astype("object")
                known = set(self.categories[col])
                token = self.unseen_tokens[col]
                s = s.mask(s.notna() & ~s.isin(known), token)
                out[col] = s.astype(
                    pd.CategoricalDtype(categories=self.categories[col], ordered=False)
                )
            else:
                out[col] = np.where(s.notna(), 0.0, np.nan).astype(np.float32)

        return out


def main():
    set_seed(SEED)
    # Step 1: data processing
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")
    if TARGET not in train_raw.columns:
        raise
    test_id = pd.RangeIndex(len(test_raw))
    test_raw = test_raw.reset_index(drop=True)
    train_raw = (
        train_raw.reset_index(drop=True).dropna(subset=[TARGET]).reset_index(drop=True)
    )
    train_raw[TARGET] = pd.to_numeric(train_raw[TARGET], errors="coerce")
    train_raw = train_raw.dropna(subset=[TARGET]).reset_index(drop=True)
    if len(train_raw) == 0:
        raise

    date_col = detect_date_column(train_raw)
    if date_col is not None:
        tmp = train_raw.copy()
        tmp["_split_date"] = parse_date_series(tmp[date_col])
        tmp["_orig_idx"] = tmp.index
        tmp = tmp.sort_values(["_split_date", "_orig_idx"], kind="mergesort")
        tmp = tmp.drop(columns=["_orig_idx"])
        n_val = max(1, int(round(len(tmp) * VAL_FRACTION)))
        if n_val >= len(tmp):
            n_val = max(1, len(tmp) - 1)
        val_raw = tmp.iloc[-n_val:].drop(columns=["_split_date"]).reset_index(drop=True)
        train_split = (
            tmp.iloc[:-n_val].drop(columns=["_split_date"]).reset_index(drop=True)
        )
        split_method = "temporal_last_20pct"
    else:
        try:
            strata = pd.qcut(train_raw[TARGET], q=5, duplicates="drop")
            train_split, val_raw = train_test_split(
                train_raw, test_size=VAL_FRACTION, random_state=SEED, stratify=strata
            )
        except Exception:
            train_split, val_raw = train_test_split(
                train_raw, test_size=VAL_FRACTION, random_state=SEED
            )
        train_split = train_split.reset_index(drop=True)
        val_raw = val_raw.reset_index(drop=True)
        split_method = "random_stratified_80_20"

    processor = LeakageSafeFeatureProcessor(target=TARGET).fit(train_split)  # fixed
    X_train = processor.transform(train_split)
    y_train = train_split[TARGET].to_numpy(dtype=float)
    X_val = processor.transform(val_raw)
    y_val = val_raw[TARGET].to_numpy(dtype=float)
    X_test = processor.transform(test_raw)

    # feature mask
    valid_feature_mask = [X_train[c].notna().any() for c in X_train.columns]
    feature_names = [c for c, keep in zip(X_train.columns, valid_feature_mask) if keep]
    if len(feature_names) == 0:
        feature_names = list(X_train.columns)
    X_train = X_train[feature_names].copy()
    X_val = X_val[feature_names].copy()
    X_test = X_test[feature_names].copy()
    categorical_feature = [c for c in feature_names if c in processor.categorical_cols]
    for c in feature_names:
        if c not in categorical_feature:
            X_train[c] = X_train[c].astype(np.float32)
            X_val[c] = X_val[c].astype(np.float32)
            X_test[c] = X_test[c].astype(np.float32)

    # save artifacts
    train_out = X_train.copy()
    train_out[TARGET] = y_train
    train_out.to_parquet("./working/train_processed.parquet")
    val_out = X_val.copy()
    val_out[TARGET] = y_val
    val_out.to_parquet("./working/val_processed.parquet")
    test_out = X_test.copy()
    test_out.insert(0, "test_id", test_id.to_numpy())
    test_out.to_parquet("./working/test_processed.parquet")
    metadata = {
        "split_method": split_method,
        "n_train": len(train_split),
        "n_val": len(val_raw),
        "n_test": len(test_raw),
        "n_features": len(feature_names),
        "categorical_feature": categorical_feature,
        "seed": SEED,
    }
    with open("./working/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open("./working/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    import lightgbm as lgb
    import xgboost as xgb

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature)
    dval = lgb.Dataset(
        X_val, label=y_val, reference=dtrain, categorical_feature=categorical_feature
    )
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None or best_iter <= 0:
        best_iter = model.num_boosters()

    val_pred_lgb = model.predict(X_val, num_iteration=int(best_iter))

    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        n_estimators=2000,
        eval_metric="rmse",
        random_state=SEED,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=0,
    )

    xgb_best_round = None
    try:
        xgb_evals = xgb_model.evals_result()
        for key in xgb_evals:
            if "rmse" in xgb_evals[key]:
                xgb_best_round = int(np.argmin(xgb_evals[key]["rmse"])) + 1
                break
    except Exception:
        xgb_best_round = None
    if xgb_best_round is None or xgb_best_round <= 0:
        xgb_best_round = getattr(xgb_model, "best_ntree_limit", None)
    if xgb_best_round is None or xgb_best_round <= 0:
        xgb_best_round = getattr(xgb_model, "best_iteration", None)
        if xgb_best_round is not None and xgb_best_round >= 0:
            xgb_best_round = int(xgb_best_round) + 1
    if xgb_best_round is None or xgb_best_round <= 0:
        xgb_best_round = int(xgb_model.n_estimators)

    try:
        val_pred_xgb = xgb_model.predict(
            X_val, iteration_range=(0, int(xgb_best_round))
        )
    except TypeError:
        val_pred_xgb = xgb_model.predict(X_val)

    val_pred = (val_pred_lgb + val_pred_xgb) / 2.0
    final_val_score = rmse(y_val, val_pred)

    X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_full = np.concatenate([y_train, y_val]).astype(float)
    dtrain_full = lgb.Dataset(X_full, label=y_full, categorical_feature=categorical_feature)
    full_model = lgb.train(params, dtrain_full, num_boost_round=int(max(1, best_iter)))

    full_xgb = xgb.XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        n_estimators=int(max(1, xgb_best_round)),
        eval_metric="rmse",
        random_state=SEED,
    )
    full_xgb.fit(X_full, y_full, verbose=0)

    test_pred_lgb = full_model.predict(X_test)
    test_pred_xgb = full_xgb.predict(X_test)
    test_pred = (test_pred_lgb + test_pred_xgb) / 2.0
    os.makedirs("./submission", exist_ok=True)
    submission = pd.DataFrame({"id": test_id, "LOANTOASSET": test_pred})
    submission.to_csv("./submission/submission.csv", index=False)
    print(f"Final Validation Score: {final_val_score:.6f}")


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


if __name__ == "__main__":
    main()