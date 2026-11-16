import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/food-prices_wide/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/food-prices_wide/data/test.parquet"
TARGET = "usdprice"
RANDOM_STATE = 42


class FeaturePreprocessor:
    def __init__(self):
        self.numeric_cols = []
        self.cat_cols = []
        self.categories = {}

    def fit(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        self.numeric_cols = []
        for c in numeric_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            col = col.replace([np.inf, -np.inf], np.nan)
            if col.notna().any() and col.nunique(dropna=True) > 1:
                self.numeric_cols.append(c)

        self.cat_cols = []
        self.categories = {}
        for c in df.columns:
            if c in numeric_cols:
                continue
            col = df[c].fillna("missing").astype(str)
            if col.nunique(dropna=True) > 1:
                self.cat_cols.append(c)
                cats = list(pd.unique(col))
                if "missing" not in cats:
                    cats.append("missing")
                self.categories[c] = sorted(cats, key=lambda x: str(x))

        return self

    def transform(self, df, dtype="category"):
        out = df.copy()

        for c in self.numeric_cols:
            if c not in out.columns:
                out[c] = np.nan
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)

        for c in self.cat_cols:
            if c not in out.columns:
                out[c] = "missing"
            s = out[c].fillna("missing").astype(str)
            cats = set(self.categories[c])
            s = s.where(s.isin(cats), "missing")
            if dtype == "category":
                out[c] = pd.Categorical(s, categories=self.categories[c])
            else:
                out[c] = s

        return out[self.numeric_cols + self.cat_cols].copy()


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fill = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
    y_pred = np.where(np.isfinite(y_pred), y_pred, fill)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_predictions(pred, test_df, fill_value=0.0):
    pred = np.asarray(pred, dtype=float).reshape(-1)

    if not np.isfinite(fill_value):
        fill_value = 0.0

    if len(pred) != len(test_df):
        if len(pred) == 0:
            pred = np.full(len(test_df), fill_value)
        elif len(pred) < len(test_df):
            pred = np.concatenate([pred, np.full(len(test_df) - len(pred), fill_value)])
        else:
            pred = pred[:len(test_df)]

    pred = np.where(np.isfinite(pred), pred, fill_value)

    pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "usdprice": pred
    }).to_csv("predictions.csv", index=False)


def lgbm_params(n_train):
    min_child = max(1, min(20, n_train // 10))
    return {
        "n_estimators": 1500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": min_child,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    }


def cat_params():
    return {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "allow_writing_files": False
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        save_predictions(np.zeros(len(test_df)), test_df, 0.0)
        return

    y_train = train_df[TARGET].astype(float)
    X_train = train_df.drop(columns=[TARGET])

    valid = y_train.notna() & np.isfinite(y_train)
    X_train = X_train.loc[valid]
    y_train = y_train.loc[valid]

    if len(y_train) == 0:
        save_predictions(np.zeros(len(test_df)), test_df, 0.0)
        return

    mean_y = float(y_train.mean())

    if len(y_train) < 20:
        pre = FeaturePreprocessor().fit(X_train)
        if not (pre.numeric_cols or pre.cat_cols):
            save_predictions(np.full(len(test_df), mean_y), test_df, mean_y)
            return

        X_full_lgbm = pre.transform(X_train, dtype="category")
        X_test_lgbm = pre.transform(test_df, dtype="category")

        try:
            model = LGBMRegressor(**lgbm_params(len(X_full_lgbm)))
            model.fit(X_full_lgbm, y_train, categorical_feature=pre.cat_cols)
            pred = model.predict(X_test_lgbm)
        except Exception:
            pred = np.full(len(test_df), mean_y)

        save_predictions(pred, test_df, mean_y)
        return

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    pre_val = FeaturePreprocessor().fit(X_tr)
    if not (pre_val.numeric_cols or pre_val.cat_cols):
        save_predictions(np.full(len(test_df), mean_y), test_df, mean_y)
        return

    X_tr_lgbm = pre_val.transform(X_tr, dtype="category")
    X_val_lgbm = pre_val.transform(X_val, dtype="category")
    X_tr_cat = pre_val.transform(X_tr, dtype="object")
    X_val_cat = pre_val.transform(X_val, dtype="object")

    val_preds = {}
    best_iters = {}
    ensemble_weights = None

    params_lgbm = lgbm_params(len(X_tr_lgbm))
    try:
        model = LGBMRegressor(**params_lgbm)
        model.fit(
            X_tr_lgbm,
            y_tr,
            eval_set=[(X_val_lgbm, y_val)],
            eval_metric="rmse",
            categorical_feature=pre_val.cat_cols,
            callbacks=[early_stopping(stopping_rounds=100, verbose=False)]
        )
        val_preds["lgbm"] = model.predict(X_val_lgbm)
        best_iters["lgbm"] = max(
            1,
            int(getattr(model, "best_iteration_", params_lgbm["n_estimators"]) or params_lgbm["n_estimators"])
        )
    except Exception:
        pass

    params_cat = cat_params()
    try:
        model = CatBoostRegressor(**params_cat)
        model.fit(
            X_tr_cat,
            y_tr,
            eval_set=(X_val_cat, y_val),
            early_stopping_rounds=50,
            use_best_model=True,
            cat_features=pre_val.cat_cols
        )
        val_preds["cat"] = model.predict(X_val_cat)
        best_iter = model.get_best_iteration()
        if best_iter is None or best_iter < 0:
            best_iter = params_cat["iterations"] - 1
        best_iters["cat"] = max(1, int(best_iter) + 1)
    except Exception:
        pass

    if not val_preds:
        save_predictions(np.full(len(test_df), mean_y), test_df, mean_y)
        return

    scores = {name: rmse(y_val, pred) for name, pred in val_preds.items()}
    finite_scores = {k: v for k, v in scores.items() if np.isfinite(v)}

    if "lgbm" in val_preds and "cat" in val_preds:
        s_l = scores.get("lgbm", np.inf)
        s_c = scores.get("cat", np.inf)
        if np.isfinite(s_l) and np.isfinite(s_c):
            w_l = 1.0 / max(s_l, 1e-8)
            w_c = 1.0 / max(s_c, 1e-8)
            total = w_l + w_c
            if total > 0:
                w_l /= total
                w_c /= total
                ensemble_pred = (
                    w_l * np.asarray(val_preds["lgbm"], dtype=float) +
                    w_c * np.asarray(val_preds["cat"], dtype=float)
                )
                val_preds["ensemble"] = ensemble_pred
                scores["ensemble"] = rmse(y_val, ensemble_pred)
                finite_scores["ensemble"] = scores["ensemble"]
                ensemble_weights = {"lgbm": w_l, "cat": w_c}

    if finite_scores:
        chosen = min(finite_scores, key=finite_scores.get)
    else:
        chosen = next(iter(val_preds))

    pre_final = FeaturePreprocessor().fit(X_train)
    if not (pre_final.numeric_cols or pre_final.cat_cols):
        save_predictions(np.full(len(test_df), mean_y), test_df, mean_y)
        return

    X_full_lgbm = pre_final.transform(X_train, dtype="category")
    X_test_lgbm = pre_final.transform(test_df, dtype="category")
    X_full_cat = pre_final.transform(X_train, dtype="object")
    X_test_cat = pre_final.transform(test_df, dtype="object")

    final_preds = {}

    if "lgbm" in val_preds:
        try:
            final_params = lgbm_params(len(X_full_lgbm))
            final_params["n_estimators"] = best_iters.get("lgbm", final_params["n_estimators"])
            model = LGBMRegressor(**final_params)
            model.fit(X_full_lgbm, y_train, categorical_feature=pre_final.cat_cols)
            final_preds["lgbm"] = model.predict(X_test_lgbm)
        except Exception:
            pass

    if "cat" in val_preds:
        try:
            final_params = cat_params()
            final_params["iterations"] = best_iters.get("cat", final_params["iterations"])
            model = CatBoostRegressor(**final_params)
            model.fit(X_full_cat, y_train, cat_features=pre_final.cat_cols)
            final_preds["cat"] = model.predict(X_test_cat)
        except Exception:
            pass

    if not final_preds:
        save_predictions(np.full(len(test_df), mean_y), test_df, mean_y)
        return

    if chosen == "ensemble" and "lgbm" in final_preds and "cat" in final_preds:
        w_l = ensemble_weights["lgbm"] if ensemble_weights else 0.5
        w_c = ensemble_weights["cat"] if ensemble_weights else 0.5
        pred = (
            w_l * np.asarray(final_preds["lgbm"], dtype=float) +
            w_c * np.asarray(final_preds["cat"], dtype=float)
        )
    elif chosen in final_preds:
        pred = final_preds[chosen]
    else:
        pred = final_preds[list(final_preds.keys())[0]]

    save_predictions(pred, test_df, mean_y)


if __name__ == "__main__":
    main()