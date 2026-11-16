import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/superconductivity/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/superconductivity/data/test.parquet"
TARGET = "critical_temp"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_preprocessor(numeric_cols, categorical_cols):
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        return "passthrough"
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model(name):
    if name == "lgbm" and LGBMRegressor is not None:
        return LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgb" and XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    if name == "cat" and CatBoostRegressor is not None:
        return CatBoostRegressor(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        )
    return None


def to_str_keep_nan(series):
    def convert(x):
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        return str(x)

    return series.apply(convert)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        pd.DataFrame(
            {"id": np.arange(len(test_df), dtype=int), "critical_temp": 0.0}
        ).to_csv("predictions.csv", index=False)
        return

    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.drop(columns=[TARGET], errors="ignore")

    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].astype(float)

    finite_mask = np.isfinite(y)
    X = X.loc[finite_mask]
    y = y.loc[finite_mask]

    mean_target = float(y.mean()) if len(y) > 0 else 0.0

    def save_predictions(preds):
        preds = np.asarray(preds, dtype=float).reshape(-1)
        if preds.size != len(test_df):
            if preds.size < len(test_df):
                preds = np.concatenate(
                    [preds, np.full(len(test_df) - preds.size, mean_target)]
                )
            else:
                preds = preds[: len(test_df)]
        preds = np.nan_to_num(
            preds, nan=mean_target, posinf=mean_target, neginf=mean_target
        )
        pd.DataFrame(
            {"id": np.arange(len(preds), dtype=int), "critical_temp": preds}
        ).to_csv("predictions.csv", index=False)

    if len(test_df) == 0:
        save_predictions(np.array([], dtype=float))
        return

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            converted = pd.to_numeric(X[col], errors="coerce")
            non_null = X[col].notna().sum()
            if non_null > 0 and converted.notna().sum() >= 0.5 * non_null:
                X[col] = converted

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    X = X.loc[:, X.notna().any()]
    test_df = test_df.reindex(columns=X.columns)

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(test_df[col]):
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    for col in categorical_cols:
        X[col] = to_str_keep_nan(X[col])
        test_df[col] = to_str_keep_nan(test_df[col])

    if X.shape[1] == 0 or len(X) < 2:
        save_predictions(np.full(len(test_df), mean_target))
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model_names = []
    if LGBMRegressor is not None:
        model_names.append("lgbm")
    if XGBRegressor is not None:
        model_names.append("xgb")
    if CatBoostRegressor is not None:
        model_names.append("cat")

    val_scores = {}
    for name in model_names:
        try:
            model = make_model(name)
            if model is None:
                continue
            preprocessor = make_preprocessor(numeric_cols, categorical_cols)
            pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            val_pred = pipe.predict(X_val)
            score = rmse(y_val, val_pred)
            if np.isfinite(score):
                val_scores[name] = score
        except Exception:
            pass

    if not val_scores:
        save_predictions(np.full(len(test_df), mean_target))
        return

    inv_weights = {name: 1.0 / max(score, 1e-8) for name, score in val_scores.items()}
    total_inv = sum(inv_weights.values())
    weights = {name: w / total_inv for name, w in inv_weights.items()}

    final_preds = []
    final_weights = []
    for name, weight in weights.items():
        try:
            model = make_model(name)
            if model is None:
                continue
            preprocessor = make_preprocessor(numeric_cols, categorical_cols)
            pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X, y)
            pred = pipe.predict(test_df)
            pred = np.asarray(pred, dtype=float).reshape(-1)
            if pred.size != len(test_df):
                if pred.size < len(test_df):
                    pred = np.concatenate(
                        [pred, np.full(len(test_df) - pred.size, mean_target)]
                    )
                else:
                    pred = pred[: len(test_df)]
            pred = np.nan_to_num(
                pred, nan=mean_target, posinf=mean_target, neginf=mean_target
            )
            final_preds.append(pred)
            final_weights.append(weight)
        except Exception:
            pass

    if not final_preds:
        save_predictions(np.full(len(test_df), mean_target))
        return

    if len(final_preds) == 1:
        preds = final_preds[0]
    else:
        preds = np.average(np.vstack(final_preds), axis=0, weights=final_weights)

    save_predictions(preds)


if __name__ == "__main__":
    main()