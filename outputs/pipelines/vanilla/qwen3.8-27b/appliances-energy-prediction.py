import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/appliances-energy-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/appliances-energy-prediction/data/test.parquet"
TARGET = "Appliances"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def preprocess(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    date_dt = pd.Series(pd.NaT, index=df.index)

    if "date" in df.columns:
        raw = df["date"].astype(str)
        fixed = raw.str.replace(
            r"^(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})$",
            r"\1 \2",
            regex=True,
        )
        date_dt = pd.to_datetime(fixed, errors="coerce")
        if date_dt.isna().all():
            date_dt = pd.to_datetime(raw, errors="coerce")

        hour = date_dt.dt.hour
        minute = date_dt.dt.minute
        dayofweek = date_dt.dt.dayofweek
        month = date_dt.dt.month
        day = date_dt.dt.day

        df["hour"] = hour.fillna(-1)
        df["minute"] = minute.fillna(-1)
        df["dayofweek"] = dayofweek.fillna(-1)
        df["month"] = month.fillna(-1)
        df["day"] = day.fillna(-1)
        df["is_weekend"] = (dayofweek >= 5).astype(int).fillna(0)

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24).fillna(0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24).fillna(0)
        df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7).fillna(0)
        df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7).fillna(0)

        df = df.drop(columns=["date"])

    t_cols = [f"T{i}" for i in range(1, 10) if f"T{i}" in df.columns]
    rh_cols = [f"RH_{i}" for i in range(1, 10) if f"RH_{i}" in df.columns]

    if t_cols:
        df["T_mean"] = df[t_cols].mean(axis=1)
    if rh_cols:
        df["RH_mean"] = df[rh_cols].mean(axis=1)

    if "T_mean" in df.columns and "T_out" in df.columns:
        df["T_diff_out"] = df["T_mean"] - df["T_out"]
    if "RH_mean" in df.columns and "RH_out" in df.columns:
        df["RH_diff_out"] = df["RH_mean"] - df["RH_out"]

    if "rv1" in df.columns and "rv2" in df.columns:
        df["rv_sum"] = df["rv1"] + df["rv2"]
        df["rv_diff"] = df["rv1"] - df["rv2"]

    return df, date_dt


def to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "cat": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df, train_dates = preprocess(train_df)
    test_df, test_dates = preprocess(test_df)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    feature_cols = [c for c in train_df.columns if c != TARGET]
    train_df = train_df[feature_cols + [TARGET]]
    test_df = test_df.reindex(columns=feature_cols, fill_value=np.nan)

    categorical_features = [
        c
        for c in feature_cols
        if train_df[c].dtype == object or train_df[c].dtype.name == "category"
    ]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    transformers = []
    if numeric_features:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_features)
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    if not transformers:
        transformers = [("passthrough", "passthrough", feature_cols)]

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    X = train_df[feature_cols]
    y = train_df[TARGET]

    use_time_split = False
    if train_dates.notna().sum() > 0 and test_dates.notna().sum() > 0:
        train_valid_dates = train_dates.dropna()
        test_valid_dates = test_dates.dropna()
        if len(train_valid_dates) > 0 and len(test_valid_dates) > 0:
            if test_valid_dates.min() > train_valid_dates.quantile(0.9):
                use_time_split = True

    if use_time_split and len(X) >= 20:
        order = train_dates.sort_values().index
        split_pos = int(len(order) * 0.8)
        train_idx = order[:split_pos]
        val_idx = order[split_pos:]
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=0.2,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]

    preprocessor.fit(X_train)
    X_train_prep = to_dense(preprocessor.transform(X_train))
    X_val_prep = to_dense(preprocessor.transform(X_val))

    models = make_models()
    val_preds = {}
    val_scores = {}

    for name, model in models.items():
        try:
            model.fit(X_train_prep, y_train)
            pred = model.predict(X_val_prep)
            val_preds[name] = pred
            val_scores[name] = rmse(y_val, pred)
        except Exception:
            continue

    if not val_scores:
        raise RuntimeError("No model could be trained on the validation split.")

    weights = {name: 1.0 / max(score, 1e-12) for name, score in val_scores.items()}
    total_weight = sum(weights.values())
    weights = {name: weight / total_weight for name, weight in weights.items()}

    preprocessor.fit(X)
    X_full_prep = to_dense(preprocessor.transform(X))
    X_test_prep = to_dense(preprocessor.transform(test_df))

    final_models = {}
    for name in weights:
        model = make_models()[name]
        model.fit(X_full_prep, y)
        final_models[name] = model

    test_pred = np.zeros(len(test_df), dtype=float)
    for name, weight in weights.items():
        test_pred += weight * final_models[name].predict(X_test_prep)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            TARGET: test_pred,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()