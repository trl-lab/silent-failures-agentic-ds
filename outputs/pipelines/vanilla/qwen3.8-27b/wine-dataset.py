import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/wine-dataset/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/wine-dataset/data/test.parquet"
TARGET = "Price"
RANDOM_STATE = 42


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_") or "feature"


def get_series(df: pd.DataFrame, col: str, default: str = "missing") -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )


def extract_abv(series: pd.Series) -> pd.Series:
    def _extract(value):
        if pd.isna(value):
            return np.nan
        match = re.search(r"(\d+(?:\.\d+)?)", str(value))
        return float(match.group(1)) if match else np.nan

    return series.map(_extract).astype(float)


def extract_vintage(series: pd.Series) -> pd.Series:
    def _extract(value):
        if pd.isna(value):
            return np.nan
        match = re.search(r"\b(19\d{2}|20\d{2})\b", str(value))
        return int(match.group(1)) if match else np.nan

    return series.map(_extract).astype(float)


def extract_capacity(series: pd.Series) -> pd.Series:
    def _extract(value):
        if pd.isna(value):
            return np.nan
        text = str(value).upper()
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(ML|CL|L|BTL|BOTTLE|BOTTLES|CASE|DOZEN|JERBOYA|IMPERIAL|SALUTE|NECK|PINT|QUART|GALLON)?",
            text,
        )
        if not match:
            return np.nan

        value_num = float(match.group(1))
        unit = match.group(2)

        if unit == "ML":
            return value_num
        if unit == "CL":
            return value_num * 10.0
        if unit == "L":
            return value_num * 1000.0
        if unit in {"BTL", "BOTTLE", "BOTTLES"}:
            return value_num * 750.0
        if unit in {"CASE", "DOZEN"}:
            return value_num * 750.0 * 12.0
        if unit == "JERBOYA":
            return value_num * 5000.0
        if unit == "IMPERIAL":
            return value_num * 4500.0
        if unit == "SALUTE":
            return value_num * 3000.0
        if unit == "NECK":
            return value_num * 1500.0
        if unit == "PINT":
            return value_num * 568.0
        if unit == "QUART":
            return value_num * 946.0
        if unit == "GALLON":
            return value_num * 3785.0

        return value_num

    return series.map(_extract).astype(float)


def count_list_items(series: pd.Series) -> pd.Series:
    def _count(value):
        if pd.isna(value):
            return 0
        text = str(value).strip()
        if text.lower() in {"", "none", "nan", "null"}:
            return 0
        return len([part for part in text.split(",") if part.strip()])

    return series.map(_count).astype(int)


def add_tfidf_features(
    train_series: pd.Series,
    test_series: pd.Series,
    train_dict: dict,
    test_dict: dict,
    prefix: str,
    max_features: int = 120,
) -> None:
    min_df = 2 if len(train_series) > 50 else 1
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 1),
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )

    try:
        train_matrix = vectorizer.fit_transform(train_series)
        test_matrix = vectorizer.transform(test_series)
    except ValueError:
        return

    if train_matrix.shape[1] == 0:
        return

    for i in range(train_matrix.shape[1]):
        train_dict[f"{prefix}_{i}"] = train_matrix.getcol(i).toarray().ravel()
        test_dict[f"{prefix}_{i}"] = test_matrix.getcol(i).toarray().ravel()


def fit_categorical_mapping(train_series: pd.Series, top_n: int = 250) -> dict:
    counts = train_series.value_counts(dropna=False)
    top_categories = counts.head(top_n).index.tolist()
    return {category: idx + 1 for idx, category in enumerate(top_categories)}


def transform_categorical(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).fillna(0).astype(int)


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    score = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if not np.isfinite(score):
        return 1.0
    return score


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column {TARGET} not found in train data.")

    y_train = train_df[TARGET].astype(float)
    valid = y_train.notna()
    train_df = train_df.loc[valid].copy()
    y_train = y_train.loc[valid].copy()

    X_train = train_df.drop(columns=[TARGET]).copy()

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])
    X_test = test_df.copy()

    train_dict = {}
    test_dict = {}
    categorical_feature_names = []

    text_columns = [
        col
        for col in ["Title", "Description", "Characteristics", "Secondary Grape Varieties"]
        if col in X_train.columns
    ]

    for col in text_columns:
        prefix = safe_name(col)

        train_series = get_series(X_train, col)
        test_series = get_series(X_test, col)

        train_text = clean_text_series(train_series)
        test_text = clean_text_series(test_series)

        train_dict[f"{prefix}_length"] = train_text.str.len().astype(float).to_numpy()
        test_dict[f"{prefix}_length"] = test_text.str.len().astype(float).to_numpy()

        train_dict[f"{prefix}_word_count"] = (
            train_text.str.split().str.len().fillna(0).astype(float).to_numpy()
        )
        test_dict[f"{prefix}_word_count"] = (
            test_text.str.split().str.len().fillna(0).astype(float).to_numpy()
        )

        if col in {"Characteristics", "Secondary Grape Varieties"}:
            train_dict[f"{prefix}_item_count"] = count_list_items(train_series).astype(float).to_numpy()
            test_dict[f"{prefix}_item_count"] = count_list_items(test_series).astype(float).to_numpy()

        max_features = 150 if col in {"Title", "Description"} else 100
        add_tfidf_features(
            train_series=train_text,
            test_series=test_text,
            train_dict=train_dict,
            test_dict=test_dict,
            prefix=prefix,
            max_features=max_features,
        )

    numeric_source_columns = [
        col
        for col in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[col]) and col != TARGET
    ]

    for col in numeric_source_columns:
        train_dict[col] = get_series(X_train, col).astype(float).to_numpy()
        test_dict[col] = get_series(X_test, col).astype(float).to_numpy()

    if "ABV" in X_train.columns:
        train_dict["ABV_num"] = extract_abv(get_series(X_train, "ABV")).to_numpy()
        test_dict["ABV_num"] = extract_abv(get_series(X_test, "ABV")).to_numpy()

    if "Vintage" in X_train.columns:
        train_dict["Vintage_num"] = extract_vintage(get_series(X_train, "Vintage")).to_numpy()
        test_dict["Vintage_num"] = extract_vintage(get_series(X_test, "Vintage")).to_numpy()

    if "Capacity" in X_train.columns:
        train_dict["Capacity_num"] = extract_capacity(get_series(X_train, "Capacity")).to_numpy()
        test_dict["Capacity_num"] = extract_capacity(get_series(X_test, "Capacity")).to_numpy()

    excluded_categorical = set(text_columns) | {"ABV", "Vintage"}
    categorical_columns = [
        col
        for col in X_train.columns
        if col not in excluded_categorical
        and (X_train[col].dtype == object or str(X_train[col].dtype) == "category")
    ]

    for col in categorical_columns:
        feature_name = safe_name(col)

        train_cat = get_series(X_train, col).fillna("missing").astype(str)
        test_cat = get_series(X_test, col).fillna("missing").astype(str)

        mapping = fit_categorical_mapping(train_cat, top_n=250)

        train_dict[f"{feature_name}_cat"] = transform_categorical(train_cat, mapping).to_numpy()
        test_dict[f"{feature_name}_cat"] = transform_categorical(test_cat, mapping).to_numpy()

        freq = train_cat.value_counts(normalize=True)
        train_dict[f"{feature_name}_freq"] = train_cat.map(freq).fillna(0.0).astype(float).to_numpy()
        test_dict[f"{feature_name}_freq"] = test_cat.map(freq).fillna(0.0).astype(float).to_numpy()

        categorical_feature_names.append(f"{feature_name}_cat")

    if not train_dict:
        train_dict["constant"] = np.ones(len(X_train), dtype=float)
        test_dict["constant"] = np.ones(len(X_test), dtype=float)

    train_features = pd.DataFrame(train_dict, index=X_train.index)
    test_features = pd.DataFrame(test_dict, index=X_test.index)

    for col in train_features.columns:
        if col not in test_features.columns:
            test_features[col] = 0.0
    test_features = test_features[train_features.columns]

    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    test_features = test_features.replace([np.inf, -np.inf], np.nan)

    for col in train_features.columns:
        if col in categorical_feature_names:
            train_features[col] = train_features[col].astype(int)
            test_features[col] = test_features[col].astype(int)
        else:
            train_features[col] = train_features[col].astype(np.float32)
            test_features[col] = test_features[col].astype(np.float32)

    use_validation = len(train_features) >= 200

    if use_validation:
        X_fit, X_val, y_fit, y_val = train_test_split(
            train_features,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    else:
        X_fit, X_val, y_fit, y_val = train_features, train_features, y_train, y_train

    lgbm_params = dict(
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    xgb_params = dict(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )

    cat_params = dict(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
    )

    lgbm_fit_kwargs = {}
    if categorical_feature_names:
        lgbm_fit_kwargs["categorical_feature"] = categorical_feature_names

    cat_fit_kwargs = {}
    if categorical_feature_names:
        cat_fit_kwargs["cat_features"] = categorical_feature_names

    if use_validation:
        lgbm_model = LGBMRegressor(**lgbm_params)
        lgbm_model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            **lgbm_fit_kwargs,
        )

        xgb_model = XGBRegressor(early_stopping_rounds=100, **xgb_params)
        xgb_model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_fit,
            y_fit,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=0,
            **cat_fit_kwargs,
        )

        lgbm_val_pred = lgbm_model.predict(X_val)
        xgb_val_pred = xgb_model.predict(X_val)
        cat_val_pred = cat_model.predict(X_val)

        lgbm_rmse = rmse_score(y_val.to_numpy(), lgbm_val_pred)
        xgb_rmse = rmse_score(y_val.to_numpy(), xgb_val_pred)
        cat_rmse = rmse_score(y_val.to_numpy(), cat_val_pred)

        inv_lgbm = 1.0 / max(lgbm_rmse, 1e-6)
        inv_xgb = 1.0 / max(xgb_rmse, 1e-6)
        inv_cat = 1.0 / max(cat_rmse, 1e-6)
        weight_sum = inv_lgbm + inv_xgb + inv_cat
        lgbm_weight = inv_lgbm / weight_sum
        xgb_weight = inv_xgb / weight_sum
        cat_weight = inv_cat / weight_sum
    else:
        lgbm_model = LGBMRegressor(**lgbm_params)
        lgbm_model.fit(X_fit, y_fit, **lgbm_fit_kwargs)

        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(X_fit, y_fit)

        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_fit, y_fit, verbose=0, **cat_fit_kwargs)

        lgbm_weight = xgb_weight = cat_weight = 1.0 / 3.0

    lgbm_best = getattr(lgbm_model, "best_iteration_", None)
    if lgbm_best is None or lgbm_best <= 0:
        lgbm_best = lgbm_params["n_estimators"]
    lgbm_final = LGBMRegressor(**{**lgbm_params, "n_estimators": int(lgbm_best)})
    lgbm_final.fit(train_features, y_train, **lgbm_fit_kwargs)

    xgb_best = getattr(xgb_model, "best_iteration", None)
    if xgb_best is None or xgb_best < 0:
        xgb_best_rounds = xgb_params["n_estimators"]
    else:
        xgb_best_rounds = int(xgb_best) + 1
    xgb_final = XGBRegressor(**{**xgb_params, "n_estimators": int(xgb_best_rounds)})
    xgb_final.fit(train_features, y_train)

    try:
        cat_best = cat_model.get_best_iteration()
    except Exception:
        cat_best = -1
    if cat_best is None or cat_best < 0:
        cat_best_rounds = cat_params["iterations"]
    else:
        cat_best_rounds = int(cat_best) + 1
    cat_final = CatBoostRegressor(**{**cat_params, "iterations": int(cat_best_rounds)})
    cat_final.fit(train_features, y_train, verbose=0, **cat_fit_kwargs)

    lgbm_pred = lgbm_final.predict(test_features)
    xgb_pred = xgb_final.predict(test_features)
    cat_pred = cat_final.predict(test_features)

    final_pred = (
        lgbm_weight * np.asarray(lgbm_pred, dtype=float)
        + xgb_weight * np.asarray(xgb_pred, dtype=float)
        + cat_weight * np.asarray(cat_pred, dtype=float)
    )

    final_pred = np.nan_to_num(
        final_pred,
        nan=float(y_train.median()),
        posinf=float(y_train.max()),
        neginf=float(y_train.min()),
    )
    final_pred = np.clip(final_pred, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(final_pred), dtype=int),
            "Price": final_pred,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()