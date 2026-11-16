import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/avito-demand-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/avito-demand-prediction/data/test.parquet"
TARGET = "deal_probability"
RANDOM_STATE = 42

CATEGORICAL_COLS = [
    "region",
    "city",
    "parent_category_name",
    "category_name",
    "param_1",
    "param_2",
    "param_3",
    "user_type",
]

DROP_COLS = [
    "item_id",
    "user_id",
    "image",
    "title",
    "description",
    "activation_date",
]


def clean_and_encode_categoricals(train, test):
    train = train.copy()
    test = test.copy()

    for col in CATEGORICAL_COLS:
        if col not in train.columns:
            train[col] = "missing"
        if col not in test.columns:
            test[col] = "missing"

        train[col] = train[col].fillna("missing").astype(str)
        test[col] = test[col].fillna("missing").astype(str)

        freq = train[col].value_counts(dropna=False)
        train[f"{col}_freq"] = train[col].map(freq).astype(float)
        test[f"{col}_freq"] = test[col].map(freq).fillna(0).astype(float)

        train_cats = set(train[col].unique())
        if not train_cats:
            default = "missing"
        else:
            default = "missing" if "missing" in train_cats else train[col].mode().iloc[0]

        train[f"{col}_unseen"] = 0.0
        test[f"{col}_unseen"] = (~test[col].isin(train_cats)).astype(float)
        test[col] = test[col].where(test[col].isin(train_cats), default)

        categories = sorted(train_cats)
        if default not in categories:
            categories.append(default)

        train[col] = pd.Categorical(train[col], categories=categories)
        test[col] = pd.Categorical(test[col], categories=categories)

    return train, test


def add_date_features(train, test):
    feature_cols = [
        "activation_year",
        "activation_month",
        "activation_day",
        "activation_dayofweek",
        "activation_is_weekend",
        "activation_days_since_min",
    ]

    if "activation_date" not in train.columns or "activation_date" not in test.columns:
        for col in feature_cols:
            train[col] = np.nan
            test[col] = np.nan
        return train, test

    train_date = pd.to_datetime(train["activation_date"], errors="coerce")
    test_date = pd.to_datetime(test["activation_date"], errors="coerce")
    min_date = train_date.min()

    train["activation_year"] = train_date.dt.year
    train["activation_month"] = train_date.dt.month
    train["activation_day"] = train_date.dt.day
    train["activation_dayofweek"] = train_date.dt.dayofweek
    train["activation_is_weekend"] = (train_date.dt.dayofweek >= 5).astype(float)

    test["activation_year"] = test_date.dt.year
    test["activation_month"] = test_date.dt.month
    test["activation_day"] = test_date.dt.day
    test["activation_dayofweek"] = test_date.dt.dayofweek
    test["activation_is_weekend"] = (test_date.dt.dayofweek >= 5).astype(float)

    if pd.notna(min_date):
        train["activation_days_since_min"] = (train_date - min_date).dt.days
        test["activation_days_since_min"] = (test_date - min_date).dt.days
    else:
        train["activation_days_since_min"] = np.nan
        test["activation_days_since_min"] = np.nan

    return train, test


def add_numeric_features(df):
    if "price" in df.columns:
        price = df["price"].astype(float)
        df["price_missing"] = price.isna().astype(int)
        df["price_log"] = np.log1p(price.clip(lower=0))
        df["price_is_zero"] = (price == 0).astype(int)

    if "item_seq_number" in df.columns:
        seq = df["item_seq_number"].astype(float)
        df["item_seq_number_log"] = np.log1p(seq.clip(lower=0))

    if "image_top_1" in df.columns:
        image_top = df["image_top_1"].astype(float)
        df["image_top_1_missing"] = image_top.isna().astype(int)
        df["image_top_1_log"] = np.log1p(image_top.clip(lower=0))

    return df


def add_text_features(df):
    if "title" in df.columns:
        title = df["title"].fillna("").astype(str)
    else:
        title = pd.Series("", index=df.index)

    if "description" in df.columns:
        description = df["description"].fillna("").astype(str)
    else:
        description = pd.Series("", index=df.index)

    title_len = title.str.len()
    desc_len = description.str.len()

    df["title_len"] = title_len
    df["desc_len"] = desc_len
    df["text_len"] = title_len + desc_len

    title_words = title.str.count(" ") + 1
    desc_words = description.str.count(" ") + 1
    title_words = title_words.where(title_len > 0, 0)
    desc_words = desc_words.where(desc_len > 0, 0)

    df["title_words"] = title_words
    df["desc_words"] = desc_words
    df["text_words"] = title_words + desc_words

    df["title_len_log"] = np.log1p(title_len)
    df["desc_len_log"] = np.log1p(desc_len)
    df["desc_to_title_len_ratio"] = desc_len / title_len.replace(0, np.nan)
    df["desc_to_title_word_ratio"] = desc_words / title_words.replace(0, np.nan)

    return df


def sanitize_numeric(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def prepare_features(train, test):
    train = train.copy()
    test = test.copy()

    train, test = clean_and_encode_categoricals(train, test)
    train, test = add_date_features(train, test)

    train = add_numeric_features(train)
    test = add_numeric_features(test)

    train = add_text_features(train)
    test = add_text_features(test)

    drop_cols = [c for c in DROP_COLS if c in train.columns]
    train = train.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    if TARGET in train.columns:
        train = train.drop(columns=[TARGET], errors="ignore")
    if TARGET in test.columns:
        test = test.drop(columns=[TARGET], errors="ignore")

    for col in train.columns:
        if train[col].dtype == object:
            train[col] = train[col].fillna("missing").astype(str)
            if col not in test.columns:
                test[col] = "missing"
            else:
                test[col] = test[col].fillna("missing").astype(str)

            train_cats = set(train[col].unique())
            if not train_cats:
                default = "missing"
            else:
                default = "missing" if "missing" in train_cats else train[col].mode().iloc[0]

            test[col] = test[col].where(test[col].isin(train_cats), default)

            categories = sorted(train_cats)
            if default not in categories:
                categories.append(default)

            train[col] = pd.Categorical(train[col], categories=categories)
            test[col] = pd.Categorical(test[col], categories=categories)

    for col in train.columns:
        if col not in test.columns:
            test[col] = np.nan

    test = test.reindex(columns=train.columns)

    for col in train.columns:
        if isinstance(train[col].dtype, pd.CategoricalDtype):
            if test[col].isna().any():
                default = train[col].cat.categories[0]
                test[col] = pd.Categorical(
                    test[col].fillna(default),
                    categories=train[col].cat.categories,
                )

    train = sanitize_numeric(train)
    test = sanitize_numeric(test)

    return train, test


def make_model(n_estimators, use_early_stopping=False):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": int(n_estimators),
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    if use_early_stopping:
        try:
            return LGBMRegressor(**params, early_stopping_rounds=100)
        except TypeError:
            return LGBMRegressor(**params)

    return LGBMRegressor(**params)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y = train_df[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    valid = y.notna()
    y = y[valid]
    X = train_df.loc[valid].drop(columns=[TARGET], errors="ignore")

    X, X_test = prepare_features(X, test_df)

    if len(X_test) == 0:
        pd.DataFrame({"id": [], "deal_probability": []}).to_csv("predictions.csv", index=False)
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=RANDOM_STATE,
    )

    model = make_model(1500, use_early_stopping=True)
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
        )
        best_iter = getattr(model, "best_iteration_", 0)
    except Exception:
        model = make_model(1500, use_early_stopping=False)
        model.fit(X_train, y_train)
        best_iter = 1500

    if best_iter is None or best_iter <= 0:
        best_iter = getattr(model, "n_estimators", 1500)

    final_model = make_model(max(1, int(best_iter)), use_early_stopping=False)
    final_model.fit(X, y)

    preds = final_model.predict(X_test)
    preds = np.clip(preds, 0.0, 1.0)

    output = pd.DataFrame(
        {
            "id": np.arange(len(preds), dtype=int),
            "deal_probability": preds.astype(float),
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()