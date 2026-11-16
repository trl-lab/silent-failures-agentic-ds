import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/electricity/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/electricity/data/test.parquet"
TARGET = "class"
RANDOM_STATE = 42


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def add_features(df):
    df = df.copy()

    if "period" in df.columns:
        df["period_sin"] = np.sin(2.0 * np.pi * df["period"])
        df["period_cos"] = np.cos(2.0 * np.pi * df["period"])

    if "date" in df.columns:
        df["date_sin"] = np.sin(2.0 * np.pi * df["date"])
        df["date_cos"] = np.cos(2.0 * np.pi * df["date"])

    if {"nswprice", "vicprice"}.issubset(df.columns):
        df["nsw_vic_price_diff"] = df["nswprice"] - df["vicprice"]

    if {"nswdemand", "vicdemand"}.issubset(df.columns):
        df["nsw_vic_demand_ratio"] = df["nswdemand"] / (df["vicdemand"] + 1e-6)

    if {"transfer", "nswdemand", "vicdemand"}.issubset(df.columns):
        df["transfer_demand_ratio"] = df["transfer"] / (
            df["nswdemand"] + df["vicdemand"] + 1e-6
        )

    return df


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET]).copy()
    train_df[TARGET] = train_df[TARGET].astype(str)

    classes = sorted(train_df[TARGET].unique())
    label_to_int = {label: idx for idx, label in enumerate(classes)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    y_full = train_df[TARGET].map(label_to_int).astype(int)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    feature_cols = [col for col in train_df.columns if col != TARGET]

    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

    categorical_cols = []
    numeric_cols = []

    for col in feature_cols:
        train_dtype = train_df[col].dtype
        test_dtype = test_df[col].dtype

        is_categorical = (
            train_dtype == object
            or train_dtype.name == "category"
            or test_dtype == object
            or test_dtype.name == "category"
        )

        if is_categorical:
            categorical_cols.append(col)
            train_df[col] = train_df[col].astype("string")
            test_df[col] = test_df[col].astype("string")
        else:
            numeric_cols.append(col)
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    if not numeric_cols:
        numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    sort_cols = []
    if "date" in train_df.columns:
        sort_cols.append("date")
    if "period" in train_df.columns:
        sort_cols.append("period")

    if sort_cols:
        train_df = train_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        y_full = y_full.loc[train_df.index].reset_index(drop=True)
    else:
        train_df = train_df.reset_index(drop=True)
        y_full = y_full.reset_index(drop=True)

    val_size = int(len(train_df) * 0.2)
    val_size = min(max(val_size, 1), len(train_df) - 1)

    if sort_cols:
        X_val = train_df.iloc[-val_size:][feature_cols]
        y_val = y_full.iloc[-val_size:]
        X_train = train_df.iloc[:-val_size][feature_cols]
        y_train = y_full.iloc[:-val_size]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            train_df[feature_cols],
            y_full,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_full,
        )

    models = {
        "lightgbm": LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        ),
    }

    validation_scores = {}
    validation_probabilities = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)

        validation_scores[name] = accuracy_score(y_val, pred)
        validation_probabilities[name] = proba

    weights = np.array([max(score, 1e-6) for score in validation_scores.values()])
    weights = weights / weights.sum()

    X_test = test_df[feature_cols]
    final_proba = np.zeros(len(X_test), dtype=float)

    for (name, model), weight in zip(models.items(), weights):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_cols, categorical_cols)),
                ("model", model),
            ]
        )

        pipeline.fit(train_df[feature_cols], y_full)
        final_proba += weight * pipeline.predict_proba(X_test)[:, 1]

    predicted_labels = np.where(final_proba >= 0.5, int_to_label[1], int_to_label[0])

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "class": predicted_labels,
        }
    )

    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()