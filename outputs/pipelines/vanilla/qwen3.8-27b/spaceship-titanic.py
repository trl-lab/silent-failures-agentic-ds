import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/spaceship-titanic/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/spaceship-titanic/data/test.parquet"

def clean_dataframe(df, has_target):
    df = df.copy()

    for col in ["PassengerId", "Name"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype("string").str.strip()
        parts = cabin.str.split("/", n=2, expand=True)

        if isinstance(parts, pd.Series):
            parts = pd.DataFrame({0: parts})

        while parts.shape[1] < 3:
            parts[parts.shape[1]] = pd.NA

        df["Deck"] = parts[0].str.strip()
        df["Side"] = parts[2].str.strip()
        df["CabinNum"] = pd.to_numeric(parts[1].str.strip(), errors="coerce")
        df = df.drop(columns=["Cabin"])

        for col in ["Deck", "Side"]:
            df[col] = df[col].replace(["", "nan", "None", "NaN", "null", "missing"], pd.NA)

    for col in ["CryoSleep", "VIP"]:
        if col in df.columns:
            s = df[col].astype("string").str.strip().str.lower()
            df[col] = s.map(
                {
                    "true": "True",
                    "false": "False",
                    "1": "True",
                    "0": "False",
                    "1.0": "True",
                    "0.0": "False",
                }
            ).astype(object)

    for col in ["HomePlanet", "Destination"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .replace(["", "nan", "None", "NaN", "null", "missing"], pd.NA)
                .astype(object)
            )

    if "Transported" in df.columns:
        if has_target:
            target = df["Transported"]
            if target.dtype == object:
                target = (
                    target.astype("string")
                    .str.strip()
                    .str.lower()
                    .map(
                        {
                            "true": 1,
                            "false": 0,
                            "1": 1,
                            "0": 0,
                            "1.0": 1,
                            "0.0": 0,
                        }
                    )
                )
            df["Transported"] = target.fillna(0).astype(int)
        else:
            df = df.drop(columns=["Transported"])

    return df

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def coerce_numeric(series):
    if series.dtype == object or str(series.dtype) in {"string", "category"}:
        series = series.astype("string").str.strip()
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

def coerce_categorical(series):
    series = series.astype("string").str.strip()
    series = series.replace(["", "nan", "None", "NaN", "null", "missing"], pd.NA)
    series = series.astype(object)
    series = series.where(pd.notna(series), np.nan)
    return series

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = clean_dataframe(train_df, has_target=True)
    test_df = clean_dataframe(test_df, has_target=False)

    y = train_df["Transported"]
    X = train_df.drop(columns=["Transported"])
    X_test = test_df.copy()

    numeric_features = [
        col
        for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]
        if col in X.columns
    ]
    categorical_features = [
        col
        for col in ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
        if col in X.columns
    ]

    for col in numeric_features + categorical_features:
        if col not in X_test.columns:
            X_test[col] = np.nan

    for col in numeric_features:
        X[col] = coerce_numeric(X[col])
        X_test[col] = coerce_numeric(X_test[col])

    for col in categorical_features:
        X[col] = coerce_categorical(X[col])
        X_test[col] = coerce_categorical(X_test[col])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    _ = accuracy_score(y_val, val_predictions)

    pipeline.fit(X, y)
    test_predictions = pipeline.predict(X_test).astype(int)

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_predictions), dtype=int),
            "Transported": test_predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()