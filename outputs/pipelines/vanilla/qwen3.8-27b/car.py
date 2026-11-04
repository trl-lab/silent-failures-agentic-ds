import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/car/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/car/data/test.parquet"
TARGET = "class"
MISSING_TOKEN = "__MISSING__"


def clean_features(df, feature_cols, numeric_cols, categorical_cols, numeric_fill_values=None):
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if numeric_fill_values is not None and col in numeric_fill_values:
                df[col] = df[col].fillna(numeric_fill_values[col])
            else:
                df[col] = df[col].fillna(0.0)

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")
            df[col] = df[col].apply(lambda x: MISSING_TOKEN if pd.isna(x) else str(x))

    return df[feature_cols]


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in train_df.columns:
        train_df = train_df.dropna(subset=[TARGET])

    feature_cols = [c for c in train_df.columns if c != TARGET]

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[c])
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_fill_values = {}
    for col in numeric_cols:
        median = train_df[col].median()
        numeric_fill_values[col] = 0.0 if pd.isna(median) else float(median)

    X_train = clean_features(
        train_df,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_fill_values=numeric_fill_values,
    )
    X_test = clean_features(
        test_df,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_fill_values=numeric_fill_values,
    )

    y_train = train_df[TARGET].astype(str)
    label_encoder = LabelEncoder().fit(y_train)
    y_train_enc = label_encoder.transform(y_train)

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop") if transformers else "passthrough"

    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            random_state=42,
            verbose=0,
            thread_count=-1,
        ),
    }

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train_enc,
            test_size=0.2,
            random_state=42,
            stratify=y_train_enc,
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train_enc,
            test_size=0.2,
            random_state=42,
        )

    best_name = None
    best_score = -1.0

    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_tr, y_tr)
        val_pred = pipeline.predict(X_val)
        score = accuracy_score(y_val, val_pred)

        if score > best_score:
            best_score = score
            best_name = name

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[best_name]),
    ])
    final_pipeline.fit(X_train, y_train_enc)

    test_pred_enc = final_pipeline.predict(X_test)
    test_pred = label_encoder.inverse_transform(test_pred_enc)

    predictions = pd.DataFrame({
        "id": np.arange(len(test_pred), dtype=int),
        "class": test_pred,
    })
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()