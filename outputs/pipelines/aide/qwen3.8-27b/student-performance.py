import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

TARGET = "G3"


def build_pipeline(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(categorical_features),
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = train.dropna(subset=[TARGET])

    feature_cols = [
        col
        for col in train.columns
        if col not in {TARGET, "id"} and col in test.columns
    ]

    X_train = train[feature_cols].copy()
    y_train = train[TARGET].astype(float)
    X_test = test[feature_cols].copy()

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    pipeline = build_pipeline(numeric_features, categorical_features)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )

    cv_rmse = -scores["test_score"].mean()
    print(f"CV RMSE: {cv_rmse:.4f}")

    final_model = build_pipeline(numeric_features, categorical_features)
    final_model.fit(X_train, y_train)

    predictions = final_model.predict(X_test)
    predictions = np.clip(predictions, 0.0, 20.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "G3": predictions,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
