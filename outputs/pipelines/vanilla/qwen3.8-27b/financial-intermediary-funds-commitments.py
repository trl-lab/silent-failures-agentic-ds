import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-commitments/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-commitments/data/test.parquet"
TARGET = "Amount in USD"


def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    train = train.copy()
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train[TARGET] = train[TARGET].replace([np.inf, -np.inf], np.nan)
    return train.dropna(subset=[TARGET])


def feature_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def make_medians(X: pd.DataFrame, numeric_cols) -> dict:
    medians = {}
    for col in numeric_cols:
        values = pd.to_numeric(X[col], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        median = values.median()
        medians[col] = 0.0 if pd.isna(median) else float(median)
    return medians


def prepare_features(
    X: pd.DataFrame,
    numeric_cols,
    categorical_cols,
    medians: dict
) -> pd.DataFrame:
    X = X.copy()

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        if col in medians and not pd.isna(medians[col]):
            X[col] = X[col].fillna(medians[col])

    for col in categorical_cols:
        X[col] = X[col].astype("object").where(X[col].notna(), "missing").astype(str)

    return X


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    if len(test) == 0:
        pd.DataFrame({
            "id": np.arange(0, dtype=int),
            TARGET: np.array([], dtype=float)
        }).to_csv("predictions.csv", index=False)
        return

    train = clean_train(train)

    if train.empty:
        pd.DataFrame({
            "id": np.arange(len(test), dtype=int),
            TARGET: 0.0
        }).to_csv("predictions.csv", index=False)
        return

    X_full_raw = train.drop(columns=[TARGET])
    y_full = train[TARGET].astype(float)

    X_test_raw = test.drop(columns=[TARGET], errors="ignore").reindex(columns=X_full_raw.columns)

    if X_full_raw.shape[1] == 0:
        pred = np.full(len(test), float(y_full.mean()), dtype=float)
        pd.DataFrame({
            "id": np.arange(len(test), dtype=int),
            TARGET: pred
        }).to_csv("predictions.csv", index=False)
        return

    numeric_cols, categorical_cols = feature_types(X_full_raw)
    cat_features = categorical_cols if categorical_cols else None

    medians_full = make_medians(X_full_raw, numeric_cols)
    X_full = prepare_features(X_full_raw, numeric_cols, categorical_cols, medians_full)
    X_test = prepare_features(X_test_raw, numeric_cols, categorical_cols, medians_full)

    best_iteration = 1000

    if len(train) > 20:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_full_raw,
            y_full,
            test_size=0.2,
            random_state=42
        )

        medians_val = make_medians(X_train_raw, numeric_cols)
        X_train = prepare_features(X_train_raw, numeric_cols, categorical_cols, medians_val)
        X_val = prepare_features(X_val_raw, numeric_cols, categorical_cols, medians_val)

        model = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
            allow_writing_files=False,
            loss_function="RMSE"
        )

        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_features,
            early_stopping_rounds=150,
            verbose=0
        )

        val_pred = model.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        print(f"Validation RMSE: {val_rmse:.6f}")

        best_iteration = model.get_best_iteration()
        if best_iteration is None or best_iteration <= 0:
            best_iteration = 1000

    final_iterations = max(100, int(best_iteration))

    final_model = CatBoostRegressor(
        iterations=final_iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        loss_function="RMSE"
    )

    final_model.fit(
        X_full,
        y_full,
        cat_features=cat_features,
        verbose=0
    )

    predictions = final_model.predict(X_test)
    predictions = np.asarray(predictions, dtype=float)

    predictions = np.nan_to_num(
        predictions,
        nan=float(y_full.mean()),
        posinf=float(y_full.max()),
        neginf=float(y_full.min())
    )

    if float(y_full.min()) >= 0.0:
        predictions = np.clip(predictions, 0.0, None)

    output = pd.DataFrame({
        "id": np.arange(len(test), dtype=int),
        TARGET: predictions
    })

    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()