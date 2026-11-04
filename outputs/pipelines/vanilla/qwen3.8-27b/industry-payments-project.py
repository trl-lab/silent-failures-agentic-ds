import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import warnings

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-project/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-project/data/test.parquet"
TARGET = "projectPayment"


def clean_text_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)

    s = series.astype(object)
    s = s.where(s.notna(), "")
    s = s.astype(str)
    s = s.str.strip().str.lower()

    missing_values = {"", "nan", "none", "null", "na", "n/a", "nil"}
    s = s.replace({value: "" for value in missing_values})

    return s


def is_id_like(column_name: str) -> bool:
    name = str(column_name).lower()
    id_names = {
        "id",
        "uuid",
        "sourceid",
        "source_id",
        "recordid",
        "record_id",
        "paymentid",
        "payment_id",
        "projectid",
        "project_id",
    }
    return name in id_names or name.endswith("_id") or name.endswith("id")


def prepare_data(train: pd.DataFrame, test: pd.DataFrame, target: str):
    train = train.copy()
    test = test.copy()

    if target not in train.columns:
        raise ValueError(f"Target column '{target}' not found in train data.")

    train[target] = pd.to_numeric(train[target], errors="coerce")
    train = train[np.isfinite(train[target])].copy()
    train[target] = train[target].astype(float)

    test = test.drop(columns=[target], errors="ignore")

    common_cols = [col for col in train.columns if col in test.columns and col != target]

    numeric_cols = []
    categorical_cols = []

    for col in common_cols:
        if pd.api.types.is_numeric_dtype(train[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    text_cols = [
        col
        for col in categorical_cols
        if any(keyword in col.lower() for keyword in ("note", "comment", "description", "text", "memo"))
    ]

    text_feature_cols = []
    for col in text_cols:
        train_text = clean_text_series(train[col])
        test_text = clean_text_series(test[col])

        len_col = f"{col}_len"
        has_col = f"{col}_has"

        train[len_col] = train_text.str.len().astype(float)
        train[has_col] = (train_text != "").astype(float)

        test[len_col] = test_text.str.len().astype(float)
        test[has_col] = (test_text != "").astype(float)

        text_feature_cols.extend([len_col, has_col])

        train = train.drop(columns=[col])
        test = test.drop(columns=[col])

    categorical_cols = [col for col in categorical_cols if col not in text_cols]

    final_categorical_cols = []
    for col in categorical_cols:
        train[col] = train[col].astype(object).fillna("missing")
        test[col] = test[col].astype(object).fillna("missing")

        nunique = train[col].nunique(dropna=True)
        if is_id_like(col) and nunique > max(1000, int(0.8 * len(train))):
            train = train.drop(columns=[col])
            test = test.drop(columns=[col])
            continue

        final_categorical_cols.append(col)

    for col in numeric_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

        train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)

        median_value = train[col].median()
        if not np.isfinite(median_value):
            median_value = 0.0

        train[col] = train[col].fillna(median_value)
        test[col] = test[col].fillna(median_value)

    candidate_feature_cols = numeric_cols + final_categorical_cols + text_feature_cols
    candidate_feature_cols = [
        col for col in candidate_feature_cols if col in train.columns and col in test.columns
    ]

    feature_cols = []
    for col in candidate_feature_cols:
        if train[col].nunique(dropna=True) > 1:
            feature_cols.append(col)

    X_train = train[feature_cols].copy()
    y_train = train[target].copy()
    X_test = test[feature_cols].copy()

    final_categorical_cols = [col for col in final_categorical_cols if col in feature_cols]

    return X_train, y_train, X_test, final_categorical_cols


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    X_train, y_train, X_test, categorical_cols = prepare_data(train_df, test_df, TARGET)

    if X_train.shape[1] == 0 or len(X_train) == 0:
        default_value = float(y_train.mean()) if len(y_train) > 0 else 0.0
        predictions = np.full(len(X_test), default_value, dtype=float)
    else:
        categorical_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

        base_params = {
            "iterations": 2000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "cat_features": categorical_indices,
        }

        if len(X_train) >= 200:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.2,
                random_state=42,
            )

            model = CatBoostRegressor(**base_params)
            model.fit(
                X_fit,
                y_fit,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=0,
            )

            try:
                best_iteration = model.get_best_iteration()
            except Exception:
                best_iteration = None

            if best_iteration is None or best_iteration < 0:
                best_iterations = base_params["iterations"]
            else:
                best_iterations = int(best_iteration) + 1

            best_iterations = int(max(100, min(best_iterations, base_params["iterations"])))
        else:
            best_iterations = 1000

        final_params = base_params.copy()
        final_params["iterations"] = best_iterations

        final_model = CatBoostRegressor(**final_params)
        final_model.fit(X_train, y_train, verbose=0)

        predictions = final_model.predict(X_test)
        predictions = np.asarray(predictions, dtype=float)
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            "projectPayment": predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()