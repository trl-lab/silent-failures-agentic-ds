import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "TOTAL_ACTIVE_ACCOUNTS_"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
WORK_DIR = "./working"
SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.csv")

os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

train = train.dropna(subset=[TARGET]).reset_index(drop=True)
y = train[TARGET].astype(float)
X = train.drop(columns=[TARGET])
test_features = test.drop(columns=[TARGET], errors="ignore")


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def is_categorical_series(s: pd.Series) -> bool:
    return not is_numeric_series(s)


def select_features(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Select usable numeric and categorical features using only X_train.
    This keeps feature selection leakage-safe when called per CV fold.
    """
    numeric_cols = []
    categorical_cols = []

    for col in X_train.columns:
        if col.lower() in {"id", "index", "row_id", "rowid"}:
            continue

        s = X_train[col]
        if s.isna().all():
            continue
        if s.nunique(dropna=True) <= 1:
            continue

        if is_numeric_series(s):
            numeric_cols.append(col)
        else:
            # Drop likely identifier-like high-cardinality non-numeric columns.
            n_non_missing = int(s.notna().sum())
            if n_non_missing == 0:
                continue
            if s.nunique(dropna=True) > max(20, int(0.5 * n_non_missing)):
                continue
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def build_model_data(
    X: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> pd.DataFrame:
    """
    Return only the selected columns. If no features are selected, return a
    single constant column so the pipeline still has a valid input.
    """
    cols = list(numeric_cols) + list(categorical_cols)
    if not cols:
        return pd.DataFrame({"__constant__": np.ones(len(X), dtype=float)})
    return X.reindex(columns=cols)


def make_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                (
                    "to_object",
                    FunctionTransformer(lambda df: df.astype(object), validate=False),
                ),
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="__missing__"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        transformers.append(
            (
                "const",
                FunctionTransformer(
                    lambda X: np.ones((X.shape[0], 1), dtype=float), validate=False
                ),
                ["__constant__"],
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.0
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=1.0))])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


cv = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmses = []

for train_idx, val_idx in cv.split(X):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    numeric_cols, categorical_cols = select_features(X_train_fold)
    model = make_pipeline(numeric_cols, categorical_cols)

    model.fit(
        build_model_data(X_train_fold, numeric_cols, categorical_cols),
        y_train_fold,
    )

    val_pred = model.predict(
        build_model_data(X_val_fold, numeric_cols, categorical_cols)
    )
    fold_rmses.append(rmse(y_val_fold.to_numpy(), val_pred))

cv_rmse = float(np.mean(fold_rmses))
print(f"Validation RMSE: {cv_rmse:.6f}")

# Refit on the full training set using the same feature-selection rule.
numeric_cols, categorical_cols = select_features(X)
final_model = make_pipeline(numeric_cols, categorical_cols)
final_model.fit(build_model_data(X, numeric_cols, categorical_cols), y)

test_pred = final_model.predict(
    build_model_data(test_features, numeric_cols, categorical_cols)
)
test_pred = np.clip(test_pred, 0.0, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: test_pred.astype(float),
    }
)
submission.to_csv(SUBMISSION_PATH, index=False)
