import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/pol-terminal/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/pol-terminal/data/test.parquet"
TARGET = "CAPACITY"
RANDOM_STATE = 42
MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"

DROP_COLS = {
    "OBJECTID",
    "TERM_ID",
    "NAME",
    "ADDRESS",
    "CITY",
    "ZIP",
    "ZIP4",
    "TELEPHONE",
    "WEBSITE",
    "GlobalID",
    "EXSTARS_I",
    "OWNER",
    "OPERATOR",
    "NAICS_DESC",
    "SOURCEDATE",
    "VAL_DATE",
}


def _clean_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    if TARGET not in train_df.columns:
        return train_df, test_df, [], [], []

    drop_cols = [c for c in DROP_COLS if c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    feature_cols = [c for c in train_df.columns if c != TARGET]
    train_df = train_df[feature_cols + [TARGET]]
    test_df = test_df.reindex(columns=feature_cols)

    numeric_like_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_df[c]) or pd.api.types.is_numeric_dtype(test_df[c])
    ]
    train_df = _clean_numeric(train_df, numeric_like_cols)
    test_df = _clean_numeric(test_df, numeric_like_cols)

    train_df = train_df.dropna(subset=[TARGET])
    train_df[TARGET] = train_df[TARGET].astype("float64")
    train_df = train_df[np.isfinite(train_df[TARGET])]

    if len(train_df) == 0:
        return train_df, test_df, [], [], []

    n_train = len(train_df)
    high_card_threshold = max(50, min(1000, int(0.2 * n_train)))

    selected_cols = []
    for col in feature_cols:
        s = train_df[col]

        if pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            if len(non_null) == 0:
                continue
            if not np.isfinite(non_null.astype("float64")).any():
                continue
            if non_null.nunique() <= 1:
                continue
            if s.isna().mean() > 0.95:
                continue
            selected_cols.append(col)
        else:
            non_null = s.dropna()
            if len(non_null) == 0:
                continue
            n_unique = non_null.astype(str).nunique()
            if n_unique <= 1:
                continue
            if n_unique > high_card_threshold:
                continue
            if s.isna().mean() > 0.95:
                continue
            selected_cols.append(col)

    feature_cols = selected_cols
    if not feature_cols:
        return train_df, test_df, [], [], []

    train_df = train_df[feature_cols + [TARGET]]
    test_df = test_df[feature_cols]

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return train_df, test_df, feature_cols, numeric_cols, categorical_cols


class CategoryMapper:
    def __init__(self, categorical_cols: list):
        self.categorical_cols = categorical_cols
        self.categories_ = {}

    def _to_string_series(self, s: pd.Series) -> pd.Series:
        return s.astype("string").fillna(MISSING_TOKEN)

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.categorical_cols:
            if col not in X.columns:
                continue
            s = self._to_string_series(X[col])
            cats = sorted(s.unique().tolist())
            for token in (MISSING_TOKEN, UNKNOWN_TOKEN):
                if token not in cats:
                    cats.append(token)
            self.categories_[col] = cats
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.categorical_cols:
            if col not in X.columns:
                X[col] = MISSING_TOKEN
                cats = [MISSING_TOKEN, UNKNOWN_TOKEN]
            else:
                s = self._to_string_series(X[col])
                cats = self.categories_.get(col, [MISSING_TOKEN, UNKNOWN_TOKEN])
                s = s.where(s.isin(cats), UNKNOWN_TOKEN)
                X[col] = pd.Categorical(s, categories=cats)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X).transform(X)


def make_model(n_estimators: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def transform_target(y: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.log1p(np.maximum(y, 0.0))
    return y


def inverse_target(y: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.expm1(y)
    return y


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    if len(y_true) == 0:
        return np.inf
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    try:
        train_df, test_df, feature_cols, numeric_cols, categorical_cols = prepare_data(train_df, test_df)

        if len(train_df) == 0 or not feature_cols:
            predictions = np.zeros(len(test_df), dtype="float64")
        else:
            X_all = train_df[feature_cols].copy()
            y_all = train_df[TARGET].astype("float64").to_numpy()

            if len(X_all) > 50:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE
                )
                use_validation = True
            else:
                X_train, X_val, y_train, y_val = X_all, X_all, y_all, y_all
                use_validation = False

            mapper = CategoryMapper(categorical_cols)
            X_train = mapper.fit_transform(X_train)
            X_val = mapper.transform(X_val)

            best_rmse = np.inf
            best_use_log = False
            best_n_estimators = 1000

            for use_log in (False, True):
                y_train_t = transform_target(y_train, use_log)
                y_val_t = transform_target(y_val, use_log)

                model = make_model(2000)
                try:
                    if use_validation:
                        model.fit(
                            X_train,
                            y_train_t,
                            eval_set=[(X_val, y_val_t)],
                            eval_metric="rmse",
                            categorical_feature=categorical_cols if categorical_cols else None,
                            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
                        )
                    else:
                        model.fit(
                            X_train,
                            y_train_t,
                            categorical_feature=categorical_cols if categorical_cols else None,
                        )
                except Exception:
                    model = make_model(1000)
                    model.fit(
                        X_train,
                        y_train_t,
                        categorical_feature=categorical_cols if categorical_cols else None,
                    )

                best_iter = getattr(model, "best_iteration_", None)
                if isinstance(best_iter, (int, np.integer)) and best_iter > 0:
                    n_est = int(best_iter)
                else:
                    n_est = 1000
                n_est = int(max(100, min(n_est, 3000)))

                pred_val = model.predict(X_val)
                pred_val = inverse_target(pred_val, use_log)
                if y_all.min() >= 0:
                    pred_val = np.maximum(pred_val, 0.0)

                score = rmse(y_val, pred_val)
                if np.isfinite(score) and score < best_rmse:
                    best_rmse = score
                    best_use_log = use_log
                    best_n_estimators = n_est

            mapper_all = CategoryMapper(categorical_cols)
            X_all = mapper_all.fit_transform(X_all)
            y_all_t = transform_target(y_all, best_use_log)

            final_model = make_model(best_n_estimators)
            final_model.fit(
                X_all,
                y_all_t,
                categorical_feature=categorical_cols if categorical_cols else None,
            )

            X_test = mapper_all.transform(test_df[feature_cols])
            predictions = final_model.predict(X_test)
            predictions = inverse_target(predictions, best_use_log)

            if y_all.min() >= 0:
                predictions = np.maximum(predictions, 0.0)

            predictions = np.asarray(predictions, dtype="float64")
            predictions = np.where(np.isfinite(predictions), predictions, 0.0)

    except Exception:
        predictions = np.zeros(len(test_df), dtype="float64")

    pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "CAPACITY": predictions,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()