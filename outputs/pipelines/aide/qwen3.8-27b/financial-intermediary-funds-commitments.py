import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

TARGET = "Amount in USD"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    id_like = [
        c
        for c in df.columns
        if str(c).strip().lower() in {"id", "index", "row_id", "rowid", "row id"}
    ]
    return df.drop(columns=id_like, errors="ignore")


train = drop_id_columns(train)
test = drop_id_columns(test)

if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train data.")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.dropna(subset=[TARGET]).reset_index(drop=True)
test = test.drop(columns=[TARGET], errors="ignore")

X_all = train.drop(columns=[TARGET])
y_all = train[TARGET].to_numpy(dtype=float)


def identify_feature_types(X: pd.DataFrame):
    cat_cols, num_cols, date_cols = [], [], []

    for col in X.columns:
        s = X[col]

        if pd.api.types.is_bool_dtype(s):
            cat_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            num_cols.append(col)
        else:
            non_null = s.dropna()
            if len(non_null) == 0:
                cat_cols.append(col)
                continue

            numeric_conv = pd.to_numeric(non_null, errors="coerce")
            if numeric_conv.notna().mean() > 0.8:
                num_cols.append(col)
                continue

            date_conv = pd.to_datetime(non_null, errors="coerce")
            if date_conv.notna().mean() > 0.8:
                date_cols.append(col)
            else:
                cat_cols.append(col)

    return cat_cols, num_cols, date_cols


cat_cols, num_cols, date_cols = identify_feature_types(X_all)


class Preprocessor:
    def __init__(self, cat_cols, num_cols, date_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.date_cols = list(date_cols)
        self.num_medians = {}
        self.cat_mappings = {}

    def _normalize_categorical_series(self, s: pd.Series) -> pd.Series:
        s = s.copy()

        if pd.api.types.is_numeric_dtype(s):

            def convert(x):
                if pd.isna(x):
                    return np.nan
                if isinstance(x, float) and np.isfinite(x) and x.is_integer():
                    return str(int(x))
                return str(x)

            s = s.map(convert)
        else:
            s = s.astype("string")

        return s

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        for col in self.num_cols + self.date_cols:
            if col not in X.columns:
                self.num_medians[col] = 0.0
                continue

            if col in self.date_cols:
                vals = pd.to_datetime(X[col], errors="coerce")
                vals = (vals - pd.Timestamp("1970-01-01")).dt.days
            else:
                vals = pd.to_numeric(X[col], errors="coerce")

            median = vals.median(skipna=True)
            if pd.isna(median):
                median = 0.0
            self.num_medians[col] = float(median)

        for col in self.cat_cols:
            if col not in X.columns:
                self.cat_mappings[col] = []
                continue

            normalized = self._normalize_categorical_series(X[col]).dropna()
            categories = sorted(normalized.unique().tolist())
            self.cat_mappings[col] = categories

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.date_cols:
            if col not in X.columns:
                X[col] = self.num_medians.get(col, 0.0)
                continue

            vals = pd.to_datetime(X[col], errors="coerce")
            vals = (vals - pd.Timestamp("1970-01-01")).dt.days
            vals = pd.to_numeric(vals, errors="coerce")
            X[col] = vals.fillna(self.num_medians.get(col, 0.0))

        for col in self.num_cols:
            if col not in X.columns:
                X[col] = self.num_medians.get(col, 0.0)
                continue

            vals = pd.to_numeric(X[col], errors="coerce")
            X[col] = vals.fillna(self.num_medians.get(col, 0.0))

        for col in self.cat_cols:
            if col not in X.columns:
                X[col] = 0
                continue

            categories = self.cat_mappings.get(col, [])
            mapping = {value: idx for idx, value in enumerate(categories)}
            missing_code = len(categories)

            normalized = self._normalize_categorical_series(X[col])
            X[col] = normalized.map(
                lambda v: missing_code if pd.isna(v) else mapping.get(v, missing_code)
            ).astype(int)

        return X


def make_params():
    return {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }


def clip_if_nonnegative(predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
    if np.nanmin(y) >= 0:
        return np.clip(predictions, 0.0, None)
    return predictions


if X_all.shape[1] == 0 or len(train) == 0:
    cv_rmse = 0.0
    test_pred = np.zeros(len(test), dtype=float)
    print(f"Validation RMSE: {cv_rmse:.6f}")
else:
    n_splits = min(5, len(train))
    if n_splits < 2:
        n_splits = 1

    if n_splits == 1:
        splits = [(np.arange(len(train)), np.array([], dtype=int))]
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X_all))

    cv_scores = []
    best_iterations = []
    params = make_params()
    num_boost_round = 5000

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train_fold_raw = X_all.iloc[train_idx]
        X_val_fold_raw = X_all.iloc[val_idx]
        y_train_fold = y_all[train_idx]
        y_val_fold = y_all[val_idx]

        preprocessor = Preprocessor(cat_cols, num_cols, date_cols).fit(X_train_fold_raw)
        X_train_fold = preprocessor.transform(X_train_fold_raw)
        X_val_fold = preprocessor.transform(X_val_fold_raw)

        categorical_features = [c for c in cat_cols if c in X_train_fold.columns]

        train_dataset = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )

        if len(val_idx) > 0:
            valid_dataset = lgb.Dataset(
                X_val_fold,
                label=y_val_fold,
                categorical_feature=categorical_features,
                reference=train_dataset,
                free_raw_data=False,
            )
            model = lgb.train(
                params,
                train_dataset,
                num_boost_round=num_boost_round,
                valid_sets=[valid_dataset],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            best_iter = (
                model.best_iteration if model.best_iteration > 0 else num_boost_round
            )
            best_iterations.append(best_iter)

            val_pred = model.predict(X_val_fold, num_iteration=best_iter)
            val_pred = clip_if_nonnegative(val_pred, y_val_fold)
            fold_rmse = float(np.sqrt(mean_squared_error(y_val_fold, val_pred)))
            cv_scores.append(fold_rmse)
            print(f"Fold {fold_idx} RMSE: {fold_rmse:.6f}")
        else:
            model = lgb.train(
                params,
                train_dataset,
                num_boost_round=num_boost_round,
                callbacks=[lgb.log_evaluation(-1)],
            )
            best_iterations.append(num_boost_round)

    if cv_scores:
        cv_rmse = float(np.mean(cv_scores))
        print(f"Validation RMSE: {cv_rmse:.6f}")
    else:
        cv_rmse = float(
            np.sqrt(mean_squared_error(y_all, np.full_like(y_all, np.mean(y_all))))
        )
        print(f"Validation RMSE: {cv_rmse:.6f}")

    final_rounds = (
        int(np.clip(np.mean(best_iterations), 1, num_boost_round))
        if best_iterations
        else 1000
    )

    final_preprocessor = Preprocessor(cat_cols, num_cols, date_cols).fit(X_all)
    X_train_final = final_preprocessor.transform(X_all)
    X_test_final = final_preprocessor.transform(test)

    categorical_features = [c for c in cat_cols if c in X_train_final.columns]

    final_dataset = lgb.Dataset(
        X_train_final,
        label=y_all,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        final_dataset,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test_final, num_iteration=final_rounds)
    test_pred = clip_if_nonnegative(test_pred, y_all)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "Amount in USD": test_pred.astype(float),
    }
)

submission_path = os.path.join(WORK_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")
