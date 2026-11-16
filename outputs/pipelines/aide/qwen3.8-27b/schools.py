import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

TARGET = "Enrollment_FT_Teachers_Ratio"
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def is_id_column(name: str) -> bool:
    n = str(name).lower()
    return n in {"id", "index", "school_id", "schoolid", "school"} or n.endswith("_id")


def datetime_to_days(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return (s - pd.Timestamp("1970-01-01", tz="UTC")).dt.days.astype("float64")


def preprocess_pair(X_train: pd.DataFrame, X_val: pd.DataFrame, cols: list[str]):
    """
    Preprocess train and validation/test data together using only X_train
    to determine categories and feature validity.
    """
    Xtr = X_train[cols].copy()
    Xva = X_val[cols].copy()

    keep = []
    cat_cols = []

    for c in cols:
        try:
            s_tr = Xtr[c]
            s_va = Xva[c]

            if pd.api.types.is_datetime64_any_dtype(
                s_tr
            ) or pd.api.types.is_datetime64_any_dtype(s_va):
                Xtr[c] = datetime_to_days(s_tr)
                Xva[c] = datetime_to_days(s_va)

            elif pd.api.types.is_numeric_dtype(s_tr) or pd.api.types.is_numeric_dtype(
                s_va
            ):
                Xtr[c] = pd.to_numeric(s_tr, errors="coerce")
                Xva[c] = pd.to_numeric(s_va, errors="coerce")

            elif pd.api.types.is_bool_dtype(s_tr) or pd.api.types.is_bool_dtype(s_va):
                Xtr[c] = Xtr[c].astype("float64")
                Xva[c] = Xva[c].astype("float64")

            else:
                # Treat as categorical. Fit categories on the training side only.
                if s_tr.notna().sum() == 0:
                    continue

                tr_cat = pd.Categorical(s_tr)
                categories = tr_cat.categories

                if len(categories) <= 1:
                    continue

                Xtr[c] = pd.Categorical(s_tr, categories=categories).astype("category")
                Xva[c] = pd.Categorical(s_va, categories=categories).astype("category")
                cat_cols.append(c)

            # Keep only features that are not constant or fully missing in the training side.
            if Xtr[c].notna().sum() > 0 and Xtr[c].nunique(dropna=True) > 1:
                keep.append(c)

        except Exception:
            # If a column cannot be processed safely, drop it.
            continue

    Xtr = Xtr[keep]
    Xva = Xva[keep]
    return Xtr, Xva, keep, cat_cols


def main():
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    # Remove target from test if present.
    if TARGET in test.columns:
        test = test.drop(columns=[TARGET])

    # Remove obvious identifier columns.
    train_id_cols = [c for c in train.columns if is_id_column(c) and c != TARGET]
    test_id_cols = [c for c in test.columns if is_id_column(c) and c != TARGET]
    train = train.drop(columns=train_id_cols, errors="ignore")
    test = test.drop(columns=test_id_cols, errors="ignore")

    # Extract and clean target.
    if TARGET not in train.columns:
        raise ValueError(f"Target column {TARGET} not found in train data.")

    y = train[TARGET].astype(float).replace([np.inf, -np.inf], np.nan)
    train = train.drop(columns=[TARGET])

    valid = y.notna()
    y = y[valid].reset_index(drop=True)
    train = train.loc[valid].reset_index(drop=True)

    # Use only features present in both train and test to avoid leakage/mismatch.
    common_cols = [c for c in train.columns if c in test.columns]

    if len(y) == 0:
        pred_test = np.zeros(len(test), dtype=float)
        pd.DataFrame(
            {
                "id": np.arange(len(test), dtype=int),
                TARGET: pred_test,
            }
        ).to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
        print("CV RMSE: nan")
        print("Saved submission.csv")
        return

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
        "verbosity": -1,
    }

    n_splits = 5 if len(y) >= 5 else max(2, len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores = []
    best_iterations = []

    for tr_idx, va_idx in kf.split(y):
        Xtr_raw = train.iloc[tr_idx]
        Xva_raw = train.iloc[va_idx]
        ytr = y.iloc[tr_idx]
        yva = y.iloc[va_idx]

        Xtr, Xva, keep_cols, cat_cols = preprocess_pair(Xtr_raw, Xva_raw, common_cols)

        if Xtr.shape[1] == 0:
            pred_va = np.full(len(yva), ytr.mean(), dtype=float)
            cv_scores.append(float(np.sqrt(mean_squared_error(yva, pred_va))))
            continue

        cat_arg = cat_cols if cat_cols else "auto"
        dtrain = lgb.Dataset(
            Xtr,
            label=ytr,
            categorical_feature=cat_arg,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            Xva,
            label=yva,
            categorical_feature=cat_arg,
            reference=dtrain,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = max(1, int(model.best_iteration))
        best_iterations.append(best_iter)

        pred_va = model.predict(Xva, num_iteration=best_iter)
        pred_va = np.nan_to_num(
            pred_va, nan=ytr.mean(), posinf=ytr.max(), neginf=ytr.min()
        )
        cv_scores.append(float(np.sqrt(mean_squared_error(yva, pred_va))))

    cv_rmse = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"CV RMSE: {cv_rmse:.6f} (+/- {cv_std:.6f})")

    # Fit final model on all training data.
    Xtr_full, Xte, keep_final, cat_final = preprocess_pair(train, test, common_cols)

    if Xtr_full.shape[1] == 0:
        pred_test = np.full(len(test), y.mean(), dtype=float)
    else:
        cat_arg = cat_final if cat_final else "auto"
        dtrain_full = lgb.Dataset(
            Xtr_full,
            label=y,
            categorical_feature=cat_arg,
            free_raw_data=False,
        )

        if best_iterations:
            final_rounds = max(1, int(round(np.mean(best_iterations))))
        else:
            final_rounds = 1000

        final_model = lgb.train(
            params,
            dtrain_full,
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(-1)],
        )

        pred_test = final_model.predict(Xte, num_iteration=final_rounds)
        pred_test = np.nan_to_num(
            pred_test, nan=y.mean(), posinf=y.max(), neginf=y.min()
        )

        # The target is a ratio and should normally be non-negative.
        if y.min() >= 0:
            pred_test = np.clip(pred_test, 0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            TARGET: pred_test.astype(float),
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
