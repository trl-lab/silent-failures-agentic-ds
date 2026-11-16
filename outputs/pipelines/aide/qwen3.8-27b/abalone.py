import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

TARGET = "Class_number_of_rings"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = train.dropna(subset=[TARGET])
y = train[TARGET].astype(float).to_numpy()

X_train_raw = train.drop(columns=[TARGET])
X_test_raw = test.copy()
if TARGET in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=[TARGET])

ID_LIKE = {"id", "index", "row_id", "rowid"}
X_train_raw = X_train_raw.drop(
    columns=[c for c in X_train_raw.columns if str(c).lower() in ID_LIKE]
)
X_test_raw = X_test_raw.drop(
    columns=[c for c in X_test_raw.columns if str(c).lower() in ID_LIKE]
)

EXPECTED_NUMERIC = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]


def coerce_numeric_columns(df):
    df = df.copy()
    for col in EXPECTED_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


X_train_raw = coerce_numeric_columns(X_train_raw)
X_test_raw = coerce_numeric_columns(X_test_raw)

numeric_cols = [
    c for c in X_train_raw.columns if pd.api.types.is_numeric_dtype(X_train_raw[c])
]
full_medians = X_train_raw[numeric_cols].median()

SEX_CATEGORIES = ["F", "I", "M", "Unknown"]


def prepare(df, medians, feature_cols=None):
    df = coerce_numeric_columns(df)
    df = df.loc[:, ~df.columns.duplicated()]

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in num_cols:
        if col in medians.index and not pd.isna(medians[col]):
            df[col] = df[col].fillna(medians[col])
        else:
            df[col] = df[col].fillna(0.0)

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype("string").fillna("Unknown")
        df["Sex"] = df["Sex"].str.strip().str.upper()
        df["Sex"] = df["Sex"].replace(
            {"": "Unknown", "NAN": "Unknown", "NONE": "Unknown"}
        )
        df["Sex"] = df["Sex"].where(df["Sex"].isin(["M", "F", "I"]), "Unknown")

        sex_dummies = pd.get_dummies(df["Sex"], prefix="Sex", dtype=float)
        sex_dummies = sex_dummies.reindex(
            columns=[f"Sex_{c}" for c in SEX_CATEGORIES],
            fill_value=0.0,
        )

        conflict_cols = [c for c in df.columns if c in sex_dummies.columns]
        if conflict_cols:
            df = df.drop(columns=conflict_cols)

        df = pd.concat([df.drop(columns=["Sex"]), sex_dummies], axis=1)

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        df = df.drop(columns=non_numeric)

    eps = 1e-6
    L = df["Length"] if "Length" in df.columns else 1.0
    D = df["Diameter"] if "Diameter" in df.columns else 1.0
    H = df["Height"] if "Height" in df.columns else 1.0
    W = df["Whole weight"] if "Whole weight" in df.columns else 1.0
    S = df["Shucked weight"] if "Shucked weight" in df.columns else 1.0
    V = df["Viscera weight"] if "Viscera weight" in df.columns else 1.0
    Sh = df["Shell weight"] if "Shell weight" in df.columns else 1.0

    df["Volume"] = L * D * H
    df["ShellWeightRatio"] = Sh / (W + eps)
    df["ShuckedWeightRatio"] = S / (W + eps)
    df["VisceraWeightRatio"] = V / (W + eps)
    df["MeatWeight"] = S + V
    df["MeatRatio"] = (S + V) / (W + eps)
    df["LengthDiameterRatio"] = L / (D + eps)
    df["HeightLengthRatio"] = H / (L + eps)
    df["ShellVolumeRatio"] = Sh / (df["Volume"] + eps)

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if feature_cols is not None:
        df = df.reindex(columns=feature_cols, fill_value=0.0)

    return df


feature_cols = prepare(X_train_raw, full_medians).columns.tolist()
feature_cols = list(dict.fromkeys(feature_cols))

BASE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}

SEEDS = [42, 7, 123, 2024, 31337]
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 50

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y), dtype=float)
best_iters_by_seed = {seed: [] for seed in SEEDS}

for tr_idx, va_idx in kf.split(X_train_raw):
    fold_train_raw = X_train_raw.iloc[tr_idx]
    fold_val_raw = X_train_raw.iloc[va_idx]

    fold_numeric_cols = [
        c
        for c in fold_train_raw.columns
        if pd.api.types.is_numeric_dtype(fold_train_raw[c])
    ]
    fold_medians = fold_train_raw[fold_numeric_cols].median()

    X_tr = prepare(fold_train_raw, fold_medians, feature_cols)
    X_va = prepare(fold_val_raw, fold_medians, feature_cols)

    fold_val_preds = np.zeros(len(va_idx), dtype=float)

    for seed in SEEDS:
        params = BASE_PARAMS.copy()
        params["seed"] = seed

        dtrain = lgb.Dataset(X_tr, label=y[tr_idx])
        dvalid = lgb.Dataset(X_va, label=y[va_idx], reference=dtrain)

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        best_iter = getattr(booster, "best_iteration", None)
        if best_iter is None or best_iter <= 0:
            best_iter = NUM_BOOST_ROUND
        best_iter = int(best_iter)

        best_iters_by_seed[seed].append(best_iter)
        fold_val_preds += booster.predict(X_va, num_iteration=best_iter)

    oof_preds[va_idx] = fold_val_preds / len(SEEDS)

cv_rmse = root_mean_squared_error(y, oof_preds)
print(f"CV RMSE: {cv_rmse:.6f}")

X_train = prepare(X_train_raw, full_medians, feature_cols)
X_test = prepare(X_test_raw, full_medians, feature_cols)

test_preds = np.zeros(len(X_test), dtype=float)

for seed in SEEDS:
    final_rounds = int(np.median(best_iters_by_seed[seed]))
    if final_rounds <= 0:
        final_rounds = NUM_BOOST_ROUND
    final_rounds = min(NUM_BOOST_ROUND, max(1, final_rounds))

    params = BASE_PARAMS.copy()
    params["seed"] = seed

    dtrain_all = lgb.Dataset(X_train, label=y)
    final_model = lgb.train(
        params,
        dtrain_all,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_preds += final_model.predict(X_test, num_iteration=final_rounds)

test_preds /= len(SEEDS)
test_preds = np.clip(test_preds, y.min(), y.max())

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        TARGET: test_preds,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
