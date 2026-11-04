import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "SeriousDlqin2yrs"
DROP_COLS = [TARGET, "Unnamed: 0", "id"]
BASE_FEATURES = [c for c in train.columns if c not in DROP_COLS]

OUTLIER_COLS = ["DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]
LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def get_outlier_caps(X: pd.DataFrame) -> dict:
    caps = {}
    for col in OUTLIER_COLS:
        if col not in X.columns:
            continue
        vals = X[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) > 0:
            caps[col] = float(np.nanquantile(vals.to_numpy(dtype=float), 0.99))
        else:
            caps[col] = np.nan
    return caps


def clean_data(X: pd.DataFrame, caps: dict | None = None) -> pd.DataFrame:
    X = X.copy()

    if "age" in X.columns:
        X["age"] = X["age"].replace(0, np.nan)

    if caps is not None:
        for col in OUTLIER_COLS:
            if col not in X.columns or col not in caps:
                continue
            cap = caps[col]
            if not np.isnan(cap):
                X[col] = np.where(X[col] > cap, np.nan, X[col])

    X = X.replace([np.inf, -np.inf], np.nan)

    if all(c in X.columns for c in LATE_COLS):
        X["TotalTimeLate"] = X[LATE_COLS].sum(axis=1)

    return X


X = train[BASE_FEATURES].copy()
y = train[TARGET].astype(int)

params = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
    X_train_raw = X.iloc[train_idx]
    X_val_raw = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    caps = get_outlier_caps(X_train_raw)

    X_train_clean = clean_data(X_train_raw, caps)
    X_val_clean = clean_data(X_val_raw, caps)

    feature_cols = list(X_train_clean.columns)
    X_val_clean = X_val_clean.reindex(columns=feature_cols)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_clean,
        y_train,
        eval_set=[(X_val_clean, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    val_pred = model.predict_proba(X_val_clean)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    aucs.append(auc)

    if hasattr(model, "best_iteration_") and model.best_iteration_ > 0:
        best_iterations.append(model.best_iteration_)

    print(f"Fold {fold} AUC: {auc:.6f}")

mean_auc = float(np.mean(aucs))
std_auc = float(np.std(aucs))
print(f"CV AUC: {mean_auc:.6f} +/- {std_auc:.6f}")

final_caps = get_outlier_caps(X)
X_train_final = clean_data(X, final_caps)
X_test_final = clean_data(test.reindex(columns=BASE_FEATURES).copy(), final_caps)

feature_cols = list(X_train_final.columns)
X_test_final = X_test_final.reindex(columns=feature_cols)

final_n_estimators = (
    int(np.mean(best_iterations)) if best_iterations else params["n_estimators"]
)
final_n_estimators = max(1, min(final_n_estimators, params["n_estimators"]))

final_model = lgb.LGBMClassifier(**{**params, "n_estimators": final_n_estimators})
final_model.fit(X_train_final, y)

test_pred = final_model.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame(
    {
        "id": np.arange(len(test)),
        "SeriousDlqin2yrs": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv")
