import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error

SEED = 42
np.random.seed(SEED)

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "y"
EXCLUDE = {TARGET, "id", "index", "idx"}

features = [c for c in train.columns if c not in EXCLUDE]
features = [c for c in features if c in test.columns]

if not features:
    features = [c for c in train.columns if c != TARGET]

X_train_all = train[features].copy()
y_train_all = train[TARGET].copy()
X_test = test[features].copy()


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


X_train_all = clean_numeric(X_train_all)
X_test = clean_numeric(X_test)
y_train_all = pd.to_numeric(y_train_all, errors="coerce")

valid_mask = y_train_all.notna()
X_train_all = X_train_all.loc[valid_mask].reset_index(drop=True)
y_train_all = y_train_all.loc[valid_mask].reset_index(drop=True)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": SEED,
}

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rmses = []
best_rounds = []
MAX_ROUNDS = 5000
EARLY_STOPPING = 50

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_all), start=1):
    X_tr = X_train_all.iloc[train_idx]
    y_tr = y_train_all.iloc[train_idx]
    X_val = X_train_all.iloc[val_idx]
    y_val = y_train_all.iloc[val_idx]

    X_inner_tr, X_inner_val, y_inner_tr, y_inner_val = train_test_split(
        X_tr,
        y_tr,
        test_size=0.2,
        random_state=SEED + fold,
    )

    dtrain = lgb.Dataset(
        X_inner_tr,
        label=y_inner_tr,
        feature_name=features,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_inner_val,
        label=y_inner_val,
        feature_name=features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dvalid],
        valid_names=["inner_val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(-1)],
    )

    best_iteration = getattr(model, "best_iteration", 0)
    if not best_iteration or best_iteration <= 0:
        best_iteration = MAX_ROUNDS

    best_rounds.append(best_iteration)

    val_pred = model.predict(X_val, num_iteration=best_iteration)
    fold_rmse = root_mean_squared_error(y_val, val_pred)
    cv_rmses.append(fold_rmse)

    print(f"Fold {fold} RMSE: {fold_rmse:.6f} | best_iteration: {best_iteration}")

cv_mean = float(np.mean(cv_rmses))
cv_std = float(np.std(cv_rmses))
print(f"5-Fold CV RMSE: {cv_mean:.6f} +/- {cv_std:.6f}")

final_rounds = int(np.median(best_rounds)) + 1
if final_rounds < 1:
    final_rounds = 100

dtrain_full = lgb.Dataset(
    X_train_all,
    label=y_train_all,
    feature_name=features,
    free_raw_data=False,
)

final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
)

test_pred = final_model.predict(X_test)

submission = pd.DataFrame(
    {
        "id": np.arange(len(X_test), dtype=int),
        "y": test_pred,
    }
)

submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
