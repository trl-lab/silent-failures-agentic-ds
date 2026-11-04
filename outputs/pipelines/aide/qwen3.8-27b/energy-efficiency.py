import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

TARGET = "heating_load"
ID_COL = "id"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

feature_cols = [
    col for col in train.columns if col not in {TARGET, ID_COL} and col in test.columns
]

X_train = train[feature_cols].copy()
y_train = train[TARGET].copy()
X_test = test[feature_cols].copy()

for col in feature_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
    X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

y_train = pd.to_numeric(y_train, errors="coerce")
valid_mask = y_train.notna()
X_train = X_train.loc[valid_mask].reset_index(drop=True)
y_train = y_train.loc[valid_mask].reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "seed": 42,
    "verbosity": -1,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmses = []
best_iterations = []

for train_idx, val_idx in kf.split(X_train):
    X_tr = X_train.iloc[train_idx]
    y_tr = y_train.iloc[train_idx]
    X_va = X_train.iloc[val_idx]
    y_va = y_train.iloc[val_idx]

    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=feature_cols)
    X_va_imp = pd.DataFrame(imputer.transform(X_va), columns=feature_cols)

    train_set = lgb.Dataset(X_tr_imp, label=y_tr, free_raw_data=False)
    valid_set = lgb.Dataset(
        X_va_imp, label=y_va, reference=train_set, free_raw_data=False
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    best_iter = max(1, int(model.best_iteration))
    best_iterations.append(best_iter)

    val_pred = model.predict(X_va_imp, num_iteration=best_iter)
    cv_rmses.append(float(np.sqrt(mean_squared_error(y_va, val_pred))))

cv_rmse = float(np.mean(cv_rmses))

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

final_rounds = max(1, int(np.mean(best_iterations)))
final_model = lgb.train(
    params,
    lgb.Dataset(X_train_imp, label=y_train, free_raw_data=False),
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_test_imp)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "heating_load": test_pred,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

print(f"RMSE: {cv_rmse:.6f}")
