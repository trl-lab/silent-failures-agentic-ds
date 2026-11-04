import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target = "strength"
base_features = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]


def add_features(df):
    df = df.copy()
    df["binder"] = df["cement"] + df["blast_furnace_slag"] + df["fly_ash"]
    df["water_to_binder"] = df["water"] / (df["binder"] + 1e-6)
    df["water_to_cement"] = df["water"] / (df["cement"] + 1e-6)
    df["log_age"] = np.log1p(df["age"])
    df["sqrt_age"] = np.sqrt(df["age"])
    df["agg_ratio"] = df["fine_aggregate"] / (df["coarse_aggregate"] + 1e-6)
    df["slag_cement_ratio"] = df["blast_furnace_slag"] / (df["cement"] + 1e-6)
    df["ash_cement_ratio"] = df["fly_ash"] / (df["cement"] + 1e-6)
    return df


train = add_features(train)
test = add_features(test)

feature_cols = [c for c in train.columns if c != target]

# Clean non-finite values and impute missing with train medians
train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan)
test[feature_cols] = test[feature_cols].replace([np.inf, -np.inf], np.nan)
medians = train[feature_cols].median()
train[feature_cols] = train[feature_cols].fillna(medians)
test[feature_cols] = test[feature_cols].fillna(medians)

X = train[feature_cols].values
y = train[target].values
X_test = test[feature_cols].values

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))
best_iters = []

params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
    "verbosity": 0,
}

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    preds = model.predict(dval)
    oof_preds[va_idx] = preds
    best_iters.append(model.best_iteration)
    fold_rmse = np.sqrt(mean_squared_error(y_va, preds))
    print(f"Fold {fold}: best_iter={model.best_iteration}, RMSE={fold_rmse:.4f}")

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\n5-Fold CV RMSE: {cv_rmse:.4f}")

# Final model refit on all training data
best_iter = int(np.median(best_iters))
dtrain_full = xgb.DMatrix(X, label=y)
final_model = xgb.train(
    params, dtrain_full, num_boost_round=best_iter, verbose_eval=False
)

test_preds = final_model.predict(xgb.DMatrix(X_test))
test_preds = np.clip(test_preds, 0, None)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": range(len(test_preds)), "strength": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print(
    f"Submission saved with {len(submission)} rows. Mean strength: {test_preds.mean():.2f} MPa"
)
