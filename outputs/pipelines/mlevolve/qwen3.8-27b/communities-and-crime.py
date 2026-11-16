import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ============================================================
# Setup
# ============================================================
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

TARGET = "ViolentCrimesPerPop"
MISSING_INDICATOR = -99999
SEED = 42
# Identifier / non-predictive columns (unique IDs, CV fold)
DROP_COLS = ["communityname", "community", "fold"]

# ============================================================
# 1. Load data
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")
print(f"[data] train shape = {train_df.shape}, test shape = {test_df.shape}")


# ============================================================
# 2. Feature construction (identical logic for train & test)
# ============================================================
def build_features(df):
    cols = [c for c in df.columns if c != TARGET and c not in DROP_COLS]
    X = df[cols].copy()
    X = X.replace(MISSING_INDICATOR, np.nan)  # missing indicator -> NaN
    X = X.apply(pd.to_numeric, errors="coerce")  # ensure numeric
    return X


X_all = build_features(train_df)
X_test = build_features(test_df)

# Align test columns to the exact train feature set
for c in X_all.columns:
    if c not in X_test.columns:
        X_test[c] = np.nan
X_test = X_test[X_all.columns]

# Target (drop rows with missing target)
y_all = train_df[TARGET].astype(float)
mask = y_all.notna()
X_all = X_all[mask]
y_all = y_all[mask].values

print(
    f"[data] n_features = {X_all.shape[1]}, usable train rows = {len(X_all)}, "
    f"NaN cells = {int(X_all.isna().sum().sum())}"
)
print(
    f"[data] target mean = {y_all.mean():.2f}, std = {y_all.std():.2f}, "
    f"min = {y_all.min():.2f}, max = {y_all.max():.2f}"
)

# ============================================================
# 3. Validation split (random, reproducible)
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.15, random_state=SEED
)
print(f"[split] train = {X_train.shape[0]}, val = {X_val.shape[0]}")

# ============================================================
# 4. LightGBM model
# ============================================================
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "lambda_l1": 0.5,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": SEED,
}

train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

model = lgb.train(
    params,
    train_set,
    num_boost_round=3000,
    valid_sets=[train_set, val_set],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)
best_iter = model.best_iteration if model.best_iteration > 0 else 200
print(f"[train] best_iteration = {best_iter}")


# ============================================================
# 5. Validation metric (official RMSE, same post-processing as test)
# ============================================================
def postprocess(p):
    """Post-processing: clip to non-negative (crimes per pop >= 0)."""
    return np.clip(np.asarray(p, dtype=float), 0.0, None)


val_pred = postprocess(model.predict(X_val, num_iteration=best_iter))
val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
print(f"[val] RMSE = {val_rmse:.6f}")

# ============================================================
# 6. Retrain on full data (same rounds) for the final test model
# ============================================================
full_set = lgb.Dataset(X_all, label=y_all)
model_full = lgb.train(
    params,
    full_set,
    num_boost_round=best_iter,
    callbacks=[lgb.log_evaluation(period=-1)],
)

# ============================================================
# 7. Test prediction (identical processing to validation)
# ============================================================
test_pred = postprocess(model_full.predict(X_test))
test_pred = np.nan_to_num(test_pred, nan=float(np.nanmean(y_all)))

# ============================================================
# 8. Save submission
# ============================================================
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_df)),
        "ViolentCrimesPerPop": test_pred,
    }
)
submission.to_csv("./submission/submission.csv", index=False)
print(
    f"[output] submission saved: {len(submission)} rows -> ./submission/submission.csv"
)

# ============================================================
# 9. Final reported score
# ============================================================
print(f"Final Validation Score: {val_rmse}")
