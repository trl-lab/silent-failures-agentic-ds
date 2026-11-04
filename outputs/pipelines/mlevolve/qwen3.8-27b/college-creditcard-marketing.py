import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)

# ------------------------------------------------------------------ #
# 1. LOAD DATA                                                       #
# ------------------------------------------------------------------ #
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "PAYMENTS_BY_ISSUER"

# Drop rows with missing target so the model can train cleanly
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

feature_cols = [c for c in train.columns if c != TARGET]

print(f"Train: {train.shape}  Test: {test.shape}")
print(f"Num features: {len(feature_cols)}")
print(
    f"Target  -> mean={train[TARGET].mean():.4f}  std={train[TARGET].std():.4f}  "
    f"min={train[TARGET].min():.4f}  max={train[TARGET].max():.4f}"
)

# Make sure every feature column exists in test
for col in feature_cols:
    if col not in test.columns:
        test[col] = np.nan

# ------------------------------------------------------------------ #
# 2. PREPROCESS – convert every feature to a numeric column         #
# ------------------------------------------------------------------ #
for col in feature_cols:
    t_dtype = train[col].dtype
    if t_dtype == "bool":
        train[col] = train[col].astype(int)
        test[col] = test[col].astype(int) if col in test.columns else np.nan
    elif "datetime" in str(t_dtype):
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = (
            pd.to_numeric(test[col], errors="coerce") if col in test.columns else np.nan
        )
    elif t_dtype in ("object", "category") or str(t_dtype) == "category":
        # Build a mapping from the combined train+test unique values
        combined = pd.concat(
            [
                train[col].dropna(),
                (
                    test[col].dropna()
                    if col in test.columns
                    else pd.Series([], dtype="object")
                ),
            ],
            axis=0,
        ).astype(str)
        uniques = sorted(combined.unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        sentinel = len(uniques)  # for unseen / NaN values
        train[col] = train[col].astype(str).map(mapping).fillna(sentinel).astype(int)
        test[col] = (
            test[col].astype(str).map(mapping).fillna(sentinel).astype(int)
            if col in test.columns
            else pd.Series([sentinel] * len(test))
        )
    else:
        # numeric – just ensure float64
        train[col] = pd.to_numeric(train[col], errors="coerce")
        if col in test.columns:
            test[col] = pd.to_numeric(test[col], errors="coerce")
        else:
            test[col] = np.nan

# Drop features that are entirely NaN (no signal)
train_missing = train[feature_cols].isna().mean()
keep_cols = [c for c in feature_cols if train_missing[c] < 1.0]
if len(keep_cols) < len(feature_cols):
    print(f"Dropped {len(feature_cols) - len(keep_cols)} all-NaN feature(s)")
feature_cols = keep_cols

# ------------------------------------------------------------------ #
# 3. VALIDATION SPLIT                                                #
# ------------------------------------------------------------------ #
train_part, val_part = train_test_split(train, test_size=0.20, random_state=42)
X_tr, y_tr = train_part[feature_cols], train_part[TARGET]
X_va, y_va = val_part[feature_cols], val_part[TARGET]
X_te = test[feature_cols]

print(f"Train split: {X_tr.shape}   Val split: {X_va.shape}")

# ------------------------------------------------------------------ #
# 4. TRAIN LIGHTGBM (early-stopping on validation)                  #
# ------------------------------------------------------------------ #
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    min_child_samples=15,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

model = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=3000)
model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
)

best_iter = model.best_iteration_
print(f"Early-stopping best iteration: {best_iter}")

# ------------------------------------------------------------------ #
# 5. VALIDATION SCORE  (official metric: RMSE)                      #
# ------------------------------------------------------------------ #
val_pred = model.predict(X_va)
val_rmse = root_mean_squared_error(y_va, val_pred)
print(f"Validation RMSE: {val_rmse:.6f}")

# ------------------------------------------------------------------ #
# 6. RETRAIN ON FULL TRAINING SET                                    #
# ------------------------------------------------------------------ #
n_trees = max(best_iter + 1, 10)
print(f"Retraining on full train set ({len(train)} rows) with {n_trees} trees")

full_model = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=n_trees)
full_model.fit(train[feature_cols], train[TARGET])

# ------------------------------------------------------------------ #
# 7. PREDICT TEST & SAVE SUBMISSION                                 #
# ------------------------------------------------------------------ #
test_pred = full_model.predict(X_te)

submission = pd.DataFrame({"id": np.arange(len(test)), "PAYMENTS_BY_ISSUER": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission written to ./submission/submission.csv  ({len(submission)} rows)")

# ------------------------------------------------------------------ #
# 8. FINAL SCORE                                                    #
# ------------------------------------------------------------------ #
print(f"Final Validation Score: {val_rmse:.6f}")
