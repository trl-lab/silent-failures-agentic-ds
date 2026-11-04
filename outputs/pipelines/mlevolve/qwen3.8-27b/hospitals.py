import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# ============================================================
# 2. IDENTIFY FEATURE TYPES
# ============================================================
target_col = "BEDS"
id_col = "id" if "id" in train.columns else None

y_full = train[target_col].copy()
train_features = train.drop(columns=[target_col] + ([id_col] if id_col else []))
test_drop_cols = [c for c in [target_col] + ([id_col] if id_col else []) if c in test.columns]
test_features = test.drop(columns=test_drop_cols)

# Remove high-cardinality identifier features - these are unique IDs not useful categories
high_card_drop = ['NAME', 'ADDRESS', 'TELEPHONE', 'ZIP4', 'WEBSITE', 'ALT_NAME', 'SOURCEDATE', 'VAL_DATE', 'CITY', 'STATE_ID']

# Drop high-cardinality identifier columns from feature DataFrames
drop_from_train = [c for c in high_card_drop if c in train_features.columns]
drop_from_test = [c for c in high_card_drop if c in test_features.columns]
train_features = train_features.drop(columns=drop_from_train)
test_features = test_features.drop(columns=drop_from_test)

cat_cols = train_features.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c not in high_card_drop]
num_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()

for col in num_cols:
    if train_features[col].nunique() < 20 and train_features[col].dtype in [
        "int64",
        "int32",
    ]:
        cat_cols.append(col)
        num_cols.remove(col)

print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"Numerical features ({len(num_cols)}): {num_cols}")

# ============================================================
# 3. TRAIN/VAL SPLIT (BEFORE any processing)
# ============================================================
train_idx, val_idx = train_test_split(
    range(len(train_features)), test_size=0.15, random_state=42
)
X_train_raw = train_features.iloc[train_idx].copy()
X_val_raw = train_features.iloc[val_idx].copy()
y_train = y_full.iloc[train_idx].copy()
y_val = y_full.iloc[val_idx].copy()

# ============================================================
# 4. FEATURE ENGINEERING (fit on train portion only)
# ============================================================


class FeatureEngineer:
    def __init__(self):
        self.fitted = False
        self.medians = {}
        self.modes = {}
        self.global_mean = 0
        self.target_encs = {}
        self.freqs = {}
        self.grp_stats = {}

    def fit_transform(self, X_fit, y_fit):
        self.fitted = True
        X = X_fit.copy()

        # Handle missing values
        for col in num_cols:
            if col in X.columns:
                self.medians[col] = X[col].median()
                X[col + "_missing"] = X[col].isnull().astype(int)
                X[col] = X[col].fillna(self.medians[col])

        for col in cat_cols:
            if col in X.columns:
                mode_val = X[col].mode()[0] if not X[col].mode().empty else "UNKNOWN"
                self.modes[col] = mode_val
                X[col] = X[col].fillna(mode_val)

        # Target encoding with Bayesian smoothing
        self.global_mean = y_fit.mean()
        m = 10
        for col in cat_cols:
            if col not in X.columns:
                continue
            agg = y_fit.groupby(X[col]).agg(["mean", "count"])
            smoothed = (agg["count"] * agg["mean"] + m * self.global_mean) / (
                agg["count"] + m
            )
            self.target_encs[col] = smoothed
            X[col + "_target_enc"] = X[col].map(smoothed).fillna(self.global_mean)

            counts = X[col].value_counts()
            self.freqs[col] = counts
            X[col + "_freq"] = X[col].map(counts).fillna(0)

        # Group-level aggregate features
        for cat in cat_cols[:5]:
            if cat not in X.columns:
                continue
            for num in num_cols[:8]:
                if num not in X.columns:
                    continue
                grp_stats = X.groupby(cat)[num].agg(["mean", "std", "median"])
                self.grp_stats[(cat, num)] = grp_stats
                X[f"{num}_by_{cat}_mean"] = (
                    X[cat].map(grp_stats["mean"]).fillna(X[num].mean())
                )
                X[f"{num}_by_{cat}_std"] = X[cat].map(grp_stats["std"]).fillna(0)
                X[f"{num}_by_{cat}_med"] = (
                    X[cat].map(grp_stats["median"]).fillna(X[num].median())
                )

        # Drop original categoricals
        drop_cols = [c for c in cat_cols if c in X.columns]
        X = X.drop(columns=drop_cols)

        return X

    def transform(self, X_transform):
        X = X_transform.copy()

        for col in num_cols:
            if col in X.columns:
                X[col + "_missing"] = X[col].isnull().astype(int)
                X[col] = X[col].fillna(self.medians.get(col, 0))

        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.modes.get(col, "UNKNOWN"))

        for col in cat_cols:
            if col not in X.columns:
                continue
            if col in self.target_encs:
                X[col + "_target_enc"] = (
                    X[col].map(self.target_encs[col]).fillna(self.global_mean)
                )
            else:
                X[col + "_target_enc"] = self.global_mean
            if col in self.freqs:
                X[col + "_freq"] = X[col].map(self.freqs[col]).fillna(0)
            else:
                X[col + "_freq"] = 0

        for cat in cat_cols[:5]:
            if cat not in X.columns:
                continue
            for num in num_cols[:8]:
                if num not in X.columns:
                    continue
                key = (cat, num)
                if key in self.grp_stats:
                    grp_stats = self.grp_stats[key]
                    X[f"{num}_by_{cat}_mean"] = (
                        X[cat].map(grp_stats["mean"]).fillna(X[num].mean())
                    )
                    X[f"{num}_by_{cat}_std"] = X[cat].map(grp_stats["std"]).fillna(0)
                    X[f"{num}_by_{cat}_med"] = (
                        X[cat].map(grp_stats["median"]).fillna(X[num].median())
                    )

        drop_cols = [c for c in cat_cols if c in X.columns]
        X = X.drop(columns=drop_cols)

        return X


print("\nEngineering features...")
fe = FeatureEngineer()
X_train = fe.fit_transform(X_train_raw, y_train)
X_val = fe.transform(X_val_raw)
X_test = fe.transform(test_features)

# Ensure same columns
all_cols = X_train.columns.tolist()
for col in all_cols:
    if col not in X_val.columns:
        X_val[col] = 0
    if col not in X_test.columns:
        X_test[col] = 0

X_val = X_val[all_cols]
X_test = X_test[all_cols]

print(f"Final feature count: {len(all_cols)}")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ============================================================
# 5. USE RAW TARGET (no log1p - BEDS values are 0-10, log1p compresses signal)
# ============================================================
y_train_log = y_train
y_val_log = y_val

# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================

# --- Model 1: LightGBM with Huber loss ---
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "nthread": -1,
    "seed": 42,
    "verbose": -1,
}

lgb_model = lgb.LGBMRegressor(**lgb_params)

# --- Model 2: XGBoost with squarederror on log1p target ---
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.65,
    "min_child_weight": 15,
    "reg_alpha": 0.3,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "nthread": -1,
    "seed": 123,
    "early_stopping_rounds": 200,
}

xgb_model = xgb.XGBRegressor(**xgb_params)

# ============================================================
# 7. TRAINING WITH EARLY STOPPING
# ============================================================
print("\n--- Training LightGBM (regression on raw target) ---")
lgb_model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
lgb_best_iter = lgb_model.best_iteration_
print(f"  LightGBM best iteration: {lgb_best_iter}")

print("\n--- Training XGBoost (squarederror on raw target) ---")
xgb_model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    verbose=False,
)
xgb_best_iter = xgb_model.best_iteration
print(f"  XGBoost best iteration: {xgb_best_iter}")

# ============================================================
# 8. PREDICTIONS VIA MODEL INFERENCE
# ============================================================

# --- Validation predictions ---
val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_best_iter)
val_pred_xgb = xgb_model.predict(X_val, iteration_range=(0, xgb_best_iter + 1))

val_pred = 0.5 * val_pred_lgb + 0.5 * val_pred_xgb
val_pred = np.clip(val_pred, 0, None)

# --- Test predictions ---
test_pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_best_iter)
test_pred_xgb = xgb_model.predict(X_test, iteration_range=(0, xgb_best_iter + 1))

test_pred = 0.5 * test_pred_lgb + 0.5 * test_pred_xgb
test_pred = np.clip(test_pred, 0, None)

# ============================================================
# 9. VALIDATION METRIC: RMSE on original scale (official metric)
# ============================================================
val_rmse = root_mean_squared_error(y_val, val_pred)

val_pred_lgb_orig = np.clip(val_pred_lgb, 0, None)
val_pred_xgb_orig = np.clip(val_pred_xgb, 0, None)
lgb_rmse = root_mean_squared_error(y_val, val_pred_lgb_orig)
xgb_rmse = root_mean_squared_error(y_val, val_pred_xgb_orig)

print(f"\n--- Validation Results (RMSE on original scale) ---")
print(f"  LightGBM alone:  {lgb_rmse:.4f}")
print(f"  XGBoost alone:   {xgb_rmse:.4f}")
print(f"  Ensemble (avg):  {val_rmse:.4f}")

# ============================================================
# 10. GENERATE SUBMISSION
# ============================================================
if id_col and id_col in test.columns:
    submission_ids = test[id_col]
else:
    submission_ids = range(len(test_pred))

submission = pd.DataFrame({"id": submission_ids, "BEDS": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(submission.head())

# ============================================================
# FINAL OUTPUT
# ============================================================
score = val_rmse
print(f"Final Validation Score: {score:.4f}")
