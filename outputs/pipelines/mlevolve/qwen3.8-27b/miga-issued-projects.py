import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
train_full = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_full.shape}")
print(f"Test shape: {test.shape}")

target_col = "Gross Issued Amount (US$)"
id_col = "id"

# Separate target
y_full = train_full[target_col].values
train_full = train_full.drop(columns=[target_col])

# Identify feature columns
feature_cols = [c for c in train_full.columns if c != id_col]

# Classify columns
numeric_cols = (
    train_full[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
)
categorical_cols = (
    train_full[feature_cols]
    .select_dtypes(include=["object", "category"])
    .columns.tolist()
)

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# ============================================================
# 2. CREATE TRAIN/VALIDATION SPLIT (BEFORE any feature engineering)
# ============================================================
train_data, val_data = train_test_split(train_full, test_size=0.2, random_state=42)
y_train, y_val = train_test_split(y_full, test_size=0.2, random_state=42)

print(f"Train split: {len(train_data)}, Val split: {len(val_data)}, Test: {len(test)}")


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def kfold_target_encode(train_df, val_df, test_df, col, y_train, n_splits=5, smoothing=50):
    """K-Fold CV-based target encoding to prevent leakage."""
    global_mean = y_train.mean()
    nunique = train_df[col].nunique()
    if nunique > 50:
        smoothing = 50
    else:
        smoothing = 50

    # For train: use K-Fold to compute target stats on other folds
    train_df[f"{col}_te"] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]
        fold_target = y_train[train_idx]
        agg = pd.DataFrame({"cat": fold_train[col].values, "target": fold_target})
        stats = agg.groupby("cat")["target"].agg(["mean", "count"])
        smooth_enc = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        train_df.loc[train_df.index[val_idx], f"{col}_te"] = (
            train_df.iloc[val_idx][col].map(smooth_enc).fillna(global_mean)
        )
    train_df[f"{col}_te"] = train_df[f"{col}_te"].fillna(global_mean)

    # For val and test: use all training data
    agg = pd.DataFrame({"cat": train_df[col].values, "target": y_train})
    stats = agg.groupby("cat")["target"].agg(["mean", "count"])
    smooth_enc = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    val_df[f"{col}_te"] = val_df[col].map(smooth_enc).fillna(global_mean)
    test_df[f"{col}_te"] = test_df[col].map(smooth_enc).fillna(global_mean)


def kfold_freq_encode(train_df, val_df, test_df, col, n_splits=5):
    """K-Fold CV-based frequency encoding."""
    train_df[f"{col}_freq"] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]
        freq = fold_train[col].value_counts(normalize=True)
        train_df.loc[train_df.index[val_idx], f"{col}_freq"] = (
            train_df.iloc[val_idx][col].map(freq).fillna(0)
        )
    train_df[f"{col}_freq"] = train_df[f"{col}_freq"].fillna(0)

    # For val and test: use all training data
    freq = train_df[col].value_counts(normalize=True)
    val_df[f"{col}_freq"] = val_df[col].map(freq).fillna(0)
    test_df[f"{col}_freq"] = test_df[col].map(freq).fillna(0)


def engineer_features(train_df, val_df, test_df, y_train):
    """Complete feature engineering pipeline. Fit on train, transform on val/test."""
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    global_mean = y_train.mean()

    # --- Missing values ---
    for col in numeric_cols:
        if col in train_df.columns:
            med = train_df[col].median()
            train_df[col] = train_df[col].fillna(med)
            val_df[col] = val_df[col].fillna(med)
            test_df[col] = test_df[col].fillna(med)

    for col in categorical_cols:
        for df in [train_df, val_df, test_df]:
            if col in df.columns:
                df[col] = df[col].fillna("MISSING")

    # --- Log transforms ---
    for col in numeric_cols:
        if col in train_df.columns:
            if train_df[col].min() > 0:
                train_df[f"{col}_log"] = np.log1p(train_df[col])
                val_df[f"{col}_log"] = np.log1p(val_df[col])
                test_df[f"{col}_log"] = np.log1p(test_df[col])
            else:
                train_df[f"{col}_log"] = np.sign(train_df[col]) * np.log1p(
                    np.abs(train_df[col])
                )
                val_df[f"{col}_log"] = np.sign(val_df[col]) * np.log1p(
                    np.abs(val_df[col])
                )
                test_df[f"{col}_log"] = np.sign(test_df[col]) * np.log1p(
                    np.abs(test_df[col])
                )

    # --- K-Fold CV Target encoding (smoothed) ---
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
        kfold_target_encode(train_df, val_df, test_df, col, y_train, n_splits=5, smoothing=50)

        # K-Fold CV Frequency encoding
        kfold_freq_encode(train_df, val_df, test_df, col, n_splits=5)

    # --- Label encode categoricals ---
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))
        val_df[col] = (
            val_df[col]
            .astype(str)
            .apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        )
        test_df[col] = (
            test_df[col]
            .astype(str)
            .apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        )

    # --- Drop id column ---
    for df in [train_df, val_df, test_df]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    return train_df, val_df, test_df


train_fe, val_fe, test_fe = engineer_features(train_data, val_data, test, y_train)

# ============================================================
# 4. FINALIZE FEATURES
# ============================================================
for df in [train_fe, val_fe, test_fe]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)

# Align columns
all_cols = train_fe.columns.tolist()
val_fe = val_fe.reindex(columns=all_cols, fill_value=0)
test_fe = test_fe.reindex(columns=all_cols, fill_value=0)

print(f"Final feature count: {len(all_cols)}")
print(f"Train features shape: {train_fe.shape}")
print(f"Val features shape: {val_fe.shape}")
print(f"Test features shape: {test_fe.shape}")

# ============================================================
# 5. MODEL DESIGN: 3-Model Ensemble
# ============================================================
# Use log-transformed target for training (handles right-skew)
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Model 1: LightGBM with increased capacity to address underfitting
lgb_params_1 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "max_depth": 6,
    "learning_rate": 0.08,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 2,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_split_gain": 0.01,
    "verbosity": -1,
    "random_state": 42,
}

# Model 2: XGBoost with increased capacity
xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "learning_rate": 0.08,
    "random_state": 42,
    "verbosity": 0,
}

# Model 3: Second LightGBM GBDT with different seed and params for diversity
lgb_params_3 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 20,
    "max_depth": 5,
    "learning_rate": 0.08,
    "min_child_samples": 10,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.9,
    "bagging_freq": 2,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_split_gain": 0.01,
    "verbosity": -1,
    "random_state": 123,
}

# ============================================================
# 6. TRAIN ENSEMBLE MODELS WITH EARLY STOPPING
# ============================================================
# Train Model 1: LightGBM
model_1 = lgb.LGBMRegressor(**lgb_params_1, n_estimators=3000)
model_1.fit(
    train_fe,
    y_train_log,
    eval_set=[(val_fe, y_val_log)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
best_iter_1 = model_1.best_iteration_
print(f"Model 1 (LGB) best iteration: {best_iter_1}, val RMSE (log): {model_1.best_score_['valid_0']['rmse']:.4f}")

# Train Model 2: XGBoost
model_2 = xgb.XGBRegressor(**xgb_params, n_estimators=3000, early_stopping_rounds=100)
model_2.fit(
    train_fe,
    y_train_log,
    eval_set=[(val_fe, y_val_log)],
    verbose=False,
)
best_iter_2 = model_2.best_iteration
print(f"Model 2 (XGB) best iteration: {best_iter_2}")

# Train Model 3: LightGBM GBDT (different seed/params for diversity)
model_3 = lgb.LGBMRegressor(**lgb_params_3, n_estimators=3000)
model_3.fit(
    train_fe,
    y_train_log,
    eval_set=[(val_fe, y_val_log)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ],
)
best_iter_3 = model_3.best_iteration_
print(f"Model 3 (LGB-GBDT) best iteration: {best_iter_3}, val RMSE (log): {model_3.best_score_['valid_0']['rmse']:.4f}")

# ============================================================
# 7. VALIDATION PREDICTIONS (ensemble, original scale RMSE)
# ============================================================
val_pred_log_1 = model_1.predict(val_fe, num_iteration=best_iter_1)
val_pred_log_2 = model_2.predict(val_fe, iteration_range=(0, best_iter_2 + 1))
val_pred_log_3 = model_3.predict(val_fe, num_iteration=best_iter_3)

# Average in log-space
val_pred_log = (val_pred_log_1 + val_pred_log_2 + val_pred_log_3) / 3.0
val_pred = np.expm1(val_pred_log)
val_pred = np.maximum(val_pred, 0)

val_rmse = root_mean_squared_error(y_val, val_pred)
print(f"Validation RMSE (original scale): {val_rmse:.4f}")

# Training RMSE for overfitting check
train_pred_log_1 = model_1.predict(train_fe, num_iteration=best_iter_1)
train_pred_log_2 = model_2.predict(train_fe, iteration_range=(0, best_iter_2 + 1))
train_pred_log_3 = model_3.predict(train_fe, num_iteration=best_iter_3)
train_pred_log = (train_pred_log_1 + train_pred_log_2 + train_pred_log_3) / 3.0
train_pred = np.expm1(train_pred_log)
train_pred = np.maximum(train_pred, 0)
train_rmse = root_mean_squared_error(y_train, train_pred)
print(f"Training RMSE (original scale): {train_rmse:.4f}")
print(f"Overfitting ratio (val/train): {val_rmse / max(train_rmse, 1e-8):.2f}")

# ============================================================
# 8. TEST PREDICTIONS (ensemble, identical processing to validation)
# ============================================================
test_pred_log_1 = model_1.predict(test_fe, num_iteration=best_iter_1)
test_pred_log_2 = model_2.predict(test_fe, iteration_range=(0, best_iter_2 + 1))
test_pred_log_3 = model_3.predict(test_fe, num_iteration=best_iter_3)

# Average in log-space
test_pred_log = (test_pred_log_1 + test_pred_log_2 + test_pred_log_3) / 3.0
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(test_pred, 0)

print(
    f"Test predictions: mean={test_pred.mean():.2f}, std={test_pred.std():.2f}, min={test_pred.min():.2f}, max={test_pred.max():.2f}"
)

# ============================================================
# 9. SAVE SUBMISSION
# ============================================================
test_ids = (
    pd.read_parquet("./input/test.parquet")["id"].values
    if "id" in pd.read_parquet("./input/test.parquet").columns
    else np.arange(len(test))
)

submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Gross Issued Amount (US$)": test_pred}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(submission.head())

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================
importance = model_1.feature_importances_
feat_imp = pd.Series(importance, index=train_fe.columns).sort_values(ascending=False)
print(f"\nTop 10 features:")
for feat, imp in feat_imp.head(10).items():
    print(f"  {feat}: {imp}")

# ============================================================
# 11. FINAL SCORE
# ============================================================
score = val_rmse
print(f"Final Validation Score: {score:.4f}")
