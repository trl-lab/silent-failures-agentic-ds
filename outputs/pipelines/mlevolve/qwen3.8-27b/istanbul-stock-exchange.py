import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# ============================================================
# DATA LOADING
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Handle duplicate ISE columns
if train.columns.tolist().count("ISE") > 1:
    cols = train.columns.tolist()
    first_ise = cols.index("ISE")
    second_ise = cols.index("ISE", first_ise + 1)
    train.columns = [
        (
            c
            if i not in [first_ise, second_ise]
            else ("ISE" if i == first_ise else "ISE_dup")
        )
        for i, c in enumerate(cols)
    ]
    if test.columns.tolist().count("ISE") > 1:
        cols_t = test.columns.tolist()
        first_ise_t = cols_t.index("ISE")
        second_ise_t = cols_t.index("ISE", first_ise_t + 1)
        test.columns = [
            (
                c
                if i not in [first_ise_t, second_ise_t]
                else ("ISE" if i == first_ise_t else "ISE_dup")
            )
            for i, c in enumerate(cols_t)
        ]

# Check if ISE_dup is identical to ISE
if "ISE_dup" in train.columns:
    corr = train["ISE"].corr(train["ISE_dup"])
    if abs(corr - 1.0) < 1e-10:
        train = train.drop(columns=["ISE_dup"])
        if "ISE_dup" in test.columns:
            test = test.drop(columns=["ISE_dup"])
        print("Dropped duplicate ISE column (identical to ISE)")

# Parse dates and sort
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
train = train.sort_values("date").reset_index(drop=True)
test = test.sort_values("date").reset_index(drop=True)

print(f"Train date range: {train['date'].min()} to {train['date'].max()}")
print(f"Test date range: {test['date'].min()} to {test['date'].max()}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
target_col = "ISE"

def engineer_features(df):
    """Apply all feature engineering to a single dataframe (no cross-contamination)."""
    df = df.copy()

    # If ISE is not in df (test set), create a zero placeholder for feature computation
    has_ise = "ISE" in df.columns
    if not has_ise:
        df["ISE"] = 0.0

    # 1. Calendar features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # 2. Lag features for all numeric columns (exclude target ISE to prevent leakage)
    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype in ["float64", "float32", "int64", "int32"]
        and c not in ["is_train", "ISE"]
    ]

    for col in numeric_cols:
        for lag in [1, 2, 3, 5, 10]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for window in [5, 10, 20, 50]:
            df[f"{col}_rollmean_{window}"] = (
                df[col].rolling(window, min_periods=1).mean()
            )
            df[f"{col}_rollstd_{window}"] = (
                df[col].rolling(window, min_periods=2).std()
            )
            df[f"{col}_rollmin_{window}"] = (
                df[col].rolling(window, min_periods=1).min()
            )
            df[f"{col}_rollmax_{window}"] = (
                df[col].rolling(window, min_periods=1).max()
            )
        for window in [5, 10, 20]:
            df[f"{col}_momentum_{window}"] = df[col] - df[col].shift(window)
        df[f"{col}_vol_ratio"] = df[col].rolling(5, min_periods=2).std() / (
            df[col].rolling(20, min_periods=5).std() + 1e-10
        )

    # 3. Cross-market lead-lag features (use lagged ISE to avoid target leakage)
    other_markets = [
        c
        for c in ["SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]
        if c in df.columns
    ]
    print(f"Other markets found: {other_markets}")

    for mkt in other_markets:
        for lag in [1, 2, 3, 5]:
            df[f"{mkt}_lead_{lag}"] = df[mkt].shift(lag)
        df[f"ISE_lag1_minus_{mkt}"] = df["ISE"].shift(1) - df[mkt]
        df[f"ISE_lag1_minus_{mkt}_lag1"] = (df["ISE"].shift(1) - df[mkt]).shift(1)
        df[f"ISE_lag1_ratio_{mkt}"] = df["ISE"].shift(1) / (df[mkt] + 1e-10)
        df[f"corr_ISE_{mkt}_20"] = (
            df["ISE"].rolling(20, min_periods=10).corr(df[mkt])
        )
        df[f"corr_ISE_{mkt}_50"] = (
            df["ISE"].rolling(50, min_periods=20).corr(df[mkt])
        )

    # 4. Cross-sectional rank features (use lagged ISE to avoid target leakage)
    all_markets = [m for m in ["ISE"] + other_markets if m in df.columns]
    market_returns = df[all_markets].copy()
    market_returns["ISE"] = market_returns["ISE"].shift(1)
    df["ISE_rank"] = market_returns.rank(axis=1)["ISE"]
    df["ISE_rank_pct"] = df["ISE_rank"] / len(all_markets)
    for mkt in other_markets:
        df[f"{mkt}_rank"] = market_returns.rank(axis=1)[mkt]

    # 5. Cross-market aggregate features (use lagged ISE to avoid target leakage)
    df["mean_other_markets"] = df[other_markets].mean(axis=1)
    df["std_other_markets"] = df[other_markets].std(axis=1)
    df["max_other_markets"] = df[other_markets].max(axis=1)
    df["min_other_markets"] = df[other_markets].min(axis=1)
    df["ISE_vs_mean_other"] = df["ISE"].shift(1) - df["mean_other_markets"]
    df["ISE_vs_mean_other_lag1"] = df["ISE_vs_mean_other"].shift(1)

    # 6. Interaction features (use lagged ISE to avoid target leakage)
    for mkt in other_markets[:3]:
        df[f"ISE_x_{mkt}"] = df["ISE"].shift(1) * df[mkt]
        df[f"ISE_x_{mkt}_lag1"] = df[f"ISE_x_{mkt}"].shift(1)

    # 7. Volatility regime features (use lagged ISE to avoid target leakage)
    df["ISE_abs_ret"] = df["ISE"].shift(1).abs()
    df["ISE_abs_ret_roll5"] = df["ISE_abs_ret"].rolling(5, min_periods=1).mean()
    df["ISE_abs_ret_roll20"] = (
        df["ISE_abs_ret"].rolling(20, min_periods=5).mean()
    )
    df["vol_regime"] = df["ISE_abs_ret_roll5"] / (
        df["ISE_abs_ret_roll20"] + 1e-10
    )

    # 8. Sign features (use lagged ISE to avoid target leakage)
    df["ISE_sign"] = np.sign(df["ISE"].shift(1))
    df["ISE_sign_lag1"] = df["ISE_sign"].shift(1)
    df["ISE_sign_lag2"] = df["ISE_sign"].shift(2)

    # 9. Acceleration (use lagged ISE to avoid target leakage)
    df["ISE_accel"] = df["ISE"].shift(1).diff().diff()
    df["ISE_accel_lag1"] = df["ISE_accel"].shift(1)

    # 10. Z-score features (expanding window, use lagged ISE to avoid target leakage)
    for col in ["ISE"] + other_markets:
        if col == "ISE":
            shifted = df[col].shift(1)
        else:
            shifted = df[col]
        rolling_mean = shifted.expanding(min_periods=5).mean()
        rolling_std = shifted.expanding(min_periods=5).std()
        df[f"{col}_zscore"] = (shifted - rolling_mean) / (rolling_std + 1e-10)

    # Remove the placeholder ISE column if it was added for test
    if not has_ise:
        df = df.drop(columns=["ISE"])

    return df

# Engineer features separately on train and test to prevent leakage
train_feat = engineer_features(train)
test_feat = engineer_features(test)

print(f"After feature engineering, train shape: {train_feat.shape}, test shape: {test_feat.shape}")

# ============================================================
# TEMPORAL SPLIT: last 20% of training data as validation
# ============================================================
n_train = len(train_feat)
val_size = max(1, int(n_train * 0.2))
train_split = train_feat.iloc[:-val_size].reset_index(drop=True)
val_split = train_feat.iloc[-val_size:].reset_index(drop=True)
combined_test = test_feat.reset_index(drop=True)

print(f"Train split: {len(train_split)} samples")
print(f"Validation split: {len(val_split)} samples")
print(f"Test: {len(combined_test)} samples")

# Define feature columns (exclude date, target)
exclude_cols = ["date", target_col]
feature_cols_final = sorted(
    [c for c in train_split.columns if c not in exclude_cols]
)

# Ensure all feature cols exist in test
feature_cols_final = [c for c in feature_cols_final if c in combined_test.columns]

# Remove columns that are entirely NaN in train
for col in feature_cols_final[:]:
    if train_split[col].isna().all():
        feature_cols_final.remove(col)

print(f"Final feature count: {len(feature_cols_final)}")

# Handle NaN values - fit imputer on TRAIN only
train_features = train_split[feature_cols_final].copy()
val_features = val_split[feature_cols_final].copy()
test_features = combined_test[feature_cols_final].copy()

imputer = SimpleImputer(strategy="median")
train_features = pd.DataFrame(
    imputer.fit_transform(train_features), columns=feature_cols_final
)
val_features = pd.DataFrame(imputer.transform(val_features), columns=feature_cols_final)
test_features = pd.DataFrame(
    imputer.transform(test_features), columns=feature_cols_final
)

# Replace inf values
for name, df in [
    ("train", train_features),
    ("val", val_features),
    ("test", test_features),
]:
    inf_count = np.isinf(df.values).sum()
    if inf_count > 0:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        print(f"  {name}: replaced {inf_count} inf values")

y_train = train_split[target_col].values
y_val = val_split[target_col].values

# Store original indices for test (before any reset_index)
test_original_idx = test.index.values

print(f"Train target: mean={y_train.mean():.6f}, std={y_train.std():.6f}")
print(f"Val target: mean={y_val.mean():.6f}, std={y_val.std():.6f}")

# ============================================================
# MODEL CONFIGS
# ============================================================
configs = [
    {
        "name": "strong_reg",
        "n_estimators": 500,
        "learning_rate": 0.01,
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 30,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "random_state": 42,
        "verbose": -1,
    },
    {
        "name": "moderate_reg",
        "n_estimators": 500,
        "learning_rate": 0.02,
        "num_leaves": 12,
        "max_depth": 4,
        "min_child_samples": 20,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "random_state": 42,
        "verbose": -1,
    },
    {
        "name": "light_reg",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 16,
        "max_depth": 5,
        "min_child_samples": 15,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "random_state": 42,
        "verbose": -1,
    },
    {
        "name": "deep_shallow",
        "n_estimators": 800,
        "learning_rate": 0.005,
        "num_leaves": 6,
        "max_depth": 3,
        "min_child_samples": 40,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "subsample": 0.6,
        "colsample_bytree": 0.4,
        "random_state": 42,
        "verbose": -1,
    },
    {
        "name": "balanced",
        "n_estimators": 600,
        "learning_rate": 0.015,
        "num_leaves": 10,
        "max_depth": 4,
        "min_child_samples": 25,
        "reg_alpha": 0.8,
        "reg_lambda": 4.0,
        "subsample": 0.75,
        "colsample_bytree": 0.55,
        "random_state": 42,
        "verbose": -1,
    },
]

# ============================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================
best_rmse = float("inf")
best_config_name = None
best_model = None
best_iteration = None
all_results = []

print("\n" + "=" * 60)
print("TRAINING AND EVALUATION")
print("=" * 60)

for cfg in configs:
    model = lgb.LGBMRegressor(**cfg)
    model.fit(
        train_features,
        y_train,
        eval_set=[(val_features, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_pred = model.predict(val_features)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    all_results.append(
        {"name": cfg["name"], "rmse": rmse, "best_iter": model.best_iteration_}
    )
    print(f"  {cfg['name']:>15s}: RMSE={rmse:.6f}, best_iter={model.best_iteration_}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_config_name = cfg["name"]
        best_model = model
        best_iteration = model.best_iteration_

print(
    f"\nBest config: {best_config_name} with RMSE={best_rmse:.6f}, iterations={best_iteration}"
)

# ============================================================
# RETRAIN BEST MODEL ON ALL AVAILABLE DATA (train + val)
# ============================================================
all_train_features = pd.concat([train_features, val_features], axis=0).reset_index(
    drop=True
)
all_train_target = np.concatenate([y_train, y_val])

best_cfg = next(c for c in configs if c["name"] == best_config_name)
final_model = lgb.LGBMRegressor(**{**best_cfg, "n_estimators": best_iteration})
final_model.fit(
    all_train_features, all_train_target, callbacks=[lgb.log_evaluation(period=0)]
)

print(
    f"Final model trained on {len(all_train_features)} samples, {best_iteration} iterations"
)

# ============================================================
# TEST INFERENCE
# ============================================================
test_pred = final_model.predict(test_features)
print(f"Test predictions: mean={test_pred.mean():.6f}, std={test_pred.std():.6f}")

# ============================================================
# GENERATE SUBMISSION
# ============================================================
submission = pd.DataFrame({"id": range(len(test_pred)), "ISE": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape[0]} rows")
print(f"Test original index range: {test_original_idx.min()} to {test_original_idx.max()}")

# ============================================================
# FINAL VALIDATION SCORE
# ============================================================
# Report the unbiased early-stopping RMSE from the selection phase
score = best_rmse

print(f"\n{'=' * 60}")
print(f"VALIDATION SUMMARY")
print(f"{'=' * 60}")
print(f"Split method: Temporal (last 20% of training data)")
print(f"Validation samples: {len(y_val)}")
print(f"Metric: RMSE (official competition metric)")
print(f"Best model: {best_config_name}")
print(f"Best iteration: {best_iteration}")
print(f"\nAll results:")
for r in all_results:
    marker = " *" if r["name"] == best_config_name else ""
    print(f"  {r['name']:>15s}: RMSE={r['rmse']:.6f}{marker}")

print(f"Final Validation Score: {score:.6f}")