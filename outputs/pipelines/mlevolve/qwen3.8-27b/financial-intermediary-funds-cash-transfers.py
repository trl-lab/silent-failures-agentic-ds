import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedShuffleSplit
import os
import warnings
import re

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")
print(f"\nTarget stats:\n{train['Amount in USD'].describe()}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
target_col = "Amount in USD"
id_col = "id"

y_train = train[target_col].copy()
train_features = train.drop(columns=[target_col])

numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_features.select_dtypes(
    include=["object", "category"]
).columns.tolist()
datetime_cols = [
    c
    for c in train_features.columns
    if "date" in c.lower() or "time" in c.lower() or "year" in c.lower()
]

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Datetime-like columns: {datetime_cols}")

# ============================================================
# 3. CREATE TRAIN/VALIDATION SPLIT
# ============================================================
target_bins = pd.qcut(y_train, q=5, labels=False, duplicates="drop")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(sss.split(train_features, target_bins))

train_split = train_features.iloc[train_idx].copy()
val_split = train_features.iloc[val_idx].copy()
y_train_split = y_train.iloc[train_idx].copy()
y_val_split = y_train.iloc[val_idx].copy()

print(f"\nTrain split: {len(train_split)}, Val split: {len(val_split)}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(
    df, is_training=False, ref_stats=None, ref_encodings=None, train_ref=None, target=None
):
    """Apply feature engineering with K-fold target encoding."""
    df = df.copy()

    global_mean = target.mean() if target is not None else 0.0
    m_smooth = 10

    # --- Handle datetime columns ---
    for col in datetime_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_quarter"] = df[col].dt.quarter
                df.drop(columns=[col], inplace=True)
            except:
                pass

    # --- Target encoding for high-cardinality categoricals ---
    high_card_cols = [c for c in categorical_cols if c in df.columns and df[c].nunique() > 20]
    low_card_cols = [c for c in categorical_cols if c in df.columns and df[c].nunique() <= 20]

    if is_training:
        ref_encodings = {}

        for col in high_card_cols:
            df[f"{col}_te"] = np.nan
            kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, va_idx in kf_te.split(df):
                tr_target = target.iloc[tr_idx]
                tr_col = df[col].iloc[tr_idx]
                stats = tr_target.groupby(tr_col).agg(['mean', 'count'])
                smoothed = (stats['count'] * stats['mean'] + m_smooth * global_mean) / (stats['count'] + m_smooth)
                va_vals = df[col].iloc[va_idx].map(smoothed).fillna(global_mean)
                df.iloc[va_idx, df.columns.get_loc(f"{col}_te")] = va_vals.values

            full_stats = target.groupby(df[col]).agg(['mean', 'count'])
            full_smoothed = (full_stats['count'] * full_stats['mean'] + m_smooth * global_mean) / (full_stats['count'] + m_smooth)
            ref_encodings[col] = full_smoothed
            df[col] = df[col].astype("category")

        # Interaction target encoding: Fund Name x Transfer Quarter
        if 'Fund Name' in df.columns and 'Transfer Quarter' in df.columns:
            df['_fund_quarter'] = df['Fund Name'].astype(str) + '_' + df['Transfer Quarter'].astype(str)
            df['Fund_Name_x_Quarter_te'] = np.nan
            kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, va_idx in kf_te.split(df):
                tr_target = target.iloc[tr_idx]
                tr_col = df['_fund_quarter'].iloc[tr_idx]
                stats = tr_target.groupby(tr_col).agg(['mean', 'count'])
                smoothed = (stats['count'] * stats['mean'] + m_smooth * global_mean) / (stats['count'] + m_smooth)
                va_vals = df['_fund_quarter'].iloc[va_idx].map(smoothed).fillna(global_mean)
                df.iloc[va_idx, df.columns.get_loc('Fund_Name_x_Quarter_te')] = va_vals.values
            full_stats = target.groupby(df['_fund_quarter']).agg(['mean', 'count'])
            full_smoothed = (full_stats['count'] * full_stats['mean'] + m_smooth * global_mean) / (full_stats['count'] + m_smooth)
            ref_encodings['_fund_quarter'] = full_smoothed
            df.drop(columns=['_fund_quarter'], inplace=True)

        # Interaction target encoding: Principal Recipient x Sector/Theme
        if 'Principal Recipient' in df.columns and 'Sector/Theme' in df.columns:
            df['_recipient_sector'] = df['Principal Recipient'].astype(str) + '_' + df['Sector/Theme'].astype(str)
            df['Recipient_x_Sector_te'] = np.nan
            kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, va_idx in kf_te.split(df):
                tr_target = target.iloc[tr_idx]
                tr_col = df['_recipient_sector'].iloc[tr_idx]
                stats = tr_target.groupby(tr_col).agg(['mean', 'count'])
                smoothed = (stats['count'] * stats['mean'] + m_smooth * global_mean) / (stats['count'] + m_smooth)
                va_vals = df['_recipient_sector'].iloc[va_idx].map(smoothed).fillna(global_mean)
                df.iloc[va_idx, df.columns.get_loc('Recipient_x_Sector_te')] = va_vals.values
            full_stats = target.groupby(df['_recipient_sector']).agg(['mean', 'count'])
            full_smoothed = (full_stats['count'] * full_stats['mean'] + m_smooth * global_mean) / (full_stats['count'] + m_smooth)
            ref_encodings['_recipient_sector'] = full_smoothed
            df.drop(columns=['_recipient_sector'], inplace=True)

        # Low cardinality: one-hot encoding
        for col in low_card_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df[col] = df[col].astype("category")
    else:
        for col in high_card_cols:
            if col in df.columns and col in ref_encodings:
                df[f"{col}_te"] = df[col].map(ref_encodings[col]).fillna(global_mean)
                df[col] = df[col].astype("category")

        # Interaction target encodings for val/test
        if 'Fund Name' in df.columns and 'Transfer Quarter' in df.columns and '_fund_quarter' in ref_encodings:
            df['_fund_quarter'] = df['Fund Name'].astype(str) + '_' + df['Transfer Quarter'].astype(str)
            df['Fund_Name_x_Quarter_te'] = df['_fund_quarter'].map(ref_encodings['_fund_quarter']).fillna(global_mean)
            df.drop(columns=['_fund_quarter'], inplace=True)

        if 'Principal Recipient' in df.columns and 'Sector/Theme' in df.columns and '_recipient_sector' in ref_encodings:
            df['_recipient_sector'] = df['Principal Recipient'].astype(str) + '_' + df['Sector/Theme'].astype(str)
            df['Recipient_x_Sector_te'] = df['_recipient_sector'].map(ref_encodings['_recipient_sector']).fillna(global_mean)
            df.drop(columns=['_recipient_sector'], inplace=True)

        for col in low_card_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                train_dummies = pd.get_dummies(train_ref[col], prefix=col, drop_first=True)
                for c in train_dummies.columns:
                    if c not in dummies.columns:
                        dummies[c] = 0
                dummies = dummies[train_dummies.columns]
                df = pd.concat([df, dummies], axis=1)
                df[col] = df[col].astype("category")

    # --- Numeric feature engineering ---
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if df[col].min() > 0:
            df[f"{col}_log"] = np.log1p(df[col])
        df[f"{col}_sq"] = df[col] ** 2

    # --- Interaction features ---
    if is_training:
        num_df = train_ref[[c for c in numeric_cols if c in train_ref.columns and train_ref[c].dtype != "datetime64[ns]"]]
        corr_matrix = num_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_pairs = upper.stack().nlargest(10).index.tolist()
        ref_stats = {"top_pairs": top_pairs}
    else:
        top_pairs = ref_stats["top_pairs"]

    for pair in top_pairs:
        col1, col2 = pair
        if col1 in df.columns and col2 in df.columns and pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
            df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)

    # --- Missing value indicators ---
    for col in df.columns:
        if df[col].isnull().any():
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    return df, ref_stats, ref_encodings


# Apply feature engineering (pass target for target encoding)
train_split, ref_stats, ref_encodings = engineer_features(
    train_split, is_training=True, train_ref=train_split, target=y_train_split
)
val_split, _, _ = engineer_features(
    val_split,
    is_training=False,
    ref_stats=ref_stats,
    ref_encodings=ref_encodings,
    train_ref=train_split,
    target=y_train_split,
)
test_features, _, _ = engineer_features(
    test,
    is_training=False,
    ref_stats=ref_stats,
    ref_encodings=ref_encodings,
    train_ref=train_split,
    target=y_train_split,
)

# Align columns
all_cols = [
    c
    for c in train_split.columns
    if c in val_split.columns and c in test_features.columns
]
train_split = train_split[all_cols]
val_split = val_split[all_cols]
test_features = test_features[all_cols]

# Sanitize column names: replace non-alphanumeric chars with underscore (LightGBM requirement)
def sanitize_col_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))

train_split.columns = [sanitize_col_name(c) for c in train_split.columns]
val_split.columns = [sanitize_col_name(c) for c in val_split.columns]
test_features.columns = [sanitize_col_name(c) for c in test_features.columns]

# Convert categoricals to codes
for col in train_split.columns:
    if train_split[col].dtype == "category":
        train_split[col] = train_split[col].cat.codes
        val_split[col] = val_split[col].cat.codes
        test_features[col] = test_features[col].cat.codes

# Fill NaNs
train_split = train_split.fillna(-999)
val_split = val_split.fillna(-999)
test_features = test_features.fillna(-999)

print(f"\nFinal feature count: {len(all_cols)}")

# ============================================================
# 5. TARGET (already in log-scale, use directly)
# ============================================================
y_train_target = y_train_split.values
y_val_target = y_val_split.values

# ============================================================
# 6. 5-FOLD OOF PREDICTIONS FOR ENSEMBLE BLENDING
# ============================================================
X_full = pd.concat([train_split, val_split], axis=0, ignore_index=True)
y_full = np.concatenate([y_train_target, y_val_target])

print(f"\n--- 5-Fold OOF Predictions ---")

n_folds = 5
kf_oof = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# LGB OOF
lgb_oof = np.zeros(len(X_full))
lgb_best_iters = []
lgb_params = {
    "objective": "huber",
    "huber_delta": 1.0,
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 2,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_split_gain": 0.01,
    "n_estimators": 3000,
    "random_state": 42,
    "verbose": -1,
}

for fold, (tr_idx, va_idx) in enumerate(kf_oof.split(X_full)):
    X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    lgb_oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration_)
    lgb_best_iters.append(model.best_iteration_)
    print(f"  LGB Fold {fold+1}: best_iter={model.best_iteration_}, fold_rmse={root_mean_squared_error(y_va, lgb_oof[va_idx]):.4f}")

lgb_oof_rmse = root_mean_squared_error(y_full, lgb_oof)
print(f"  LGB OOF RMSE: {lgb_oof_rmse:.4f}")
print(f"  LGB avg best_iter: {np.mean(lgb_best_iters):.0f}")

# XGB OOF
xgb_oof = np.zeros(len(X_full))
xgb_best_iters = []
xgb_params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_estimators": 3000,
    "verbosity": 0,
}

for fold, (tr_idx, va_idx) in enumerate(kf_oof.split(X_full)):
    X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=3000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    xgb_oof[va_idx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    xgb_best_iters.append(model.best_iteration + 1)
    print(f"  XGB Fold {fold+1}: best_iter={model.best_iteration+1}, fold_rmse={root_mean_squared_error(y_va, xgb_oof[va_idx]):.4f}")

xgb_oof_rmse = root_mean_squared_error(y_full, xgb_oof)
print(f"  XGB OOF RMSE: {xgb_oof_rmse:.4f}")
print(f"  XGB avg best_iter: {np.mean(xgb_best_iters):.0f}")

# Ridge OOF
ridge_oof = np.zeros(len(X_full))
for fold, (tr_idx, va_idx) in enumerate(kf_oof.split(X_full)):
    X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
    y_tr = y_full[tr_idx]
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X_tr, y_tr)
    ridge_oof[va_idx] = model.predict(X_va)

ridge_oof_rmse = root_mean_squared_error(y_full, ridge_oof)
print(f"  Ridge OOF RMSE: {ridge_oof_rmse:.4f}")

# ============================================================
# 7. BLEND WEIGHT OPTIMIZATION ON OOF
# ============================================================
print(f"\n--- Blend Weight Optimization (on OOF) ---")
from itertools import product

best_w_lgb, best_w_xgb, best_w_ridge = 1.0, 0.0, 0.0
best_blend_oof_rmse = float("inf")

for w_l in np.arange(0.0, 1.01, 0.1):
    for w_x in np.arange(0.0, 1.01 - w_l, 0.1):
        w_r = 1.0 - w_l - w_x
        if w_r < -0.01:
            continue
        w_r = max(w_r, 0.0)
        blend = w_l * lgb_oof + w_x * xgb_oof + w_r * ridge_oof
        rmse = root_mean_squared_error(y_full, blend)
        if rmse < best_blend_oof_rmse:
            best_blend_oof_rmse = rmse
            best_w_lgb, best_w_xgb, best_w_ridge = w_l, w_x, w_r

print(f"  Best weights: LGB={best_w_lgb:.1f}, XGB={best_w_xgb:.1f}, Ridge={best_w_ridge:.1f}")
print(f"  Best OOF Blend RMSE: {best_blend_oof_rmse:.4f}")

# ============================================================
# 8. TRAIN FINAL MODELS ON FULL DATA
# ============================================================
print(f"\n--- Training Final Models on Full Data ---")
avg_lgb_iter = int(np.mean(lgb_best_iters))
avg_xgb_iter = int(np.mean(xgb_best_iters))
print(f"  LGB avg iterations: {avg_lgb_iter}")
print(f"  XGB avg iterations: {avg_xgb_iter}")

# Final LGB
lgb_params_final = dict(lgb_params)
lgb_params_final["n_estimators"] = avg_lgb_iter
final_lgb = lgb.LGBMRegressor(**lgb_params_final)
final_lgb.fit(X_full, y_full)

# Final XGB
xgb_params_final = dict(xgb_params)
xgb_params_final.pop("early_stopping_rounds", None)
dtrain_full = xgb.DMatrix(X_full, label=y_full)
final_xgb = xgb.train(xgb_params_final, dtrain_full, num_boost_round=avg_xgb_iter, verbose_eval=False)

# Final Ridge
final_ridge = Ridge(alpha=1.0, fit_intercept=True)
final_ridge.fit(X_full, y_full)

# Cap LGB predictions to avg iterations
test_pred_lgb = final_lgb.booster_.predict(test_features, num_iteration=avg_lgb_iter)

# ============================================================
# 9. VALIDATION PREDICTIONS (using OOF for honest estimate)
# ============================================================
val_oof_blend = best_w_lgb * lgb_oof[len(train_split):] + best_w_xgb * xgb_oof[len(train_split):] + best_w_ridge * ridge_oof[len(train_split):]
val_rmse = root_mean_squared_error(y_val_target, val_oof_blend)
print(f"\n  Val RMSE (OOF blend, held-out split): {val_rmse:.4f}")

# ============================================================
# 10. TEST PREDICTIONS
# ============================================================
test_pred_xgb = final_xgb.predict(xgb.DMatrix(test_features))
test_pred_ridge = final_ridge.predict(test_features)
test_pred_final = best_w_lgb * test_pred_lgb + best_w_xgb * test_pred_xgb + best_w_ridge * test_pred_ridge

print(f"  Test predictions: mean={test_pred_final.mean():.2f}, min={test_pred_final.min():.2f}, max={test_pred_final.max():.2f}")

# ============================================================
# 11. SAVE SUBMISSION
# ============================================================
os.makedirs("./submission", exist_ok=True)
test_ids = test["id"] if "id" in test.columns else pd.RangeIndex(len(test))
submission = pd.DataFrame(
    {"id": test_ids.astype(int), "Amount in USD": test_pred_final}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape[0]} rows")

# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================
fi = pd.DataFrame(
    {"feature": X_full.columns, "importance": final_lgb.feature_importances_}
).sort_values("importance", ascending=False)
print(f"\nTop 10 features by importance:")
print(fi.head(10).to_string(index=False))

# ============================================================
# 13. FINAL SCORE
# ============================================================
score = best_blend_oof_rmse
print(f"Final Validation Score (OOF RMSE): {score:.4f}")