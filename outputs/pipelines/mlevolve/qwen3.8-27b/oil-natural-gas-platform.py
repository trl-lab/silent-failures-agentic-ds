import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import os
import json
import re

# ============ SETUP ============
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============ LOAD DATA ============
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(
    f"Target stats: mean={train['WATERDEPTH'].mean():.2f}, std={train['WATERDEPTH'].std():.2f}"
)
print(
    f"Target range: [{train['WATERDEPTH'].min():.2f}, {train['WATERDEPTH'].max():.2f}]"
)

# ============ IDENTIFY FEATURE TYPES ============
target = "WATERDEPTH"
id_col = "id" if "id" in train.columns else None

y_train_full = train[target].values
train_features = train.drop(columns=[target])

numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = train_features.select_dtypes(
    include=["object", "category"]
).columns.tolist()
bool_cols = [c for c in train_features.columns if train_features[c].dtype == "bool"]

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Boolean columns ({len(bool_cols)}): {bool_cols}")

print(
    f"\nMissing values in train:\n{train_features.isnull().sum()[train_features.isnull().sum() > 0]}"
)

# ============ PREPARE TEST FEATURES ============
test_features = test.copy()
if id_col and id_col in test_features.columns:
    test_id = test_features[id_col].values
    test_features = test_features.drop(columns=[id_col])
else:
    test_id = np.arange(len(test_features))

# ============ COMBINE FOR CONSISTENT FEATURE ENGINEERING ============
train_features = train_features.copy()
test_features = test_features.copy()
train_features["is_test"] = 0
test_features["is_test"] = 1
combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# ============ FEATURE ENGINEERING ============

# 1. Handle boolean columns
for col in bool_cols:
    combined[col] = combined[col].astype(int)

# 2. Parse BLOCKNO, LEASENO, AREANUMBER into numeric spatial components
for col in ['BLOCKNO', 'LEASENO', 'AREANUMBER']:
    if col in combined.columns:
        # Extract numeric components using regex
        numeric_parts = combined[col].astype(str).apply(
            lambda x: [int(p) for p in re.findall(r'\d+', x)]
        )
        max_parts = numeric_parts.apply(len).max() if len(numeric_parts) > 0 else 0
        for i in range(max_parts):
            combined[f'{col}_part{i}'] = numeric_parts.apply(
                lambda x: x[i] if i < len(x) else 0
            )
        combined[f'{col}_num_parts'] = numeric_parts.apply(len)
        print(f"Parsed '{col}' into {max_parts} numeric parts")

# 3. Target encoding for high-cardinality categoricals (K-fold to prevent leakage)
high_card_cols = ['GlobalID', 'PLATFRMID', 'COMPLEXID', 'BLOCKNO', 'LEASENO', 'AREANUMBER']
low_card_cols = [c for c in categorical_cols if c not in high_card_cols]

# For low-cardinality categoricals, use LabelEncoder
label_encoders = {}
for col in low_card_cols:
    le = LabelEncoder()
    train_vals = combined.loc[combined["is_test"] == 0, col].astype(str)
    le.fit(train_vals)
    combined[col] = (
        combined[col]
        .astype(str)
        .map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    )
    label_encoders[col] = le
    print(f"LabelEncoded '{col}': {len(le.classes_)} categories")

# For high-cardinality categoricals, use K-fold target encoding
target_col = 'WATERDEPTH'
train_mask = combined["is_test"] == 0
global_mean = y_train_full.mean()
n_folds_te = 5
kf_te = KFold(n_splits=n_folds_te, shuffle=True, random_state=42)

for col in high_card_cols:
    if col not in combined.columns:
        continue
    combined[f'{col}_target_enc'] = np.nan
    train_indices = np.where(train_mask.values)[0]

    # K-fold target encoding on training data only
    for fold_idx, (tr_idx, va_idx) in enumerate(kf_te.split(train_indices)):
        tr_abs = train_indices[tr_idx]
        va_abs = train_indices[va_idx]

        # Compute target mean per category on training fold
        fold_data = pd.DataFrame({
            'cat': combined.loc[tr_abs, col].astype(str).values,
            'target': y_train_full[tr_abs]
        })
        cat_stats = fold_data.groupby('cat')['target'].agg(['mean', 'count'])

        # Smoothing: (count * mean + global_mean * smoothing) / (count + smoothing)
        smoothing = 10
        cat_stats['smoothed'] = (cat_stats['count'] * cat_stats['mean'] + global_mean * smoothing) / (cat_stats['count'] + smoothing)

        # Apply to validation fold
        va_cats = combined.loc[va_abs, col].astype(str).values
        va_enc = pd.Series(va_cats).map(cat_stats['smoothed']).fillna(global_mean).values
        combined.loc[va_abs, f'{col}_target_enc'] = va_enc

    # For test data, use global target encoding (all training data)
    test_indices = np.where(~train_mask.values)[0]
    if len(test_indices) > 0:
        all_train_data = pd.DataFrame({
            'cat': combined.loc[train_indices, col].astype(str).values,
            'target': y_train_full
        })
        all_cat_stats = all_train_data.groupby('cat')['target'].agg(['mean', 'count'])
        all_cat_stats['smoothed'] = (all_cat_stats['count'] * all_cat_stats['mean'] + global_mean * smoothing) / (all_cat_stats['count'] + smoothing)

        test_cats = combined.loc[test_indices, col].astype(str).values
        test_enc = pd.Series(test_cats).map(all_cat_stats['smoothed']).fillna(global_mean).values
        combined.loc[test_indices, f'{col}_target_enc'] = test_enc

    print(f"Target encoded '{col}' (K-fold={n_folds_te}, smoothing={smoothing})")

# Remove original high-cardinality categorical columns (replaced by target encoding)
for col in high_card_cols:
    if col in combined.columns:
        combined = combined.drop(columns=[col])

# 4. Log transform for skewed features (fit stats on train only)
spatial_cols = [c for c in combined.columns if '_part' in c or 'num_parts' in c]
key_numeric = [c for c in numeric_cols if c not in spatial_cols and c not in bool_cols and c in combined.columns]

for col in key_numeric:
    train_vals = combined.loc[combined["is_test"] == 0, col]
    if train_vals.min() > 0 and train_vals.skew() > 1:
        combined[f"{col}_log"] = np.log1p(combined[col])
        print(f"  Log-transformed: {col} (skew={train_vals.skew():.2f})")

# 5. Interaction features on parsed spatial components
parsed_cols = [c for c in combined.columns if '_part' in c or 'num_parts' in c]
# Also ensure we don't reference dropped columns in interactions
if len(parsed_cols) >= 2:
    for i in range(len(parsed_cols) - 1):
        c1, c2 = parsed_cols[i], parsed_cols[i + 1]
        combined[f"{c1}_x_{c2}"] = combined[c1] * combined[c2]

# Interaction between target-encoded features
te_cols = [c for c in combined.columns if c.endswith('_target_enc')]
# Guard: only create interactions if columns still exist
if len(te_cols) >= 2:
    for i in range(min(3, len(te_cols) - 1)):
        c1, c2 = te_cols[i], te_cols[i + 1]
        combined[f"{c1}_x_{c2}"] = combined[c1] * combined[c2]

# 6. Handle missing values (fill with train statistics)
for col in combined.columns:
    if col == "is_test":
        continue
    train_col = combined.loc[combined["is_test"] == 0, col]
    if train_col.isnull().any():
        if pd.api.types.is_numeric_dtype(combined[col]):
            fill_val = train_col.median()
            combined[col] = combined[col].fillna(fill_val)
            print(f"  Filled missing in '{col}' with median={fill_val:.2f}")
        else:
            fill_val = train_col.mode()[0] if not train_col.mode().empty else 0
            combined[col] = combined[col].fillna(fill_val)

# 7. Scale numeric features (fit on train only)
scale_cols = [
    c
    for c in combined.columns
    if c != "is_test" and combined[c].dtype in ["int64", "float64"]
]

scaler = StandardScaler()
train_portion = combined[combined["is_test"] == 0]
scaler.fit(train_portion[scale_cols])
combined[scale_cols] = scaler.transform(combined[scale_cols])
print(f"\nScaled {len(scale_cols)} numeric features")

# 8. Remove is_test flag and split back
combined = combined.drop(columns=["is_test"])

n_train = len(train_features)
train_processed = combined.iloc[:n_train].reset_index(drop=True)
test_processed = combined.iloc[n_train:].reset_index(drop=True)

# ============ TRAIN/VAL SPLIT ============
train_idx, val_idx = train_test_split(
    np.arange(len(train_processed)), test_size=0.2, random_state=42
)

X_train = train_processed.iloc[train_idx].values
X_val = train_processed.iloc[val_idx].values
y_train = y_train_full[train_idx]
y_val = y_train_full[val_idx]
X_test = test_processed.values

print(f"\n=== FINAL SPLIT SUMMARY ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")

# Save feature names
with open("./working/feature_names.json", "w") as f:
    json.dump(list(train_processed.columns), f)

# Baseline RMSE (mean predictor)
baseline_rmse = np.sqrt(np.mean((y_val - y_train.mean()) ** 2))
print(f"\nBaseline RMSE (mean predictor): {baseline_rmse:.4f}")
print(f"Target std: {y_train.std():.4f}")

# ============ MODEL TRAINING - STACKED ENSEMBLE ============

# Model 1: LightGBM
lgbm_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

callbacks_lgb = [
    lgb.early_stopping(stopping_rounds=100, verbose=False),
    lgb.log_evaluation(period=500),
]

lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks_lgb)
best_iter_lgb = lgbm_model.best_iteration_
best_l2_lgb = lgbm_model.best_score_["valid_0"]["l2"]
print(f"[LGBM] Best iteration: {best_iter_lgb}, Best val l2: {best_l2_lgb:.6f}")

# Model 2: XGBoost
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    reg_alpha=0.5,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=100,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
# Find best iteration from xgb
best_iter_xgb = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') and xgb_model.best_iteration >= 0 else 3000
print(f"[XGB] Best iteration: {best_iter_xgb}")

# Model 3: Ridge regression on target-encoded features (spatial smoother)
te_feature_cols = [c for c in train_processed.columns if c.endswith('_target_enc')]
if len(te_feature_cols) > 0:
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train[:, [list(train_processed.columns).index(c) for c in te_feature_cols]], y_train)
    ridge_available = True
else:
    ridge_available = False
    ridge_model = None

print(f"[Ridge] Available: {ridge_available}, features: {len(te_feature_cols)}")

# ============ VALIDATION - ENSEMBLE ============
val_preds_lgbm = lgbm_model.predict(X_val)
val_preds_xgb = xgb_model.predict(X_val, iteration_range=(0, best_iter_xgb + 1)) if best_iter_xgb < 5000 else xgb_model.predict(X_val)

if ridge_available:
    te_indices = [list(train_processed.columns).index(c) for c in te_feature_cols]
    val_preds_ridge = ridge_model.predict(X_val[:, te_indices])
else:
    val_preds_ridge = np.full_like(val_preds_lgbm, y_train.mean())

# Optimize weights on validation set
from scipy.optimize import minimize

def ensemble_rmse(weights):
    w_lgb, w_xgb, w_ridge = weights
    preds = w_lgb * val_preds_lgbm + w_xgb * val_preds_xgb + w_ridge * val_preds_ridge
    return np.sqrt(mean_squared_error(y_val, preds))

# Constraint: weights sum to 1, all non-negative
constraints = {'type': 'eq', 'fun': lambda w: w[0] + w[1] + w[2] - 1}
bounds = [(0, 1), (0, 1), (0, 1)]
initial_weights = [1/3, 1/3, 1/3]
result = minimize(ensemble_rmse, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

opt_weights = result.x
print(f"\nOptimal ensemble weights: LGBM={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, Ridge={opt_weights[2]:.4f}")

val_preds = opt_weights[0] * val_preds_lgbm + opt_weights[1] * val_preds_xgb + opt_weights[2] * val_preds_ridge
val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
print(f"Validation RMSE (ensemble): {val_rmse:.6f}")
print(f"Val preds range: [{val_preds.min():.2f}, {val_preds.max():.2f}]")
print(f"Val preds mean: {val_preds.mean():.2f} vs target mean: {y_val.mean():.2f}")

# Individual model validation scores
val_rmse_lgbm = float(np.sqrt(mean_squared_error(y_val, val_preds_lgbm)))
val_rmse_xgb = float(np.sqrt(mean_squared_error(y_val, val_preds_xgb)))
val_rmse_ridge = float(np.sqrt(mean_squared_error(y_val, val_preds_ridge)))
print(f"  LGBM RMSE: {val_rmse_lgbm:.6f}")
print(f"  XGB RMSE: {val_rmse_xgb:.6f}")
print(f"  Ridge RMSE: {val_rmse_ridge:.6f}")

# ============ TEST PREDICTIONS ============
test_preds_lgbm = lgbm_model.predict(X_test)
test_preds_xgb = xgb_model.predict(X_test, iteration_range=(0, best_iter_xgb + 1)) if best_iter_xgb < 5000 else xgb_model.predict(X_test)

if ridge_available:
    te_indices = [list(train_processed.columns).index(c) for c in te_feature_cols]
    test_preds_ridge = ridge_model.predict(X_test[:, te_indices])
else:
    test_preds_ridge = np.full_like(test_preds_lgbm, y_train.mean())

test_preds = opt_weights[0] * test_preds_lgbm + opt_weights[1] * test_preds_xgb + opt_weights[2] * test_preds_ridge
print(f"Test preds range: [{test_preds.min():.2f}, {test_preds.max():.2f}]")
print(f"Test preds mean: {test_preds.mean():.2f}")

# ============ SAVE SUBMISSION ============
submission = pd.DataFrame({"id": test_id.astype(int), "WATERDEPTH": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {submission.shape[0]} rows")
print(submission.head())

# ============ FEATURE IMPORTANCE ============
with open("./working/feature_names.json", "r") as f:
    feature_names = json.load(f)

importance = lgbm_model.feature_importances_
top_idx = np.argsort(importance)[::-1][:10]
print("\nTop 10 features by importance (LGBM):")
for i in top_idx:
    print(f"  {feature_names[i]}: {importance[i]}")

# Save models
lgbm_model.booster_.save_model("./working/lgbm_model.txt")
xgb_model.save_model("./working/xgb_model.json")
print("Models saved to ./working/")

# ============ FINAL SCORE ============
score = val_rmse
print(f"Final Validation Score: {score}")