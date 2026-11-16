import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/commitments-in-trust-funds/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/commitments-in-trust-funds/data/test.parquet')

# Define target
target = 'New Commitments (US$)'

# Drop high-cardinality identifier columns that won't generalize
drop_cols = ['Trust Fund', 'Trust Fund name', 'Trustee', 'Trustee Name']

# Features to use
feature_cols = [c for c in train_df.columns if c != target and c not in drop_cols]

X_train = train_df[feature_cols].copy()
y_train = train_df[target].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'category':
        X_train[col] = X_train[col].fillna('MISSING')
        X_test[col] = X_test[col].fillna('MISSING')
    else:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_train[col].median())

# Convert categorical columns to category type
cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Ensure test categories are a superset of train categories
for col in cat_cols:
    all_cats = set(X_train[col].cat.categories) | set(X_test[col].cat.categories)
    X_train[col] = X_train[col].cat.set_categories(all_cats)
    X_test[col] = X_test[col].cat.set_categories(all_cats)

# Validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model 1: LightGBM
lgbm_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_tr, y_tr)
lgbm_val_pred = lgbm_model.predict(X_val)
lgbm_rmse = np.sqrt(np.mean((lgbm_val_pred - y_val) ** 2))
print(f"LightGBM Validation RMSE: {lgbm_rmse:.6f}")

# Model 2: XGBoost
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='rmse',
    enable_categorical=True
)
xgb_model.fit(X_tr, y_tr)
xgb_val_pred = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(np.mean((xgb_val_pred - y_val) ** 2))
print(f"XGBoost Validation RMSE: {xgb_rmse:.6f}")

# Model 3: CatBoost
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=0,
    cat_features=cat_cols
)
cat_model.fit(X_tr, y_tr)
cat_val_pred = cat_model.predict(X_val)
cat_rmse = np.sqrt(np.mean((cat_val_pred - y_val) ** 2))
print(f"CatBoost Validation RMSE: {cat_rmse:.6f}")

# Retrain on full training data
lgbm_model_full = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    random_state=42,
    verbose=-1
)
lgbm_model_full.fit(X_train, y_train)

xgb_model_full = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='rmse',
    enable_categorical=True
)
xgb_model_full.fit(X_train, y_train)

cat_model_full = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=0,
    cat_features=cat_cols
)
cat_model_full.fit(X_train, y_train)

# Predict on test
pred_lgbm = lgbm_model_full.predict(X_test)
pred_xgb = xgb_model_full.predict(X_test)
pred_cat = cat_model_full.predict(X_test)

# Weighted ensemble based on validation performance
rmse_list = [lgbm_rmse, xgb_rmse, cat_rmse]
weights = [1.0 / r for r in rmse_list]
total_weight = sum(weights)
weights = [w / total_weight for w in weights]

print(f"Weights - LGBM: {weights[0]:.3f}, XGB: {weights[1]:.3f}, Cat: {weights[2]:.3f}")

final_pred = weights[0] * pred_lgbm + weights[1] * pred_xgb + weights[2] * pred_cat

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'New Commitments (US$)': final_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")