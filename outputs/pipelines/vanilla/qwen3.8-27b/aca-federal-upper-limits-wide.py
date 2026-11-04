import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/aca-federal-upper-limits-wide/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/aca-federal-upper-limits-wide/data/test.parquet')

# Separate target
y_train = train_df['ACA FUL'].values
X_train_full = train_df.drop('ACA FUL', axis=1)
X_test = test_df.copy()

# Identify column types
numeric_cols = ['Product Group', 'Package Size', 'Year', 'Month']
categorical_cols = ['Ingredient', 'Strength', 'Dosage', 'Route', 'MDR Unit Type', 'NDC', 'A-Rated', 'Multiplier Greater Than 175 Percent of Weighted Avg of AMPs']

# Fill missing values
for col in numeric_cols:
    if col in X_train_full.columns:
        X_train_full[col] = X_train_full[col].fillna(X_train_full[col].median())
        X_test[col] = X_test[col].fillna(X_train_full[col].median())

for col in categorical_cols:
    if col in X_train_full.columns:
        X_train_full[col] = X_train_full[col].fillna('MISSING')
        X_test[col] = X_test[col].fillna('MISSING')

# Encode categoricals using label encoding (LightGBM handles them natively)
# For high-cardinality columns, use frequency encoding as additional features
# Low cardinality: Dosage, Route, MDR Unit Type, A-Rated, Multiplier Greater Than 175 Percent of Weighted Avg of AMPs
# High cardinality: Ingredient, Strength, NDC

low_card_cols = ['Dosage', 'Route', 'MDR Unit Type', 'A-Rated', 'Multiplier Greater Than 175 Percent of Weighted Avg of AMPs']
high_card_cols = ['Ingredient', 'Strength', 'NDC']

# Label encode low cardinality
for col in low_card_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train_full[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Frequency encode high cardinality columns
for col in high_card_cols:
    freq_map = X_train_full[col].astype(str).value_counts(normalize=True).to_dict()
    X_train_full[f'{col}_freq'] = X_train_full[col].astype(str).map(freq_map).fillna(0)
    X_test[f'{col}_freq'] = X_test[col].astype(str).map(freq_map).fillna(0)
    # Also label encode for tree-based models
    le = LabelEncoder()
    combined = pd.concat([X_train_full[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train, test_size=0.15, random_state=42)

# Train LightGBM
lgbm_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[]
)
lgbm_val_pred = lgbm_model.predict(X_val)
lgbm_rmse = np.sqrt(np.mean((lgbm_val_pred - y_val) ** 2))
print(f"LightGBM Validation RMSE: {lgbm_rmse:.6f}")

# Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmse',
    verbosity=0
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
xgb_val_pred = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(np.mean((xgb_val_pred - y_val) ** 2))
print(f"XGBoost Validation RMSE: {xgb_rmse:.6f}")

# Train CatBoost
cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0
)
cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
cat_val_pred = cat_model.predict(X_val)
cat_rmse = np.sqrt(np.mean((cat_val_pred - y_val) ** 2))
print(f"CatBoost Validation RMSE: {cat_rmse:.6f}")

# Ensemble: weighted average based on validation performance
rmses = np.array([lgbm_rmse, xgb_rmse, cat_rmse])
weights = (1.0 / rmses) / np.sum(1.0 / rmses)
print(f"Weights - LGBM: {weights[0]:.4f}, XGB: {weights[1]:.4f}, CB: {weights[2]:.4f}")

# Retrain on full training data and predict test
lgbm_full = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgbm_full.fit(X_train_full, y_train)

xgb_full = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmse',
    verbosity=0
)
xgb_full.fit(X_train_full, y_train, verbose=False)

cat_full = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0
)
cat_full.fit(X_train_full, y_train, verbose=0)

# Predict on test set
pred_lgbm = lgbm_full.predict(X_test)
pred_xgb = xgb_full.predict(X_test)
pred_cat = cat_full.predict(X_test)

# Weighted ensemble
final_pred = weights[0] * pred_lgbm + weights[1] * pred_xgb + weights[2] * pred_cat

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'ACA FUL': final_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")