import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/contract-awards-investment-project-financing/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/contract-awards-investment-project-financing/data/test.parquet')

target = 'Supplier Contract Amount (USD)'

# Drop columns that are too high-cardinality or not useful
drop_cols = ['Project Name', 'Contract Description', 'Borrower Contract Reference Number', 'Supplier', 'As of Date', 'Contract Signing Date']
drop_cols = [c for c in drop_cols if c in train_df.columns]

train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

# Identify column types
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
if target in numeric_cols:
    numeric_cols.remove(target)

categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Combine train and test to fit encoder
    combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Prepare features
X = train_df.drop(columns=[target])
y = train_df[target]
X_test = test_df.copy()

# Handle missing values in numeric columns - fill with -999 (LGBM handles natively but let's be safe)
for col in numeric_cols:
    if col in X.columns:
        X[col] = X[col].fillna(-999)
        X_test[col] = X_test[col].fillna(-999)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Model 1: LightGBM
lgbm_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse',
               callbacks=[])
lgbm_val_pred = lgbm_model.predict(X_val)
lgbm_val_rmse = np.sqrt(mean_squared_error(y_val, lgbm_val_pred))
print(f"LightGBM Val RMSE: {lgbm_val_rmse:.6f}")

# Model 2: XGBoost
xgb_model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='rmse',
    verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_val_pred = xgb_model.predict(X_val)
xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
print(f"XGBoost Val RMSE: {xgb_val_rmse:.6f}")

# Model 3: CatBoost
cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    subsample=0.8,
    random_state=42,
    verbose=0
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
cat_val_pred = cat_model.predict(X_val)
cat_val_rmse = np.sqrt(mean_squared_error(y_val, cat_val_pred))
print(f"CatBoost Val RMSE: {cat_val_rmse:.6f}")

# Ensemble: weighted average based on validation performance
rmses = np.array([lgbm_val_rmse, xgb_val_rmse, cat_val_rmse])
weights = (1.0 / rmses) ** 2
weights = weights / weights.sum()
print(f"Weights: LGBM={weights[0]:.4f}, XGB={weights[1]:.4f}, CB={weights[2]:.4f}")

# Retrain on full training data for final predictions
lgbm_final = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1
)
lgbm_final.fit(X, y)

xgb_final = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='rmse',
    verbosity=0
)
xgb_final.fit(X, y, verbose=False)

cat_final = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    subsample=0.8,
    random_state=42,
    verbose=0
)
cat_final.fit(X, y, verbose=0)

# Final predictions
pred_lgbm = lgbm_final.predict(X_test)
pred_xgb = xgb_final.predict(X_test)
pred_cat = cat_final.predict(X_test)

final_pred = weights[0] * pred_lgbm + weights[1] * pred_xgb + weights[2] * pred_cat

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'Supplier Contract Amount (USD)': final_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")