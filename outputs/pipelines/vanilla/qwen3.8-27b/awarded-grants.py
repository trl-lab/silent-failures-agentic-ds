import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/awarded-grants/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/awarded-grants/data/test.parquet')

# Define columns to drop (high-cardinality text, IDs, redundant info)
drop_cols = [
    'Grantee Address',
    'Grantee Name',
    'Grant Number',
    'Grant Serial Number',
    'Project Period Start Date Text String',
    'Grant Project Period End Date Text',
    'Congressional District Name',
    'U.S. Congressional Representative Name',
    'State and County Federal Information Processing Standard Code',
    'Name of U.S. Senator Number One',
    'Name of U.S. Senator Number Two',
    'Uniform Data System Grant Program Description',
    'DUNS Number',
    'Unique Entity Identifier',
    'Data Warehouse Record Create Date Text',
    'Grantee ZIP Code',
    'Complete County Name',
    'State Name',
    'HRSA Region',
    'HRSA Program Area Name',
    'Grant Program Name',
    'Congressional District Number',
    'HHS Region Number',
    'State FIPS Code',
    'U.S. - Mexico Border 100 Kilometer Indicator',
    'U.S. - Mexico Border County Indicator',
]

target = 'Financial Assistance'
drop_cols = [c for c in drop_cols if c in train_df.columns and c != target]

X_train = train_df.drop(columns=[target] + drop_cols).copy()
y_train = train_df[target].copy()
X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).copy()

# Convert datetime columns to numeric features
datetime_cols = ['Project Period Start Date', 'Grant Project Period End Date', 'Data Warehouse Record Create Date']
for col in datetime_cols:
    if col in X_train.columns:
        X_train[col + '_year'] = X_train[col].dt.year
        X_train[col + '_month'] = X_train[col].dt.month
        X_train = X_train.drop(columns=[col])
    if col in X_test.columns:
        X_test[col + '_year'] = X_test[col].dt.year
        X_test[col + '_month'] = X_test[col].dt.month
        X_test = X_test.drop(columns=[col])

# Compute project duration
if 'Project Period Start Date_year' in X_train.columns and 'Grant Project Period End Date_year' in X_train.columns:
    X_train['project_duration_years'] = X_train['Grant Project Period End Date_year'] - X_train['Project Period Start Date_year']
    X_test['project_duration_years'] = X_test['Grant Project Period End Date_year'] - X_test['Project Period Start Date_year']

# Identify categorical and numeric columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Fill missing values in numeric columns
for col in numeric_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Encode categorical columns using LabelEncoder
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str).fillna('MISSING')
    X_test[col] = X_test[col].astype(str).fillna('MISSING')
    
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    
    # For test, handle unseen categories by mapping to -1
    test_vals = X_test[col].values
    known_cats = set(le.classes_)
    encoded = np.array([le.transform([v])[0] if v in known_cats else -1 for v in test_vals])
    X_test[col] = encoded

# Ensure all columns are numeric
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Train LightGBM
lgbm_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_tr, y_tr)
lgbm_val_pred = lgbm_model.predict(X_val)

# Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmse'
)
xgb_model.fit(X_tr, y_tr)
xgb_val_pred = xgb_model.predict(X_val)

# Train CatBoost
cb_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0
)
cb_model.fit(X_tr, y_tr)
cb_val_pred = cb_model.predict(X_val)

# Evaluate
lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_val_pred))
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
cb_rmse = np.sqrt(mean_squared_error(y_val, cb_val_pred))

print(f"LightGBM RMSE: {lgbm_rmse:.4f}")
print(f"XGBoost RMSE: {xgb_rmse:.4f}")
print(f"CatBoost RMSE: {cb_rmse:.4f}")

# Ensemble prediction on test
lgbm_test_pred = lgbm_model.predict(X_test)
xgb_test_pred = xgb_model.predict(X_test)
cb_test_pred = cb_model.predict(X_test)

# Weighted ensemble (inverse RMSE weighting)
weights = np.array([1/lgbm_rmse, 1/xgb_rmse, 1/cb_rmse])
weights = weights / weights.sum()

final_pred = weights[0] * lgbm_test_pred + weights[1] * xgb_test_pred + weights[2] * cb_test_pred

# Save predictions
predictions = pd.DataFrame({
    'id': range(len(final_pred)),
    'Financial Assistance': final_pred
})
predictions.to_csv('predictions.csv', index=False)