import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/broadband-availability/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/broadband-availability/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Population for whom broadband available, 2019 (%)'

# 2. Preprocessing
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Drop identifier text columns
drop_cols = ['GEO_ID', 'NAME']
X = X.drop(columns=[c for c in drop_cols if c in X.columns])
X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

# Clean column names to remove special characters for LightGBM / XGBoost
def clean_col_name(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name))

X.columns = [clean_col_name(c) for c in X.columns]
X_test.columns = [clean_col_name(c) for c in X_test.columns]

# Handle categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str).astype('category')
    X_test[col] = X_test[col].astype(str).astype('category')

# 3. Model Training with K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

# XGBoost copy with encoded categories
X_xgb = X.copy()
X_test_xgb = X_test.copy()
for col in cat_cols:
    X_xgb[col] = X_xgb[col].cat.codes
    X_test_xgb[col] = X_test_xgb[col].cat.codes

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        verbose=0,
        random_state=42
    )
    cat_model.fit(X_tr, y_tr)
    cat_preds += cat_model.predict(X_test) / kf.n_splits

    # XGBoost
    X_tr_xgb = X_xgb.iloc[train_idx]
    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        random_state=42,
        tree_method='hist'
    )
    xgb_model.fit(X_tr_xgb, y_tr)
    xgb_preds += xgb_model.predict(X_test_xgb) / kf.n_splits

# 4. Ensemble predictions and post-process
final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0
final_preds = np.clip(final_preds, 0.0, 100.0)

# 5. Save submission
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    target_col: final_preds
})

submission.to_csv('predictions.csv', index=False)