import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/conflict-events_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/conflict-events_wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'fatalities'
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].values
X_test = test_df[feature_cols].copy()

# Identify categorical and numerical columns
cat_cols = [c for c in feature_cols if X[c].dtype == 'object' or str(X[c].dtype) == 'category']
num_cols = [c for c in feature_cols if c not in cat_cols]

# Handle categorical missing values and convert to string category
for c in cat_cols:
    X[c] = X[c].fillna('missing').astype(str).astype('category')
    X_test[c] = X_test[c].fillna('missing').astype(str).astype('category')

# K-Fold Cross Validation & Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

# Format for CatBoost
X_cb = X.copy()
X_test_cb = X_test.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb.predict(X_test) / 5.0
    
    # CatBoost
    cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        verbose=0,
        random_seed=42
    )
    cb.fit(
        X_cb.iloc[train_idx], y_tr,
        eval_set=(X_cb.iloc[val_idx], y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb.predict(X_test_cb) / 5.0

# Blend predictions
final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

# Clip at minimum observed training target (usually 0 for fatalities)
min_target = y.min()
final_preds = np.clip(final_preds, a_min=min_target, a_max=None)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_preds)),
    'fatalities': final_preds
})
output_df.to_csv('predictions.csv', index=False)