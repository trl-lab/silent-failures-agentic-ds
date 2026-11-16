import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-cash-transfers/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-cash-transfers/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Amount in USD'
feature_cols = [col for col in train_df.columns if col != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Categorical column handling
cat_cols = ['Fund Name', 'Principal Recipient', 'Transfer Quarter', 'Sector/Theme']
for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].fillna('Missing').astype(str)
        X_test[col] = X_test[col].fillna('Missing').astype(str)

# Preprocessing for LightGBM
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    if col in X_lgb.columns:
        X_lgb[col] = X_lgb[col].astype('category')
        X_test_lgb[col] = pd.Categorical(X_test_lgb[col], categories=X_lgb[col].cat.categories)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test_df))

for train_idx, val_idx in kf.split(X):
    X_train_fold, y_train_fold = X_lgb.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X_lgb.iloc[val_idx], y.iloc[val_idx]
    
    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[]
    )
    
    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        random_seed=42,
        verbose=0
    )
    cat_model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
        early_stopping_rounds=50,
        verbose=0
    )
    
    pred_lgb = lgb_model.predict(X_test_lgb)
    pred_cat = cat_model.predict(X_test)
    
    test_preds += (0.5 * pred_lgb + 0.5 * pred_cat) / kf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    target_col: test_preds
})
output_df.to_csv('predictions.csv', index=False)