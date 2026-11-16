import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-funding-decisions/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-funding-decisions/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Amount in USD'
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].values
X_test = test_df.copy()

# Identify column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

# Preprocess categorical features for CatBoost and LightGBM
for col in categorical_cols:
    X[col] = X[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)

# Prepare categorical dtype for LightGBM
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in categorical_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# K-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

cb_test_preds = np.zeros(len(X_test))
lgb_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y[val_idx]
    
    # Train CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=categorical_cols,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f), early_stopping_rounds=50, verbose=False)
    cb_test_preds += cb_model.predict(X_test) / n_splits
    
    # Train LightGBM
    X_train_lgb, y_train_lgb = X_lgb.iloc[train_idx], y[train_idx]
    X_val_lgb, y_val_lgb = X_lgb.iloc[val_idx], y[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1
    )
    lgb_model.fit(
        X_train_lgb, y_train_lgb,
        eval_set=[(X_val_lgb, y_val_lgb)],
        callbacks=[]
    )
    lgb_test_preds += lgb_model.predict(X_test_lgb) / n_splits

# Ensemble predictions
final_preds = 0.5 * cb_test_preds + 0.5 * lgb_test_preds

# Save submission
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'Amount in USD': final_preds
})
output_df.to_csv('predictions.csv', index=False)