import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-commitments/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-intermediary-funds-commitments/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Amount in USD'
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].values
X_test = test_df.copy()

categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or str(X[col].dtype) == 'category']

# Preprocessing for LightGBM and CatBoost
for col in categorical_cols:
    X[col] = X[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)

# LightGBM requires category dtype
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in categorical_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X_lgb.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X_lgb.iloc[val_idx], y[val_idx]
    
    # Train LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test_lgb) / n_splits
    
    # Train CatBoost
    cat_train_fold = X.iloc[train_idx]
    cat_val_fold = X.iloc[val_idx]
    
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=categorical_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        cat_train_fold, y_train_fold,
        eval_set=(cat_val_fold, y_val_fold),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test) / n_splits

final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Amount in USD': final_predictions
})
output_df.to_csv('predictions.csv', index=False)