import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/aijob_ai-ml-ds-salaries/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/aijob_ai-ml-ds-salaries/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'salary_in_usd'
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].values
X_test = test_df.copy()

# Feature preprocessing
cat_cols = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
num_cols = ['work_year', 'remote_ratio']

for col in cat_cols:
    X[col] = X[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
    X_test[col] = X_test[col].fillna(X[col].median())

# For LightGBM, convert object columns to category
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# K-Fold Cross Validation & Ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cb_preds = np.zeros(len(X_test))
lgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        verbose=0,
        random_seed=42 + fold
    )
    cb_model.fit(X.iloc[train_idx], y[train_idx], eval_set=(X.iloc[val_idx], y[val_idx]), early_stopping_rounds=50)
    cb_preds += cb_model.predict(X_test) / kf.n_splits

    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_lgb.iloc[train_idx], y[train_idx],
        eval_set=[(X_lgb.iloc[val_idx], y[val_idx])],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test_lgb) / kf.n_splits

# Blend predictions
final_predictions = 0.5 * cb_preds + 0.5 * lgb_preds

# Save submission
submission = pd.DataFrame({
    'id': range(len(final_predictions)),
    'salary_in_usd': final_predictions
})
submission.to_csv('predictions.csv', index=False)