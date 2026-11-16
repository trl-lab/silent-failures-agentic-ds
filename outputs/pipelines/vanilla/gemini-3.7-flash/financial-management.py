import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-management/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/financial-management/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'State Share'
y_train = train_df[target_col].values

# Feature engineering
def preprocess(df):
    data = df.copy()
    
    # Extract latitude and longitude from Location string
    coords = data['Location'].astype(str).str.extract(r'\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)')
    data['latitude'] = coords[0].astype(float)
    data['longitude'] = coords[1].astype(float)
    
    # Numerical interactions
    data['diff_fed_med'] = data['Federal Share'] - data['Federal Share Medicaid']
    data['sum_fed_shares'] = (
        data['Federal Share Medicaid'].fillna(0) +
        data['Federal Share ARRA'].fillna(0) +
        data['Federal Share BIPP'].fillna(0)
    )
    data['diff_fed_sum'] = data['Federal Share'] - data['sum_fed_shares']
    
    # Categorical columns
    cat_cols = ['State', 'Program', 'Service Category']
    for col in cat_cols:
        data[col] = data[col].astype('category')
        
    cols_to_drop = ['Location']
    if target_col in data.columns:
        cols_to_drop.append(target_col)
        
    data = data.drop(columns=cols_to_drop)
    return data

X_train = preprocess(train_df)
X_test = preprocess(test_df)

cat_features = ['State', 'Program', 'Service Category']

# K-Fold Cross-Validation & Prediction
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test) / n_splits
    
    # CatBoost
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_test_cb = X_test.copy()
    for col in cat_features:
        X_tr_cb[col] = X_tr_cb[col].astype(str).fillna('missing')
        X_va_cb[col] = X_va_cb[col].astype(str).fillna('missing')
        X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')
        
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_features,
        random_seed=42 + fold,
        verbose=False
    )
    cat_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50)
    cat_preds += cat_model.predict(X_test_cb) / n_splits

final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'State Share': final_predictions
})
output_df.to_csv('predictions.csv', index=False)