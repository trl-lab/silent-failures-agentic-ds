import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/summary-of-deposit_wide/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/summary-of-deposit_wide/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'DEPSUM'
y = train_df[target_col].values.astype(np.float64)

X = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col], errors='ignore')

id_cols = [c for c in X.columns if c.lower() in ['id', 'index', 'unnamed: 0']]
X = X.drop(columns=id_cols, errors='ignore')
X_test = X_test.drop(columns=id_cols, errors='ignore')

cols = [c for c in X.columns if c in X_test.columns]
X = X[cols].copy()
X_test = X_test[cols].copy()

for col in cols:
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        X[col + '_year'] = X[col].dt.year
        X[col + '_month'] = X[col].dt.month
        X[col + '_day'] = X[col].dt.day
        X[col + '_dayofweek'] = X[col].dt.dayofweek
        X = X.drop(columns=[col])
        
        X_test[col + '_year'] = X_test[col].dt.year
        X_test[col + '_month'] = X_test[col].dt.month
        X_test[col + '_day'] = X_test[col].dt.day
        X_test[col + '_dayofweek'] = X_test[col].dt.dayofweek
        X_test = X_test.drop(columns=[col])

cat_cols = [c for c in X.columns if X[c].dtype == 'object' or X[c].dtype.name == 'category' or X[c].dtype == 'bool']

for col in cat_cols:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

if cat_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[cat_cols] = encoder.fit_transform(X[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

X = X.astype(np.float32)
X_test = X_test.astype(np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / 5.0
    
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=0,
        thread_count=-1
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test) / 5.0

    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test) / 5.0

final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'DEPSUM': final_preds
})
output_df.to_csv('predictions.csv', index=False)