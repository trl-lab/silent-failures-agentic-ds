import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

def preprocess_features(df_train, df_test):
    tr = df_train.copy()
    te = df_test.copy()
    
    cols = list(tr.columns)
    for c in cols:
        sample = tr[c].dropna().iloc[0] if len(tr[c].dropna()) > 0 else None
        
        if isinstance(sample, (list, np.ndarray, tuple)):
            tr[c + '_len'] = tr[c].apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
            te[c + '_len'] = te[c].apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
            tr.drop(columns=[c], inplace=True)
            te.drop(columns=[c], inplace=True)
            continue
        elif isinstance(sample, dict):
            tr.drop(columns=[c], inplace=True)
            te.drop(columns=[c], inplace=True)
            continue
            
        if tr[c].dtype == 'object':
            try:
                converted = pd.to_datetime(tr[c], errors='raise')
                tr[c] = converted
                te[c] = pd.to_datetime(te[c], errors='coerce')
            except Exception:
                pass
                
        if pd.api.types.is_datetime64_any_dtype(tr[c]):
            for col_df, name in [(tr, 'tr'), (te, 'te')]:
                col_df[c + '_year'] = col_df[c].dt.year
                col_df[c + '_month'] = col_df[c].dt.month
                col_df[c + '_day'] = col_df[c].dt.day
                col_df[c + '_dayofweek'] = col_df[c].dt.dayofweek
                col_df[c + '_hour'] = col_df[c].dt.hour
            tr.drop(columns=[c], inplace=True)
            te.drop(columns=[c], inplace=True)
            continue
            
        if tr[c].dtype == 'object' or pd.api.types.is_categorical_dtype(tr[c]) or str(tr[c].dtype) == 'category':
            tr[c] = tr[c].astype(str)
            te[c] = te[c].astype(str)
            
            freq = tr[c].value_counts().to_dict()
            tr[c + '_freq'] = tr[c].map(freq).fillna(0)
            te[c + '_freq'] = te[c].map(freq).fillna(0)
            
            categories = pd.concat([tr[c], te[c]]).unique()
            mapping = {val: idx for idx, val in enumerate(categories)}
            tr[c] = tr[c].map(mapping).fillna(-1).astype(int)
            te[c] = te[c].map(mapping).fillna(-1).astype(int)
            
    for col in tr.columns:
        tr[col] = pd.to_numeric(tr[col], errors='coerce')
        te[col] = pd.to_numeric(te[col], errors='coerce')
        
    return tr, te

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/scrabble-player-rating/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/scrabble-player-rating/data/test.parquet'
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    target_col = 'rating'
    y = train_df[target_col].values
    X_train = train_df.drop(columns=[target_col], errors='ignore')
    X_test = test_df.copy()
    
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    X_train_proc, X_test_proc = preprocess_features(X_train, X_test)
    
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 2500,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    cb_params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'iterations': 2500,
        'learning_rate': 0.04,
        'depth': 6,
        'random_seed': 42,
        'verbose': 0
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    preds_lgb = np.zeros(len(X_test_proc))
    preds_cb = np.zeros(len(X_test_proc))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y)):
        X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]
        
        # LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds_lgb += model_lgb.predict(X_test_proc) / kf.n_splits
        
        # CatBoost
        model_cb = CatBoostRegressor(**cb_params)
        model_cb.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False
        )
        preds_cb += model_cb.predict(X_test_proc) / kf.n_splits
        
    final_preds = 0.5 * preds_lgb + 0.5 * preds_cb
    
    submission = pd.DataFrame({
        'id': np.arange(len(final_preds)),
        'rating': final_preds
    })
    
    submission.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()