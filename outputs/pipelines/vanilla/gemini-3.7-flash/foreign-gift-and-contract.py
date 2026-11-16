import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/foreign-gift-and-contract/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/foreign-gift-and-contract/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Amount'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

def to_seconds(s):
    try:
        s_s = s.astype('datetime64[s]')
    except Exception:
        s_s = pd.to_datetime(s, errors='coerce').astype('datetime64[s]')
    arr = s_s.to_numpy()
    is_nat = pd.isna(arr)
    secs = np.where(is_nat, np.nan, arr.astype('int64'))
    return pd.Series(secs, index=s.index, dtype='float64')

def extract_features(df):
    df = df.copy()
    
    date_cols = ['Receipt Date', 'Contract Start Date', 'Contract End Date', 'Foreign Source Ownership Date']
    date_secs = {}
    
    for col in date_cols:
        if col in df.columns:
            s = df[col]
            try:
                s_s = s.astype('datetime64[s]')
            except Exception:
                s_s = pd.to_datetime(s, errors='coerce').astype('datetime64[s]')
            
            df[f'{col}_year'] = s_s.dt.year
            df[f'{col}_month'] = s_s.dt.month
            df[f'{col}_day'] = s_s.dt.day
            df[f'{col}_dayofweek'] = s_s.dt.dayofweek
            df[f'{col}_is_na'] = s_s.isna().astype(int)
            date_secs[col] = to_seconds(s)
    
    if 'Contract Start Date' in date_secs and 'Contract End Date' in date_secs:
        df['contract_duration_days'] = (date_secs['Contract End Date'] - date_secs['Contract Start Date']) / 86400.0
        
    if 'Receipt Date' in date_secs and 'Contract Start Date' in date_secs:
        df['receipt_start_diff_days'] = (date_secs['Receipt Date'] - date_secs['Contract Start Date']) / 86400.0

    if 'Receipt Date' in date_secs and 'Contract End Date' in date_secs:
        df['receipt_end_diff_days'] = (date_secs['Receipt Date'] - date_secs['Contract End Date']) / 86400.0

    text_cols = ['Restricted Transaction Description', 'Foreign Source Owner Name', 
                 'Restricted Transaction Foreign Government Legal Name', 'Restricted Transaction Foreign Government Name']
    for col in text_cols:
        if col in df.columns:
            str_s = df[col].astype(str).fillna('')
            df[f'{col}_len'] = str_s.str.len()
            df[f'{col}_word_count'] = str_s.apply(lambda x: len(x.split()))
            
    df = df.drop(columns=[c for c in date_cols if c in df.columns])
    return df

X_train = extract_features(X_train)
X_test = extract_features(X_test)

cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

for col in cat_cols:
    X_train[col] = X_train[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')
    
    le = LabelEncoder()
    le.fit(pd.concat([X_train[col], X_test[col]], axis=0))
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 800
}

for train_idx, val_idx in kf.split(X_train, y_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits
    
    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        random_seed=42,
        verbose=0
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test) / kf.n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Amount': final_preds
})

submission.to_csv('predictions.csv', index=False)