import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/terms-cc-plans/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/terms-cc-plans/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Drop missing target rows if any
train_df = train_df.dropna(subset=['APR']).reset_index(drop=True)

y_train = train_df['APR'].values
n_train = len(train_df)
n_test = len(test_df)

# Feature engineering function
def extract_features(df):
    df_feat = pd.DataFrame(index=df.index)
    
    # Date features
    dates = pd.to_datetime(df['Agreement Date'], errors='coerce')
    df_feat['year'] = dates.dt.year
    df_feat['month'] = dates.dt.month
    df_feat['day'] = dates.dt.day
    df_feat['dayofweek'] = dates.dt.dayofweek
    df_feat['dayofyear'] = dates.dt.dayofyear
    df_feat['year_frac'] = dates.dt.year + (dates.dt.dayofyear - 1) / 365.25
    
    # Phone area code
    phone_str = df['Telephone Number for Consumers'].astype(str)
    df_feat['area_code'] = phone_str.str.extract(r'\(?(\d{3})\)?')[0].fillna('missing')
    
    # Numerical features
    num_cols = ['Grace Period', 'Annual Fee', 'Late Fee']
    for col in num_cols:
        if col in df.columns:
            df_feat[col] = df[col].astype(float)
            df_feat[f'{col}_is_na'] = df[col].isna().astype(int)
            
    # Additional numerical interactions
    df_feat['total_fees'] = df_feat['Annual Fee'].fillna(0) + df_feat['Late Fee'].fillna(0)
    
    # Categorical features
    cat_cols = ['Institution Name', 'Plan Name', 'Availability of Credit Card Plan', 
                'Location', 'Index', 'Variable Rate Index', 'area_code']
    
    for col in cat_cols:
        if col in df.columns:
            df_feat[col] = df[col].astype(str).fillna('missing')
        elif col in df_feat.columns:
            df_feat[col] = df_feat[col].astype(str).fillna('missing')
            
    return df_feat, cat_cols

train_features, cat_cols = extract_features(train_df)
test_features, _ = extract_features(test_df)

# Combine for consistent categorical encoding
all_features = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

# For LightGBM: convert categorical columns to category dtype
lgb_all_features = all_features.copy()
for col in cat_cols:
    lgb_all_features[col] = lgb_all_features[col].astype('category')

X_train_lgb = lgb_all_features.iloc[:n_train].copy()
X_test_lgb = lgb_all_features.iloc[n_train:].copy()

# For CatBoost: keep categorical as strings
cb_all_features = all_features.copy()
for col in cat_cols:
    cb_all_features[col] = cb_all_features[col].astype(str)

X_train_cb = cb_all_features.iloc[:n_train].copy()
X_test_cb = cb_all_features.iloc[n_train:].copy()

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(n_test)
cb_preds = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_lgb, y_train)):
    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train_lgb.iloc[train_idx], y_train[train_idx],
        eval_set=[(X_train_lgb.iloc[val_idx], y_train[val_idx])],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test_lgb) / kf.n_splits

    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=False,
        thread_count=-1
    )
    
    cb_model.fit(
        X_train_cb.iloc[train_idx], y_train[train_idx],
        eval_set=(X_train_cb.iloc[val_idx], y_train[val_idx]),
        early_stopping_rounds=100,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / kf.n_splits

# Ensemble blend
final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

# Save submission
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'APR': final_predictions
})
output_df.to_csv('predictions.csv', index=False)