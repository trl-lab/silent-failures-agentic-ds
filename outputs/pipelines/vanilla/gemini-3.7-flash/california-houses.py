import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/california-houses/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/california-houses/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Total interior livable area'
y = train_df[target_col].values

train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined = pd.concat([train_df, test_df], ignore_index=True)

combined['Summary_len'] = combined['Summary'].fillna('').astype(str).str.len()
combined['Summary_word_count'] = combined['Summary'].fillna('').astype(str).apply(lambda x: len(x.split()))

def extract_bedrooms(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.search(r'^\d+(\.\d+)?', val_str)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return np.nan
    return np.nan

combined['Bedrooms_num'] = combined['Bedrooms'].apply(extract_bedrooms)

for date_col in ['Listed On', 'Last Sold On']:
    dt = pd.to_datetime(combined[date_col], errors='coerce')
    combined[f'{date_col}_year'] = dt.dt.year
    combined[f'{date_col}_month'] = dt.dt.month
    combined[f'{date_col}_dayofweek'] = dt.dt.dayofweek
    combined[f'{date_col}_epoch_days'] = (dt - pd.Timestamp("2000-01-01")).dt.days

combined['days_since_last_sold'] = combined['Listed On_epoch_days'] - combined['Last Sold On_epoch_days']
combined['price_diff_listed_sold'] = combined['Sold Price'] - combined['Listed Price']
combined['price_ratio_listed_sold'] = combined['Sold Price'] / (combined['Listed Price'] + 1e-5)
combined['tax_to_price_ratio'] = combined['Annual tax amount'] / (combined['Sold Price'] + 1e-5)
combined['assessed_to_price_ratio'] = combined['Tax assessed value'] / (combined['Sold Price'] + 1e-5)

school_score_cols = ['Elementary School Score', 'Middle School Score', 'High School Score']
school_dist_cols = ['Elementary School Distance', 'Middle School Distance', 'High School Distance']

combined['mean_school_score'] = combined[school_score_cols].mean(axis=1)
combined['min_school_score'] = combined[school_score_cols].min(axis=1)
combined['max_school_score'] = combined[school_score_cols].max(axis=1)

combined['mean_school_dist'] = combined[school_dist_cols].mean(axis=1)
combined['min_school_dist'] = combined[school_dist_cols].min(axis=1)
combined['max_school_dist'] = combined[school_dist_cols].max(axis=1)

combined['total_baths'] = combined['Bathrooms'].fillna(0) + combined['Full bathrooms'].fillna(0)

drop_cols = ['Id', 'Address', 'Summary', 'Listed On', 'Last Sold On']
combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])

cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
if 'Zip' in combined.columns and 'Zip' not in cat_cols:
    cat_cols.append('Zip')

for c in cat_cols:
    combined[c] = combined[c].astype(str).astype('category')

X_train = combined.iloc[:n_train].copy()
X_test = combined.iloc[n_train:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_oof = np.zeros(n_train)
lgb_preds = np.zeros(len(X_test))

cb_oof = np.zeros(n_train)
cb_preds = np.zeros(len(X_test))

X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for c in cat_cols:
    X_train_cb[c] = X_train_cb[c].astype(str).fillna('missing')
    X_test_cb[c] = X_test_cb[c].astype(str).fillna('missing')

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
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
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    lgb_oof[val_idx] = lgb_model.predict(X_va)
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    cb_tr = X_train_cb.iloc[train_idx]
    cb_va = X_train_cb.iloc[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    
    cb_model.fit(
        cb_tr, y_tr,
        eval_set=(cb_va, y_va),
        use_best_model=True
    )
    
    cb_oof[val_idx] = cb_model.predict(cb_va)
    cb_preds += cb_model.predict(X_test_cb) / kf.n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Total interior livable area': final_preds
})

output_df.to_csv('predictions.csv', index=False)