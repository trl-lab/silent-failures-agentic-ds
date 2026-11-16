import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/prison-boundaries/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/prison-boundaries/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'CAPACITY'
y = train_df[target_col].values

# Feature engineering
def preprocess(df):
    df_feat = pd.DataFrame(index=df.index)
    
    # Numerical / Geometric features
    area = df['Shape__Area'].fillna(0).values
    length = df['Shape__Length'].fillna(0).values
    df_feat['Shape__Area'] = area
    df_feat['Shape__Length'] = length
    df_feat['log_Area'] = np.log1p(np.maximum(0, area))
    df_feat['log_Length'] = np.log1p(np.maximum(0, length))
    df_feat['compactness'] = area / (length**2 + 1e-9)
    
    # ID & Numeric codes
    df_feat['ZIP'] = pd.to_numeric(df['ZIP'], errors='coerce').fillna(-1)
    df_feat['ZIP4'] = pd.to_numeric(df['ZIP4'], errors='coerce').fillna(-1)
    df_feat['COUNTYFIPS'] = pd.to_numeric(df['COUNTYFIPS'], errors='coerce').fillna(-1)
    df_feat['OBJECTID'] = pd.to_numeric(df['OBJECTID'], errors='coerce').fillna(-1)
    df_feat['FACILITYID'] = pd.to_numeric(df['FACILITYID'], errors='coerce').fillna(-1)
    
    # Dates
    for col in ['SOURCEDATE', 'VAL_DATE']:
        dt = pd.to_datetime(df[col], errors='coerce')
        df_feat[f'{col}_year'] = dt.dt.year.fillna(-1)
        df_feat[f'{col}_month'] = dt.dt.month.fillna(-1)
        df_feat[f'{col}_day'] = dt.dt.day.fillna(-1)
        df_feat[f'{col}_epoch'] = dt.astype('int64') // 10**9
        df_feat[f'{col}_epoch'] = df_feat[f'{col}_epoch'].replace({-9223372037: -1})
    
    # Categoricals
    cat_cols = ['STATE', 'TYPE', 'STATUS', 'COUNTY', 'SECURELVL', 'VAL_METHOD', 'CITY', 'COUNTRY']
    for col in cat_cols:
        df_feat[col] = df[col].astype(str).fillna('missing').astype('category')
        
    # Text length / simple metadata
    df_feat['NAME_len'] = df['NAME'].astype(str).str.len()
    df_feat['ADDRESS_len'] = df['ADDRESS'].astype(str).str.len()
    df_feat['has_phone'] = (~df['TELEPHONE'].isna() & (df['TELEPHONE'] != 'NOT AVAILABLE')).astype(int)
    df_feat['has_website'] = (~df['WEBSITE'].isna() & (df['WEBSITE'] != 'NOT AVAILABLE')).astype(int)
    
    return df_feat

X = preprocess(train_df)
X_test = preprocess(test_df)

# Align categories
cat_cols = [col for col in X.columns if X[col].dtype.name == 'category']
for col in cat_cols:
    categories = list(set(X[col].dropna().unique()).union(set(X_test[col].dropna().unique())))
    X[col] = pd.Categorical(X[col], categories=categories)
    X_test[col] = pd.Categorical(X_test[col], categories=categories)

# K-Fold CV Ensembling
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

# Prepare data for CatBoost (convert category to str)
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    X_cb[col] = X_cb[col].astype(str)
    X_test_cb[col] = X_test_cb[col].astype(str)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'random_state': 42,
    'n_estimators': 1000,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # LightGBM
    X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / n_splits
    
    # CatBoost
    X_train_cb, y_train_cb = X_cb.iloc[train_idx], y[train_idx]
    X_val_cb, y_val_cb = X_cb.iloc[val_idx], y[val_idx]
    
    model_cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=False
    )
    model_cb.fit(
        X_train_cb, y_train_cb,
        eval_set=(X_val_cb, y_val_cb),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += model_cb.predict(X_test_cb) / n_splits

final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'CAPACITY': final_predictions
})
output_df.to_csv('predictions.csv', index=False)