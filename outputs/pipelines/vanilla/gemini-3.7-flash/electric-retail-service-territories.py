import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/electric-retail-service-territories/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/electric-retail-service-territories/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'TOTAL_MWH'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Feature Engineering
def preprocess_features(df):
    df = df.copy()
    
    # Parse VAL_DATE if available
    if 'VAL_DATE' in df.columns:
        dt_series = pd.to_datetime(df['VAL_DATE'], errors='coerce')
        df['VAL_YEAR'] = dt_series.dt.year.fillna(-1).astype(int)
        df['VAL_MONTH'] = dt_series.dt.month.fillna(-1).astype(int)
        df['VAL_DAY'] = dt_series.dt.day.fillna(-1).astype(int)
    
    # Shape features
    if 'Shape__Area' in df.columns and 'Shape__Length' in df.columns:
        df['log_shape_area'] = np.log1p(np.maximum(df['Shape__Area'].fillna(0), 0))
        df['log_shape_length'] = np.log1p(np.maximum(df['Shape__Length'].fillna(0), 0))
        df['area_to_length_ratio'] = df['Shape__Area'] / (df['Shape__Length'] + 1e-5)
        df['compactness'] = (df['Shape__Length'] ** 2) / (df['Shape__Area'] + 1e-5)
        
    # Text length features
    for col in ['NAME', 'ADDRESS', 'CITY', 'SOURCE', 'WEBSITE']:
        if col in df.columns:
            df[f'{col}_len'] = df[col].astype(str).apply(len)
            
    return df

X_train_proc = preprocess_features(X_train)
X_test_proc = preprocess_features(X_test)

# Identify categorical and numerical columns
cat_cols = []
num_cols = []

for col in X_train_proc.columns:
    if X_train_proc[col].dtype == 'object' or str(X_train_proc[col].dtype) == 'category':
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Fill categorical missing values and convert to category/string
for col in cat_cols:
    X_train_proc[col] = X_train_proc[col].fillna('MISSING').astype(str)
    X_test_proc[col] = X_test_proc[col].fillna('MISSING').astype(str)

# For LightGBM, use category dtype
X_train_lgb = X_train_proc.copy()
X_test_lgb = X_test_proc.copy()
for col in cat_cols:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# K-Fold Ensemble Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test_proc))
cb_test_preds = np.zeros(len(X_test_proc))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc)):
    # LightGBM
    X_tr_lgb, y_tr = X_train_lgb.iloc[train_idx], y_train[train_idx]
    X_va_lgb, y_va = X_train_lgb.iloc[val_idx], y_train[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_test_preds += lgb_model.predict(X_test_lgb) / kf.n_splits
    
    # CatBoost
    X_tr_cb = X_train_proc.iloc[train_idx]
    X_va_cb = X_train_proc.iloc[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_test_preds += cb_model.predict(X_test_proc) / kf.n_splits

# Ensemble predictions
final_preds = 0.5 * lgb_test_preds + 0.5 * cb_test_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds), dtype=int),
    'TOTAL_MWH': final_preds
})
output_df.to_csv('predictions.csv', index=False)