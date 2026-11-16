import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/diamonds/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/diamonds/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate target
y_train = train_df['price'].values
X_train = train_df.drop(columns=['price'])
X_test = test_df.copy()

# Feature Engineering
def preprocess_features(df):
    df = df.copy()
    
    # Ordinal mappings for diamond categorical features
    cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    
    df['cut_num'] = df['cut'].astype(str).map(cut_map).fillna(-1)
    df['color_num'] = df['color'].astype(str).map(color_map).fillna(-1)
    df['clarity_num'] = df['clarity'].astype(str).map(clarity_map).fillna(-1)
    
    # Volume and dimension interaction features
    df['volume'] = df['x'] * df['y'] * df['z']
    df['density_proxy'] = df['carat'] / (df['volume'] + 1e-5)
    df['xy_ratio'] = df['x'] / (df['y'] + 1e-5)
    df['xz_ratio'] = df['x'] / (df['z'] + 1e-5)
    df['yz_ratio'] = df['y'] / (df['z'] + 1e-5)
    
    # Convert original categories to category dtype for LightGBM
    for col in ['cut', 'color', 'clarity']:
        df[col] = df[col].astype('category')
        
    return df

X_train_proc = preprocess_features(X_train)
X_test_proc = preprocess_features(X_test)

# 5-fold cross validation with LightGBM and CatBoost
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_proc))
cb_preds = np.zeros(len(X_test_proc))

categorical_features = ['cut', 'color', 'clarity']

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train[val_idx]
    
    # LightGBM Model
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
    lgb_preds += lgb_model.predict(X_test_proc) / kf.n_splits

    # CatBoost Model
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=6,
        cat_features=categorical_features,
        random_seed=42 + fold,
        verbose=False
    )
    # Ensure categorical columns are strings for CatBoost
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test_proc.copy()
    for col in categorical_features:
        X_tr_cb[col] = X_tr_cb[col].astype(str)
        X_va_cb[col] = X_va_cb[col].astype(str)
        X_te_cb[col] = X_te_cb[col].astype(str)
        
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_te_cb) / kf.n_splits

# Ensemble predictions
final_preds = 0.5 * lgb_preds + 0.5 * cb_preds
final_preds = np.clip(final_preds, 0, None)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'price': final_preds
})
output_df.to_csv('predictions.csv', index=False)