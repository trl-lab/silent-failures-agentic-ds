import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/miami-housing/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/miami-housing/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

drop_cols = ['PARCELNO', 'SALE_PRC']
feature_cols = [c for c in train.columns if c not in drop_cols]

def add_features(df):
    df = df.copy()
    df['lvg_to_lnd_ratio'] = df['TOT_LVG_AREA'] / (df['LND_SQFOOT'] + 1.0)
    df['spec_to_lvg_ratio'] = df['SPEC_FEAT_VAL'] / (df['TOT_LVG_AREA'] + 1.0)
    df['spec_to_lnd_ratio'] = df['SPEC_FEAT_VAL'] / (df['LND_SQFOOT'] + 1.0)
    
    skewed = ['LND_SQFOOT', 'TOT_LVG_AREA', 'SPEC_FEAT_VAL', 'RAIL_DIST', 
              'OCEAN_DIST', 'WATER_DIST', 'CNTR_DIST', 'SUBCNTR_DI', 'HWY_DIST']
    for col in skewed:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(np.maximum(0, df[col]))
            
    df['lat_plus_lon'] = df['LATITUDE'] + df['LONGITUDE']
    df['lat_minus_lon'] = df['LATITUDE'] - df['LONGITUDE']
    df['lat_lon_prod'] = df['LATITUDE'] * df['LONGITUDE']
    
    for angle in [15, 30, 45, 60]:
        rad = np.radians(angle)
        df[f'rot_{angle}_x'] = df['LATITUDE'] * np.cos(rad) - df['LONGITUDE'] * np.sin(rad)
        df[f'rot_{angle}_y'] = df['LATITUDE'] * np.sin(rad) + df['LONGITUDE'] * np.cos(rad)
        
    return df

X_train = add_features(train[feature_cols])
X_test = add_features(test[feature_cols])
y_train_log = np.log1p(train['SALE_PRC'])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_log)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_log.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train_log.iloc[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += np.expm1(model_lgb.predict(X_test)) / kf.n_splits
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += np.expm1(model_xgb.predict(X_test)) / kf.n_splits
    
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.04,
        depth=7,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    cat_preds += np.expm1(model_cat.predict(X_test)) / kf.n_splits

final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cat_preds

output = pd.DataFrame({
    'id': np.arange(len(test)),
    'SALE_PRC': final_preds
})

output.to_csv('predictions.csv', index=False)