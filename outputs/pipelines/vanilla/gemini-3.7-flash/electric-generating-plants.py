import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/electric-generating-plants/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/electric-generating-plants/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'NET_GEN'
drop_cols = ['OBJECTID', target_col]

y = train_df[target_col].values
X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
X_test = test_df.drop(columns=[c for c in ['OBJECTID'] if c in test_df.columns])

# Preprocessing
def preprocess_features(df_train, df_test):
    train_feat = df_train.copy()
    test_feat = df_test.copy()
    
    # Process datetime columns
    datetime_cols = [c for c in train_feat.columns if pd.api.types.is_datetime64_any_dtype(train_feat[c])]
    for col in datetime_cols:
        for df in [train_feat, test_feat]:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        train_feat.drop(columns=[col], inplace=True)
        test_feat.drop(columns=[col], inplace=True)

    # Process categorical / object columns
    cat_cols = [c for c in train_feat.columns if train_feat[c].dtype == 'object' or isinstance(train_feat[c].dtype, pd.CategoricalDtype)]
    
    for col in cat_cols:
        train_feat[col] = train_feat[col].fillna('MISSING').astype(str)
        test_feat[col] = test_feat[col].fillna('MISSING').astype(str)
        
        # Convert to categorical for LightGBM
        train_feat[col] = train_feat[col].astype('category')
        test_feat[col] = pd.Categorical(test_feat[col], categories=train_feat[col].cat.categories)
        
    return train_feat, test_feat, cat_cols

X, X_test, cat_cols = preprocess_features(X, X_test)

# K-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))

cat_oof = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))

# Prepare data for CatBoost (convert category to string and handle NaNs)
X_catboost = X.copy()
X_test_catboost = X_test.copy()
for col in cat_cols:
    X_catboost[col] = X_catboost[col].astype(str).fillna('MISSING')
    X_test_catboost[col] = X_test_catboost[col].astype(str).fillna('MISSING')

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    # LightGBM Model
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
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
    
    # CatBoost Model
    X_tr_cat, X_va_cat = X_catboost.iloc[train_idx], X_catboost.iloc[val_idx]
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr_cat, y_tr,
        eval_set=(X_va_cat, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    
    cat_oof[val_idx] = cb_model.predict(X_va_cat)
    cat_preds += cb_model.predict(X_test_catboost) / kf.n_splits

# Ensemble Predictions
final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions to CSV
submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'NET_GEN': final_preds
})
submission.to_csv('predictions.csv', index=False)