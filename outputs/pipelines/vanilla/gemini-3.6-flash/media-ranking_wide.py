import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/media-ranking_wide/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/media-ranking_wide/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

train_df = train_df.dropna(subset=['Overall']).reset_index(drop=True)

y = train_df['Overall'].values
X_train = train_df.drop(columns=['Overall'])
X_test = test_df.copy()

if 'Overall' in X_test.columns:
    X_test = X_test.drop(columns=['Overall'])

for col in list(X_train.columns):
    if pd.api.types.is_datetime64_any_dtype(X_train[col]):
        X_train[f'{col}_year'] = X_train[col].dt.year
        X_train[f'{col}_month'] = X_train[col].dt.month
        X_train[f'{col}_day'] = X_train[col].dt.day
        X_train[f'{col}_dayofweek'] = X_train[col].dt.dayofweek
        X_test[f'{col}_year'] = X_test[col].dt.year
        X_test[f'{col}_month'] = X_test[col].dt.month
        X_test[f'{col}_day'] = X_test[col].dt.day
        X_test[f'{col}_dayofweek'] = X_test[col].dt.dayofweek
        X_train.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)
    elif X_train[col].dtype == 'object' or isinstance(X_train[col].dtype, pd.CategoricalDtype) or X_train[col].dtype == 'string':
        s_tr = X_train[col].astype(str)
        s_te = X_test[col].astype(str)
        s_tr[X_train[col].isna()] = np.nan
        s_te[X_test[col].isna()] = np.nan
        
        categories = sorted(list(set(s_tr.dropna().unique()).union(set(s_te.dropna().unique()))))
        cat_type = pd.CategoricalDtype(categories=categories)
        X_train[col] = s_tr.astype(cat_type).cat.codes.astype(float)
        X_train.loc[X_train[col] == -1, col] = np.nan
        X_test[col] = s_te.astype(cat_type).cat.codes.astype(float)
        X_test.loc[X_test[col] == -1, col] = np.nan
    else:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

clean_cols = [re.sub(r'[^\w\s]', '_', str(c)) for c in X_train.columns]
X_train.columns = clean_cols
X_test.columns = clean_cols

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
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
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    cat_preds += cat_model.predict(X_test) / 5.0

    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test) / 5.0

final_preds = 0.4 * lgb_preds + 0.3 * cat_preds + 0.3 * xgb_preds

submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Overall': final_preds
})
submission.to_csv('predictions.csv', index=False)