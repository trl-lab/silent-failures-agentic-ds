import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/us-school-bus-fleet/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/us-school-bus-fleet/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Bus age'
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

drop_cols = []
for col in X.columns:
    if col.lower() in ['id', 'index', 'unnamed: 0']:
        drop_cols.append(col)

if drop_cols:
    X = X.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

cat_cols = []
num_cols = []
date_cols = []

for col in X.columns:
    if pd.api.types.is_bool_dtype(X[col]):
        X[col] = X[col].astype(int)
        X_test[col] = X_test[col].astype(int)
        num_cols.append(col)
    elif pd.api.types.is_datetime64_any_dtype(X[col]):
        date_cols.append(col)
    elif X[col].dtype == 'object' or X[col].dtype.name == 'category':
        try:
            parsed = pd.to_datetime(X[col], errors='coerce')
            if parsed.notna().sum() > 0.5 * len(X):
                X[col] = parsed
                X_test[col] = pd.to_datetime(X_test[col], errors='coerce')
                date_cols.append(col)
                continue
        except Exception:
            pass
        cat_cols.append(col)
    else:
        num_cols.append(col)

for col in date_cols:
    for df in [X, X_test]:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
    X = X.drop(columns=[col])
    X_test = X_test.drop(columns=[col])

for col in cat_cols:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

X_encoded = X.copy()
X_test_encoded = X_test.copy()

for col in cat_cols:
    categories = list(set(X[col].cat.categories).union(set(X_test[col].cat.categories)))
    cat_type = pd.CategoricalDtype(categories=categories)
    X_encoded[col] = X[col].astype(cat_type).cat.codes
    X_test_encoded[col] = X_test[col].astype(cat_type).cat.codes

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / n_splits
    
    cat_cols_idx = [X.columns.get_loc(c) for c in cat_cols]
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    X_test_cat = X_test.copy()
    for c in cat_cols:
        X_train_cat[c] = X_train_cat[c].astype(str)
        X_val_cat[c] = X_val_cat[c].astype(str)
        X_test_cat[c] = X_test_cat[c].astype(str)
        
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        eval_metric='RMSE',
        random_seed=42 + fold,
        cat_features=cat_cols_idx,
        verbose=0
    )
    cat_model.fit(
        X_train_cat, y_train,
        eval_set=(X_val_cat, y_val),
        early_stopping_rounds=50
    )
    cat_preds += cat_model.predict(X_test_cat) / n_splits

    X_train_xgb, y_train_xgb = X_encoded.iloc[train_idx], y[train_idx]
    X_val_xgb, y_val_xgb = X_encoded.iloc[val_idx], y[val_idx]
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_train_xgb, y_train_xgb,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test_encoded) / n_splits

final_preds = 0.4 * lgb_preds + 0.4 * cat_preds + 0.2 * xgb_preds

if y.min() >= 0:
    final_preds = np.clip(final_preds, a_min=0, a_max=None)

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Bus age': final_preds
})

sub.to_csv('predictions.csv', index=False)