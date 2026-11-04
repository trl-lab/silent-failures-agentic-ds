import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/historic-perimeters-wildfires/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/historic-perimeters-wildfires/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'gisacres'
y_train = train_df[target_col].values

# Feature engineering function
def extract_features(df):
    data = pd.DataFrame(index=df.index)
    
    # Numerical features
    if 'OBJECTID' in df.columns:
        data['OBJECTID'] = pd.to_numeric(df['OBJECTID'], errors='coerce')
    if 'fireyear' in df.columns:
        data['fireyear'] = pd.to_numeric(df['fireyear'], errors='coerce')

    # Date parsing safely
    for date_col in ['datecurrent', 'perimeterdatetime']:
        if date_col in df.columns:
            # Convert to string and parse basic components
            dt = pd.to_datetime(df[date_col], errors='coerce')
            data[f'{date_col}_year'] = dt.dt.year.fillna(-1).astype('int32')
            data[f'{date_col}_month'] = dt.dt.month.fillna(-1).astype('int32')
            data[f'{date_col}_day'] = dt.dt.day.fillna(-1).astype('int32')
            data[f'{date_col}_dayofweek'] = dt.dt.dayofweek.fillna(-1).astype('int32')
            data[f'{date_col}_dayofyear'] = dt.dt.dayofyear.fillna(-1).astype('int32')
            data[f'{date_col}_hour'] = dt.dt.hour.fillna(-1).astype('int32')

    # Categorical / string features
    cat_cols = [
        'agency', 'mapmethod', 'state', 'incomplex', 'pooownerunit',
        'pooresponsibleunit', 'complexname', 'firecode', 'complexfirecode'
    ]
    
    for col in cat_cols:
        if col in df.columns:
            data[col] = df[col].astype(str).fillna('missing').astype('category')
            data[f'{col}_len'] = df[col].astype(str).str.len()

    # Presence indicators for sparse ID columns
    id_cols = ['comments', 'uniquefireidentifier', 'complexparentirwinid', 'inciwebid', 'localincidentidentifier', 'irwinid', 'incidentname']
    for col in id_cols:
        if col in df.columns:
            data[f'{col}_is_null'] = df[col].isna().astype(int)
            data[f'{col}_len'] = df[col].astype(str).str.len()

    return data

X_train = extract_features(train_df)
X_test = extract_features(test_df)

# Align categories
categorical_features = [c for c in X_train.columns if X_train[c].dtype.name == 'category']

for col in categorical_features:
    train_cats = set(X_train[col].dropna().unique())
    test_cats = set(X_test[col].dropna().unique())
    all_cats = list(train_cats.union(test_cats))
    X_train[col] = pd.Categorical(X_train[col], categories=all_cats)
    X_test[col] = pd.Categorical(X_test[col], categories=all_cats)

# Transform target to log1p
y_train_clean = np.nan_to_num(y_train, nan=0.0)
y_train_log = np.log1p(np.maximum(0, y_train_clean))

# K-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

preds_log = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_log)):
    X_tr, y_tr_log = X_train.iloc[train_idx], y_train_log[train_idx]
    X_va, y_va_log = X_train.iloc[val_idx], y_train_log[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    preds_log += model.predict(X_test) / n_splits

final_predictions = np.maximum(0, np.expm1(preds_log))

# Save submission
submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'gisacres': final_predictions
})
submission.to_csv('predictions.csv', index=False)