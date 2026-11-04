import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/medically-underserved-areas-populations/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/medically-underserved-areas-populations/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

train_df = train_df.dropna(subset=['IMU Score']).reset_index(drop=True)
y = train_df['IMU Score'].values

X_train = train_df.drop(columns=['IMU Score'])
X_test = test_df.copy()

combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

datetime_cols = combined.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
for col in datetime_cols:
    combined[f'{col}_year'] = combined[col].dt.year
    combined[f'{col}_month'] = combined[col].dt.month
    combined[f'{col}_day'] = combined[col].dt.day
    combined[f'{col}_dayofweek'] = combined[col].dt.dayofweek
    combined[f'{col}_timestamp'] = combined[col].astype('int64') // 10**9
    combined.loc[combined[col].isna(), f'{col}_timestamp'] = np.nan
    combined.drop(columns=[col], inplace=True)

cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    combined[col] = combined[col].astype(str).fillna('__MISSING__')

if cat_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    combined[cat_cols] = encoder.fit_transform(combined[cat_cols])

n_train = len(X_train)
X = combined.iloc[:n_train].reset_index(drop=True)
X_sub = combined.iloc[n_train:].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_sub))
cat_preds = np.zeros(len(X_sub))

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_sub) / kf.n_splits
    
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_sub) / kf.n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'IMU Score': final_preds
})

output_df.to_csv('predictions.csv', index=False)