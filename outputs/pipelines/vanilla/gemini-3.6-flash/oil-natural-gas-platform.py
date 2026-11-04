import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/oil-natural-gas-platform/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/oil-natural-gas-platform/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "WATERDEPTH"

train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

n_train = len(X_train)
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

for col in list(combined.columns):
    if combined[col].dtype == 'object' or 'date' in col.lower() or 'time' in col.lower():
        try:
            parsed = pd.to_datetime(combined[col], errors='coerce')
            if parsed.notna().sum() > 0.5 * len(combined):
                combined[col + '_year'] = parsed.dt.year
                combined[col + '_month'] = parsed.dt.month
                combined[col + '_day'] = parsed.dt.day
                combined[col + '_dayofweek'] = parsed.dt.dayofweek
        except Exception:
            pass

cat_cols = combined.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
for c in cat_cols:
    combined[c] = combined[c].fillna('missing').astype(str).astype('category')

X_train_proc = combined.iloc[:n_train].copy()
X_test_proc = combined.iloc[n_train:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_proc))
cat_preds = np.zeros(len(X_test_proc))
xgb_preds = np.zeros(len(X_test_proc))

for train_idx, val_idx in kf.split(X_train_proc, y_train):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    lgb_preds += lgb_model.predict(X_test_proc) / kf.n_splits
    
    cat_model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        random_seed=42,
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        verbose=0
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    cat_preds += cat_model.predict(X_test_proc) / kf.n_splits
    
    xgb_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        enable_categorical=True,
        early_stopping_rounds=50
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_preds += xgb_model.predict(X_test_proc) / kf.n_splits

final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0

if y_train.min() >= 0:
    final_preds = np.clip(final_preds, a_min=0, a_max=None)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'WATERDEPTH': final_preds
})

submission.to_csv('predictions.csv', index=False)