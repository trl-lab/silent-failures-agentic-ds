import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/medicines/data/train.parquet'
test_path  = '/path/to/user/workspace/ds_failures/datasets/strable/medicines/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Medicine status'

y_raw = train_df[target_col].astype(str).values
le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

for col in list(df_all.columns):
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        df_all[f'{col}_year'] = df_all[col].dt.year
        df_all[f'{col}_month'] = df_all[col].dt.month
        df_all[f'{col}_day'] = df_all[col].dt.day
        df_all[f'{col}_dayofweek'] = df_all[col].dt.dayofweek
        df_all.drop(columns=[col], inplace=True)
    elif df_all[col].dtype == 'object' or df_all[col].dtype.name == 'category':
        converted = False
        try:
            dt_series = pd.to_datetime(df_all[col], errors='coerce')
            if dt_series.notna().sum() > 0.5 * len(df_all):
                df_all[f'{col}_year'] = dt_series.dt.year
                df_all[f'{col}_month'] = dt_series.dt.month
                df_all[f'{col}_day'] = dt_series.dt.day
                df_all[f'{col}_dayofweek'] = dt_series.dt.dayofweek
                df_all.drop(columns=[col], inplace=True)
                converted = True
        except Exception:
            pass
        
        if not converted:
            str_series = df_all[col].astype(str)
            freq = str_series.value_counts(dropna=False).to_dict()
            df_all[f'{col}_freq'] = str_series.map(freq)
            
            col_le = LabelEncoder()
            df_all[col] = col_le.fit_transform(str_series)

df_all.columns = [f'col_{i}' for i in range(df_all.shape[1])]

X_train_proc = df_all.iloc[:n_train].copy()
X_test_proc = df_all.iloc[n_train:].copy()

for c in X_train_proc.columns:
    X_train_proc[c] = pd.to_numeric(X_train_proc[c], errors='coerce').astype(np.float32)
    X_test_proc[c] = pd.to_numeric(X_test_proc[c], errors='coerce').astype(np.float32)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test_proc))
cb_preds = np.zeros(len(X_test_proc))
xgb_preds = np.zeros(len(X_test_proc))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]
    
    # LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict_proba(X_test_proc)[:, 1] / 5.0

    # CatBoost
    model_cb = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += model_cb.predict_proba(X_test_proc)[:, 1] / 5.0

    # XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict_proba(X_test_proc)[:, 1] / 5.0

final_preds = (lgb_preds + cb_preds + xgb_preds) / 3.0

predictions_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Medicine status': final_preds
})

predictions_df.to_csv('predictions.csv', index=False)