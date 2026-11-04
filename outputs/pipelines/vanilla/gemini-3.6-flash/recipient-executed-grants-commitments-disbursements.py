import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import re
import warnings

warnings.filterwarnings('ignore')

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/recipient-executed-grants-commitments-disbursements/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/recipient-executed-grants-commitments-disbursements/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "Disbursements USD (Trust funds)"

y_train = train[target_col].values
X_train = train.drop(columns=[target_col])
X_test = test.drop(columns=[target_col], errors='ignore')

n_train = len(X_train)
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

def clean_col_name(col):
    return re.sub(r'[\[\]\{\}:,"]', '_', str(col))

combined.columns = [clean_col_name(c) for c in combined.columns]

cat_cols = []
num_cols = []

for col in list(combined.columns):
    if pd.api.types.is_datetime64_any_dtype(combined[col]):
        combined[f"{col}_year"] = combined[col].dt.year
        combined[f"{col}_month"] = combined[col].dt.month
        combined[f"{col}_day"] = combined[col].dt.day
        combined[f"{col}_dayofweek"] = combined[col].dt.dayofweek
        combined.drop(columns=[col], inplace=True)
    elif combined[col].dtype == 'object':
        sample = combined[col].dropna().head(100)
        is_date = False
        if len(sample) > 0:
            try:
                dt_sample = pd.to_datetime(sample, errors='coerce')
                if dt_sample.notna().mean() > 0.8:
                    dt_col = pd.to_datetime(combined[col], errors='coerce')
                    combined[f"{col}_year"] = dt_col.dt.year
                    combined[f"{col}_month"] = dt_col.dt.month
                    combined[f"{col}_day"] = dt_col.dt.day
                    combined[f"{col}_dayofweek"] = dt_col.dt.dayofweek
                    combined.drop(columns=[col], inplace=True)
                    is_date = True
            except Exception:
                pass
        if not is_date:
            combined[col] = combined[col].astype(str)
            cat_cols.append(col)
    elif pd.api.types.is_numeric_dtype(combined[col]):
        num_cols.append(col)
    else:
        combined[col] = combined[col].astype(str)
        cat_cols.append(col)

group_candidates = []
for col in cat_cols:
    freq = combined[col].value_counts(normalize=True)
    combined[f"{col}_freq"] = combined[col].map(freq).fillna(0)
    
    nunique = combined[col].nunique()
    if 3 <= nunique <= 1000:
        group_candidates.append(col)
        
    combined[col] = combined[col].astype('category').cat.codes

num_cols_proc = [c for c in num_cols if c in combined.columns]
for g_col in group_candidates[:5]:
    for n_col in num_cols_proc[:5]:
        try:
            combined[f"{g_col}_{n_col}_grp_mean"] = combined.groupby(g_col)[n_col].transform('mean')
            combined[f"{g_col}_{n_col}_grp_std"] = combined.groupby(g_col)[n_col].transform('std').fillna(0)
        except Exception:
            pass

X_train_proc = combined.iloc[:n_train].copy()
X_test_proc = combined.iloc[n_train:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(X_test_proc))
cb_preds = np.zeros(len(X_test_proc))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc, y_train)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        random_state=42 + fold,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        n_jobs=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test_proc) / kf.n_splits
    
    model_cb = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        random_seed=42 + fold,
        verbose=0
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    cb_preds += model_cb.predict(X_test_proc) / kf.n_splits

final_preds = 0.5 * lgb_preds + 0.5 * cb_preds

if y_train.min() >= 0:
    final_preds = np.clip(final_preds, 0, None)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    target_col: final_preds
})

submission.to_csv('predictions.csv', index=False)