import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/health-professional-shortage-areas/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/health-professional-shortage-areas/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'HPSA Score'
train_clean = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train_clean[target_col].values
X_train = train_clean.drop(columns=[target_col])
X_test = test_df.copy()

def parse_ratio(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if ':' in val_str:
        parts = val_str.split(':')
        try:
            num = float(parts[0])
            denom = float(parts[1])
            return num / denom if denom != 0 else np.nan
        except (ValueError, ZeroDivisionError):
            return np.nan
    try:
        return float(val_str)
    except ValueError:
        return np.nan

combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

ratio_cols = ['HPSA Formal Ratio', 'HPSA Provider Ratio Goal']
for col in ratio_cols:
    if col in combined.columns:
        combined[f'{col}_numeric'] = combined[col].apply(parse_ratio)
        combined.drop(columns=[col], inplace=True)

date_cols = ['HPSA Designation Date', 'HPSA Designation Last Update Date']
d1 = pd.to_datetime(combined['HPSA Designation Date'], errors='coerce') if 'HPSA Designation Date' in combined.columns else None
d2 = pd.to_datetime(combined['HPSA Designation Last Update Date'], errors='coerce') if 'HPSA Designation Last Update Date' in combined.columns else None

if d1 is not None and d2 is not None:
    combined['date_diff_days'] = (d2 - d1).dt.days

for col in date_cols:
    if col in combined.columns:
        dt = pd.to_datetime(combined[col], errors='coerce')
        combined[f'{col}_year'] = dt.dt.year
        combined[f'{col}_month'] = dt.dt.month
        combined[f'{col}_day'] = dt.dt.day
        combined[f'{col}_dayofweek'] = dt.dt.dayofweek
        combined[f'{col}_epoch_days'] = (dt - pd.Timestamp("1970-01-01")).dt.days
        combined.drop(columns=[col], inplace=True)

categorical_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    freq = combined[col].value_counts()
    combined[f'{col}_freq'] = combined[col].map(freq)
    combined[col] = pd.factorize(combined[col].astype(str))[0]
    combined.loc[combined[col] == -1, col] = np.nan

n_train = len(X_train)
X_train_processed = combined.iloc[:n_train].reset_index(drop=True)
X_test_processed = combined.iloc[n_train:].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_test_preds = np.zeros(len(X_test_processed))
cat_test_preds = np.zeros(len(X_test_processed))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_processed, y_train)):
    X_tr, y_tr = X_train_processed.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_processed.iloc[val_idx], y_train[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
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
    lgb_test_preds += lgb_model.predict(X_test_processed) / kf.n_splits
    
    cat_model = CatBoostRegressor(
        iterations=1200,
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
    cat_test_preds += cat_model.predict(X_test_processed) / kf.n_splits

final_preds = 0.5 * lgb_test_preds + 0.5 * cat_test_preds

submission = pd.DataFrame({
    'id': range(len(final_preds)),
    'HPSA Score': final_preds
})

submission.to_csv('predictions.csv', index=False)