import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/power-plants/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/power-plants/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'NAMPLT_CAP'
y = train_df[target_col].values

train_len = len(train_df)
combined_df = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0, ignore_index=True)

# Feature Engineering
def parse_date(date_str):
    if pd.isna(date_str) or str(date_str).strip() in ['NOT AVAILABLE', 'None', '', 'nan']:
        return np.nan, np.nan
    parts = str(date_str).split('-')
    try:
        year = float(parts[0])
        month = float(parts[1]) if len(parts) > 1 else np.nan
        return year, month
    except Exception:
        return np.nan, np.nan

run_parsed = combined_df['RUN_DATE'].apply(parse_date)
combined_df['RUN_YEAR'] = [x[0] for x in run_parsed]
combined_df['RUN_MONTH'] = [x[1] for x in run_parsed]

ret_parsed = combined_df['RET_DATE'].apply(parse_date)
combined_df['RET_YEAR'] = [x[0] for x in ret_parsed]
combined_df['RET_MONTH'] = [x[1] for x in ret_parsed]
combined_df['IS_RETIRED'] = combined_df['RET_YEAR'].notna().astype(int)

# Frequency encoding for high cardinality categoricals
high_card_cols = ['CITY', 'ZIP', 'PLANT', 'OPERATOR', 'NAME', 'ID']
for col in high_card_cols:
    freq = combined_df[col].astype(str).value_counts()
    combined_df[f'{col}_FREQ'] = combined_df[col].astype(str).map(freq)

# Categorical column encoding
cat_cols = [
    'TYPE', 'NAICS_CODE', 'NAICS_DESC', 'PRIM_FUEL', 'FUEL1_TRAN',
    'SEC_FUEL', 'FUEL2_TRAN', 'PRIM_MVR', 'STATE', 'CITY', 'ZIP',
    'PLANT', 'OPERATOR', 'NAME', 'ID'
]

for col in cat_cols:
    combined_df[col] = combined_df[col].fillna('MISSING').astype(str)

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
combined_df[cat_cols] = encoder.fit_transform(combined_df[cat_cols])

drop_cols = ['RUN_DATE', 'RET_DATE', 'OBJECTID']
features_df = combined_df.drop(columns=[c for c in drop_cols if c in combined_df.columns])

X_train = features_df.iloc[:train_len].copy()
X_test = features_df.iloc[train_len:].copy()

# Cross-validation and ensemble training
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    lgb_model = LGBMRegressor(
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
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test) / n_splits

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test) / n_splits

final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

submission = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'NAMPLT_CAP': final_predictions
})

submission.to_csv('predictions.csv', index=False)