import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/ibrd-statement-loans-guarantees/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/ibrd-statement-loans-guarantees/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Original Principal Amount (US$)'
y = train_df[target_col].values

date_cols = [
    'First Repayment Date', 'Last Repayment Date', 'Agreement Signing Date',
    'Board Approval Date', 'Effective Date (Most Recent)',
    'Closed Date (Most Recent)', 'Last Disbursement Date'
]

cat_cols = [
    'Region', 'Country / Economy Code', 'Country / Economy', 'Borrower',
    'Guarantor Country / Economy Code', 'Guarantor', 'Loan Type',
    'Loan Status', 'Project ID', 'Project Name', 'Loan Number'
]

def preprocess(df, combined_df=None):
    data = pd.DataFrame(index=df.index)
    
    # Numeric features
    data['Interest Rate'] = pd.to_numeric(df['Interest Rate'], errors='coerce')
    
    # Process dates
    dt_series = {}
    for col in date_cols:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors='coerce')
            dt_series[col] = s
            data[f'{col}_year'] = s.dt.year
            data[f'{col}_month'] = s.dt.month
            data[f'{col}_day'] = s.dt.day
            data[f'{col}_dayofweek'] = s.dt.dayofweek
            data[f'{col}_timestamp'] = s.astype('int64') // 10**9
            data[f'{col}_timestamp'] = data[f'{col}_timestamp'].replace(-9223372037, np.nan)

    # Date differences (durations in days)
    if 'Last Repayment Date' in dt_series and 'First Repayment Date' in dt_series:
        data['repayment_duration_days'] = (dt_series['Last Repayment Date'] - dt_series['First Repayment Date']).dt.days
    if 'Agreement Signing Date' in dt_series and 'Board Approval Date' in dt_series:
        data['approval_to_signing_days'] = (dt_series['Agreement Signing Date'] - dt_series['Board Approval Date']).dt.days
    if 'Effective Date (Most Recent)' in dt_series and 'Agreement Signing Date' in dt_series:
        data['signing_to_effective_days'] = (dt_series['Effective Date (Most Recent)'] - dt_series['Agreement Signing Date']).dt.days
    if 'Closed Date (Most Recent)' in dt_series and 'Effective Date (Most Recent)' in dt_series:
        data['effective_to_closed_days'] = (dt_series['Closed Date (Most Recent)'] - dt_series['Effective Date (Most Recent)']).dt.days
    if 'Last Disbursement Date' in dt_series and 'Effective Date (Most Recent)' in dt_series:
        data['effective_to_disbursement_days'] = (dt_series['Last Disbursement Date'] - dt_series['Effective Date (Most Recent)']).dt.days

    # Categorical string features
    for col in cat_cols:
        if col in df.columns:
            s = df[col].astype(str).fillna('missing')
            data[col] = s
            # Frequency encoding
            if combined_df is not None:
                freq = combined_df[col].astype(str).value_counts()
                data[f'{col}_freq'] = s.map(freq).fillna(0)
                
    return data

full_combined = pd.concat([train_df.drop(columns=[target_col], errors='ignore'), test_df], axis=0, ignore_index=True)
X_train = preprocess(train_df, full_combined)
X_test = preprocess(test_df, full_combined)

for col in cat_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

# Modeling with KFold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

# Prepare for CatBoost
X_train_cb = X_train.copy()
X_test_cb = X_test.copy()
for col in cat_cols:
    if col in X_train_cb.columns:
        X_train_cb[col] = X_train_cb[col].astype(str).fillna('missing')
        X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')

cat_indices = [X_train_cb.columns.get_loc(col) for col in cat_cols if col in X_train_cb.columns]

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1500
}

for train_idx, val_idx in kf.split(X_train, y):
    # LightGBM
    X_tr_lgb, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va_lgb, y_va = X_train.iloc[val_idx], y[val_idx]
    
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits
    
    # CatBoost
    X_tr_cb = X_train_cb.iloc[train_idx]
    X_va_cb = X_train_cb.iloc[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=False,
        cat_features=cat_indices
    )
    cb_model.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / kf.n_splits

final_predictions = 0.5 * lgb_preds + 0.5 * cb_preds

output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Original Principal Amount (US$)': final_predictions
})
output_df.to_csv('predictions.csv', index=False)