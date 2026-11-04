import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

# Load dataset
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/ida-statement-credits-grants-guarantees/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/ida-statement-credits-grants-guarantees/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Original Principal Amount (US$)'
y_train = train_df[target_col].values

date_cols = [
    'End of Period', 'First Repayment Date', 'Last Repayment Date',
    'Agreement Signing Date', 'Board Approval Date',
    'Effective Date (Most Recent)', 'Closed Date (Most Recent)',
    'Last Disbursement Date'
]

def preprocess(df):
    df_feat = pd.DataFrame(index=df.index)
    
    # Process date features
    parsed_dates = {}
    for col in date_cols:
        if col in df.columns:
            d = pd.to_datetime(df[col], errors='coerce')
            parsed_dates[col] = d
            df_feat[col + '_year'] = d.dt.year.astype(float)
            df_feat[col + '_month'] = d.dt.month.astype(float)
            df_feat[col + '_day'] = d.dt.day.astype(float)
            df_feat[col + '_ts'] = d.astype('int64') // 10**9
            df_feat.loc[d.isna(), col + '_ts'] = np.nan
            
    # Compute date durations
    if 'First Repayment Date' in parsed_dates and 'Last Repayment Date' in parsed_dates:
        df_feat['repay_duration_days'] = (parsed_dates['Last Repayment Date'] - parsed_dates['First Repayment Date']).dt.days.astype(float)
    if 'Agreement Signing Date' in parsed_dates and 'Board Approval Date' in parsed_dates:
        df_feat['approval_to_signing_days'] = (parsed_dates['Agreement Signing Date'] - parsed_dates['Board Approval Date']).dt.days.astype(float)
    if 'Effective Date (Most Recent)' in parsed_dates and 'Agreement Signing Date' in parsed_dates:
        df_feat['signing_to_effective_days'] = (parsed_dates['Effective Date (Most Recent)'] - parsed_dates['Agreement Signing Date']).dt.days.astype(float)
    if 'Closed Date (Most Recent)' in parsed_dates and 'Effective Date (Most Recent)' in parsed_dates:
        df_feat['effective_to_closed_days'] = (parsed_dates['Closed Date (Most Recent)'] - parsed_dates['Effective Date (Most Recent)']).dt.days.astype(float)
        
    # Numerical features
    if 'Service Charge Rate' in df.columns:
        df_feat['Service Charge Rate'] = df['Service Charge Rate'].astype(float)
        
    # Categorical features
    cat_cols = ['Region', 'Country / Economy Code', 'Country / Economy', 'Borrower', 'Credit Status', 'Currency of Commitment', 'Credit Number', 'Project ID', 'Project Name']
    for col in cat_cols:
        if col in df.columns:
            df_feat[col] = df[col].astype('category')
            
    return df_feat

X_train = preprocess(train_df)
X_test = preprocess(test_df)

# Align categorical levels across train and test
cat_cols = [c for c in X_train.columns if X_train[c].dtype.name == 'category']
for col in cat_cols:
    combined_cats = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0).astype('category').cat.categories
    X_train[col] = pd.Categorical(X_train[col].astype(str), categories=combined_cats)
    X_test[col] = pd.Categorical(X_test[col].astype(str), categories=combined_cats)

# Cross-validated training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    
    test_preds += model.predict(X_test) / kf.n_splits

if np.nanmin(y_train) >= 0:
    test_preds = np.clip(test_preds, a_min=0, a_max=None)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_preds)),
    'Original Principal Amount (US$)': test_preds
})
output_df.to_csv('predictions.csv', index=False)