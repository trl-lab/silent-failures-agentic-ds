import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/default-of-credit-card-clients/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/default-of-credit-card-clients/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

rename_dict = {
    'X1': 'LIMIT_BAL', 'X2': 'SEX', 'X3': 'EDUCATION', 'X4': 'MARRIAGE', 'X5': 'AGE',
    'X6': 'PAY_0', 'X7': 'PAY_2', 'X8': 'PAY_3', 'X9': 'PAY_4', 'X10': 'PAY_5', 'X11': 'PAY_6',
    'X12': 'BILL_AMT1', 'X13': 'BILL_AMT2', 'X14': 'BILL_AMT3', 'X15': 'BILL_AMT4', 'X16': 'BILL_AMT5', 'X17': 'BILL_AMT6',
    'X18': 'PAY_AMT1', 'X19': 'PAY_AMT2', 'X20': 'PAY_AMT3', 'X21': 'PAY_AMT4', 'X22': 'PAY_AMT5', 'X23': 'PAY_AMT6',
    'Y': 'default payment next month'
}

train_df = train_df.rename(columns=rename_dict)
test_df = test_df.rename(columns=rename_dict)

target_col = "default payment next month"
if target_col not in train_df.columns:
    for c in train_df.columns:
        if "default" in c.lower() or c.lower() == "y":
            target_col = c
            break

def engineer_features(df):
    df = df.copy()
    
    pay_cols = [c for c in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'] if c in df]
    bill_cols = [c for c in ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'] if c in df]
    pay_amt_cols = [c for c in ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'] if c in df]

    if 'LIMIT_BAL' in df:
        if bill_cols:
            df['BILL_AMT_MEAN'] = df[bill_cols].mean(axis=1)
            df['BILL_AMT_STD'] = df[bill_cols].std(axis=1)
            df['BILL_AMT_MAX'] = df[bill_cols].max(axis=1)
            df['BILL_AMT_MIN'] = df[bill_cols].min(axis=1)
            if 'BILL_AMT1' in df:
                df['BILL_TO_LIMIT_RATIO'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1)
            df['MEAN_BILL_TO_LIMIT_RATIO'] = df['BILL_AMT_MEAN'] / (df['LIMIT_BAL'] + 1)

        if pay_amt_cols:
            df['PAY_AMT_MEAN'] = df[pay_amt_cols].mean(axis=1)
            df['PAY_AMT_STD'] = df[pay_amt_cols].std(axis=1)
            df['PAY_AMT_SUM'] = df[pay_amt_cols].sum(axis=1)
            df['PAY_TO_LIMIT_RATIO'] = df['PAY_AMT_SUM'] / (df['LIMIT_BAL'] + 1)

    if pay_cols:
        df['PAY_MAX'] = df[pay_cols].max(axis=1)
        df['PAY_MEAN'] = df[pay_cols].mean(axis=1)
        df['PAY_SUM'] = df[pay_cols].sum(axis=1)
        df['PAY_DELAY_COUNT'] = (df[pay_cols] > 0).sum(axis=1)

    if len(bill_cols) == 6 and len(pay_amt_cols) == 6:
        for i in range(1, 6):
            b_col = f'BILL_AMT{i+1}'
            p_col = f'PAY_AMT{i}'
            if f'BILL_AMT{i}' in df and b_col in df and p_col in df:
                df[f'UTIL_DIFF_{i}'] = df[f'BILL_AMT{i}'] - df[b_col]
                df[f'PAY_VS_BILL_{i}'] = df[p_col] / (np.abs(df[b_col]) + 1)

    return df

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

ignore_cols = [target_col, 'ID', 'id', 'Id', 'X0', 'Unnamed: 0', 'index', 'row_id']
feature_cols = [c for c in train_df.columns if c not in ignore_cols]

X = train_df[feature_cols].copy()
y = train_df[target_col].values
X_test = test_df[feature_cols].copy()

cat_cols = [c for c in ['SEX', 'EDUCATION', 'MARRIAGE'] if c in X.columns]
for c in cat_cols:
    X[c] = X[c].astype('category')
    X_test[c] = X_test[c].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    cb_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        eval_metric='AUC',
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    xgb_model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=42 + fold,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    p_cb = cb_model.predict_proba(X_test)[:, 1]
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]

    fold_pred = (p_lgb + p_cb + p_xgb) / 3.0
    test_preds += fold_pred / skf.n_splits

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'default payment next month': test_preds
})
sub.to_csv('predictions.csv', index=False)