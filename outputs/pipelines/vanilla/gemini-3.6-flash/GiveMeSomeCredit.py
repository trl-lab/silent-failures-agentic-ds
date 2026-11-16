import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/GiveMeSomeCredit/data/train.parquet'
test_path  = '/path/to/user/workspace/ds_failures/datasets/kaggle/GiveMeSomeCredit/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'SeriousDlqin2yrs'

for col in ['Unnamed: 0', 'id', 'Id']:
    if col in train.columns:
        train = train.drop(columns=[col])
    if col in test.columns:
        test = test.drop(columns=[col])

y_train = train[target_col].values
train = train.drop(columns=[target_col])

test_ids = np.arange(len(test))
if target_col in test.columns:
    test = test.drop(columns=[target_col])

def clean_and_engineer(df):
    df = df.copy()
    df.columns = [re.sub(r'[^\w\s]', '_', c) for c in df.columns]
    
    col_inc = 'MonthlyIncome'
    col_dep = 'NumberOfDependents'
    col_rev = 'RevolvingUtilizationOfUnsecuredLines'
    col_age = 'age'
    col_debt = 'DebtRatio'
    col_open = 'NumberOfOpenCreditLinesAndLoans'
    col_re = 'NumberRealEstateLoansOrLines'
    
    late_30_59 = 'NumberOfTime30_59DaysPastDueNotWorse' if 'NumberOfTime30_59DaysPastDueNotWorse' in df.columns else 'NumberOfTime30_59DaysPastDueNotWorse'
    late_60_89 = 'NumberOfTime60_89DaysPastDueNotWorse'
    late_90 = 'NumberOfTimes90DaysLate'

    if col_inc in df.columns:
        df['MonthlyIncome_isna'] = df[col_inc].isna().astype(int)
    if col_dep in df.columns:
        df['NumberOfDependents_isna'] = df[col_dep].isna().astype(int)

    late_cols = [c for c in [late_30_59, late_60_89, late_90] if c in df.columns]
    for c in late_cols:
        df[f'{c}_is_special'] = df[c].isin([96, 98]).astype(int)
        df[f'{c}_clean'] = df[c].replace({96: np.nan, 98: np.nan})

    if len(late_cols) == 3:
        df['TotalLate'] = df[late_30_59].clip(upper=10) + df[late_60_89].clip(upper=10) + df[late_90].clip(upper=10)
        df['TotalLate_clean'] = df[f'{late_30_59}_clean'].fillna(0) + df[f'{late_60_89}_clean'].fillna(0) + df[f'{late_90}_clean'].fillna(0)
        df['WeightedLate_clean'] = (df[f'{late_30_59}_clean'].fillna(0) * 1 +
                                    df[f'{late_60_89}_clean'].fillna(0) * 2 +
                                    df[f'{late_90}_clean'].fillna(0) * 3)

    if col_rev in df.columns:
        rev = df[col_rev]
        df['RevUtil_gt_1'] = (rev > 1.0).astype(int)
        df['RevUtil_gt_10'] = (rev > 10.0).astype(int)
        df['Log_RevUtil'] = np.log1p(np.maximum(0, rev))

    if col_debt in df.columns:
        df['Log_DebtRatio'] = np.log1p(np.maximum(0, df[col_debt]))
        df['DebtRatio_gt_1'] = (df[col_debt] > 1.0).astype(int)

    if col_inc in df.columns:
        inc = df[col_inc].fillna(df[col_inc].median())
        df['Log_MonthlyIncome'] = np.log1p(np.maximum(0, df[col_inc].fillna(0)))

        if col_dep in df.columns:
            deps = df[col_dep].fillna(0)
            df['IncomePerPerson'] = inc / (deps + 1.0)

        if col_open in df.columns:
            lines = df[col_open]
            df['IncomePerLine'] = inc / (lines + 1.0)

        if col_debt in df.columns:
            has_inc = df[col_inc].notna()
            df['EstimatedMonthlyDebt'] = np.where(has_inc, df[col_inc] * df[col_debt], df[col_debt])
            df['DisposableIncome'] = np.where(has_inc, df[col_inc] - df['EstimatedMonthlyDebt'], np.nan)

    if col_open in df.columns and col_re in df.columns:
        df['TotalCreditLines'] = df[col_open] + df[col_re]
        df['RealEstateRatio'] = df[col_re] / (df['TotalCreditLines'] + 1e-5)

    if col_age in df.columns and col_rev in df.columns:
        df['Age_x_RevUtil'] = df[col_age] * df[col_rev]

    return df

X_train = clean_and_engineer(train)
X_test = clean_and_engineer(test)

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'random_state': 42,
    'verbose': -1
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.03,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'n_estimators': 1500
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_val, reference=lgb_train)
    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1500,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / skf.n_splits
    
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        verbose=False
    )
    xgb_preds += model_xgb.predict_proba(X_test)[:, 1] / skf.n_splits
    
    model_cat = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=5,
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_val), use_best_model=True)
    cat_preds += model_cat.predict_proba(X_test)[:, 1] / skf.n_splits

final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cat_preds

sub = pd.DataFrame({
    'id': test_ids,
    'SeriousDlqin2yrs': final_preds
})

sub.to_csv('predictions.csv', index=False)