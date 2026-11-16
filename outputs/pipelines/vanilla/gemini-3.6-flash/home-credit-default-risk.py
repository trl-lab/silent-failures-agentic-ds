import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/home-credit-default-risk/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/home-credit-default-risk/data/test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target_col = 'TARGET'
    y = train[target_col].values

    features = [c for c in train.columns if c != target_col]
    id_cols = [c for c in features if 'id' in c.lower()]
    features = [c for c in features if c not in id_cols]

    X = train[features].copy()
    X_test = test[features].copy()

    def add_domain_features(df):
        df = df.copy()
        if 'DAYS_BIRTH' in df.columns:
            df['DAYS_BIRTH_AGE'] = -df['DAYS_BIRTH'] / 365.25
        if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
            df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1e-5)
        if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
            df['INCOME_TO_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + 1e-5)
        if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
            df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
        if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
            df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
        if 'DAYS_EMPLOYED' in df.columns:
            df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
            df['DAYS_EMPLOYED_REPLACED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
            if 'DAYS_BIRTH' in df.columns:
                df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED_REPLACED'] / (df['DAYS_BIRTH'] - 1e-5)
        ext_cols = [c for c in df.columns if 'EXT_SOURCE' in c]
        if len(ext_cols) > 0:
            df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
            df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
            df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
            df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
        return df

    X = add_domain_features(X)
    X_test = add_domain_features(X_test)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            categories = pd.concat([X[col], X_test[col]]).astype(str).unique()
            X[col] = pd.Categorical(X[col].astype(str), categories=categories)
            X_test[col] = pd.Categorical(X_test[col].astype(str), categories=categories)
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    test_preds = np.zeros(len(X_test))

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

        test_preds += model.predict(X_test) / n_splits

    sub = pd.DataFrame({
        'id': np.arange(len(test_preds)),
        'TARGET': test_preds
    })
    sub.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()