import pandas as pd
import numpy as np
import lightgbm as lgb

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/tax-incentives/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/tax-incentives/data/test.parquet'

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = 'credit_amount'
    feature_cols = [col for col in train_df.columns if col != target_col]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col]
    X_test = test_df[feature_cols].copy()

    model = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    output_df = pd.DataFrame({
        'id': range(len(y_pred)),
        'credit_amount': y_pred
    })

    output_df.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()