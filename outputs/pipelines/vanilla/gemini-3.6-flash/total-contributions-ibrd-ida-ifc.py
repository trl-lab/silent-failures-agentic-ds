import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import catboost as cb

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/total-contributions-ibrd-ida-ifc/data/train.parquet'
    test_path = '/path/to/user/workspace/ds_failures/datasets/strable/total-contributions-ibrd-ida-ifc/data/test.parquet'

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = 'Total Contribution (USD)'

    if target_col in train_df.columns:
        train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
        y = train_df[target_col].values
        train_feats = train_df.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found in train data.")

    test_feats = test_df.drop(columns=[target_col], errors='ignore').copy()

    n_train = len(train_feats)
    all_data = pd.concat([train_feats, test_feats], axis=0, ignore_index=True)

    for col in list(all_data.columns):
        if pd.api.types.is_datetime64_any_dtype(all_data[col]):
            all_data[f'{col}_year'] = all_data[col].dt.year
            all_data[f'{col}_month'] = all_data[col].dt.month
            all_data[f'{col}_day'] = all_data[col].dt.day
            all_data[f'{col}_dayofweek'] = all_data[col].dt.dayofweek
            all_data[col] = all_data[col].astype('int64') // 10**9
        elif all_data[col].dtype == 'object':
            try:
                parsed = pd.to_datetime(all_data[col], errors='coerce')
                if parsed.notna().sum() > 0.5 * len(all_data):
                    all_data[f'{col}_year'] = parsed.dt.year
                    all_data[f'{col}_month'] = parsed.dt.month
                    all_data[f'{col}_day'] = parsed.dt.day
                    all_data[f'{col}_dayofweek'] = parsed.dt.dayofweek
            except Exception:
                pass

    cat_cols = []
    for col in all_data.columns:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            all_data[col] = all_data[col].fillna('Missing').astype(str)
            cat_cols.append(col)

    for col in cat_cols:
        freq = all_data[col].value_counts()
        all_data[f'{col}_freq'] = all_data[col].map(freq)

    X_train = all_data.iloc[:n_train].copy()
    X_test = all_data.iloc[n_train:].copy()

    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    lgb_preds_raw = np.zeros(len(X_test))
    cb_preds_raw = np.zeros(len(X_test))

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]

        model_lgb = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_preds_raw += model_lgb.predict(X_test) / n_splits

    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_cb, y)):
        X_tr, y_tr = X_train_cb.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X_train_cb.iloc[val_idx], y[val_idx]

        model_cb = cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.04,
            depth=6,
            cat_features=cat_cols,
            random_seed=42 + fold,
            verbose=0
        )
        model_cb.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=50,
            verbose=False
        )
        cb_preds_raw += model_cb.predict(X_test_cb) / n_splits

    use_log = (y >= 0).all() and (np.max(y) > 10 * np.median(y) if np.median(y) > 0 else True)
    if use_log:
        y_log = np.log1p(y)
        lgb_preds_log = np.zeros(len(X_test))
        cb_preds_log = np.zeros(len(X_test))

        for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_log)):
            X_tr, y_tr = X_train.iloc[trn_idx], y_log[trn_idx]
            X_va, y_va = X_train.iloc[val_idx], y_log[val_idx]

            model_lgb = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=63,
                random_state=142 + fold,
                n_jobs=-1,
                verbose=-1
            )
            model_lgb.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            lgb_preds_log += np.expm1(model_lgb.predict(X_test)) / n_splits

        for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_cb, y_log)):
            X_tr, y_tr = X_train_cb.iloc[trn_idx], y_log[trn_idx]
            X_va, y_va = X_train_cb.iloc[val_idx], y_log[val_idx]

            model_cb = cb.CatBoostRegressor(
                iterations=1000,
                learning_rate=0.04,
                depth=6,
                cat_features=cat_cols,
                random_seed=142 + fold,
                verbose=0
            )
            model_cb.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                early_stopping_rounds=50,
                verbose=False
            )
            cb_preds_log += np.expm1(model_cb.predict(X_test_cb)) / n_splits

        final_preds = 0.35 * lgb_preds_raw + 0.35 * cb_preds_raw + 0.15 * lgb_preds_log + 0.15 * cb_preds_log
    else:
        final_preds = 0.5 * lgb_preds_raw + 0.5 * cb_preds_raw

    if (y >= 0).all():
        final_preds = np.maximum(0, final_preds)

    sub = pd.DataFrame({
        'id': np.arange(len(test_df)),
        'Total Contribution (USD)': final_preds
    })

    sub.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()