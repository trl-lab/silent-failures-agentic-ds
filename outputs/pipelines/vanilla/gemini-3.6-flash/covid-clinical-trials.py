import os
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/covid-clinical-trials/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/covid-clinical-trials/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'days_to_complete'

# Drop target NaNs in train if any
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train_df[target_col].values.astype(np.float32)

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

n_train = len(X_train_raw)
n_test = len(X_test_raw)

combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

processed_features = []

for col in combined.columns:
    col_data = combined[col]

    if pd.api.types.is_datetime64_any_dtype(col_data):
        dt = col_data
        df_dt = pd.DataFrame({
            f'{col}_year': dt.dt.year,
            f'{col}_month': dt.dt.month,
            f'{col}_day': dt.dt.day,
            f'{col}_dayofweek': dt.dt.dayofweek,
            f'{col}_days_ref': (dt - pd.Timestamp('2020-01-01')).dt.total_seconds() / 86400.0
        })
        processed_features.append(df_dt)
        continue

    if pd.api.types.is_numeric_dtype(col_data):
        processed_features.append(pd.DataFrame({col: col_data.astype(np.float32)}))
        continue

    non_nulls = col_data.dropna().astype(str)
    if len(non_nulls) > 0:
        sample = non_nulls.head(100)
        is_date = False
        try:
            parsed_sample = pd.to_datetime(sample, errors='coerce')
            if parsed_sample.notna().mean() > 0.7:
                is_date = True
        except Exception:
            is_date = False

        if is_date:
            dt = pd.to_datetime(col_data, errors='coerce')
            df_dt = pd.DataFrame({
                f'{col}_year': dt.dt.year,
                f'{col}_month': dt.dt.month,
                f'{col}_day': dt.dt.day,
                f'{col}_dayofweek': dt.dt.dayofweek,
                f'{col}_days_ref': (dt - pd.Timestamp('2020-01-01')).dt.total_seconds() / 86400.0
            })
            processed_features.append(df_dt)
            continue

        avg_len = non_nulls.str.len().mean()
        nunique = non_nulls.nunique()

        if avg_len > 30 or nunique > 200:
            str_series = col_data.fillna('').astype(str)
            try:
                tfidf = TfidfVectorizer(max_features=5000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
                tfidf_mat = tfidf.fit_transform(str_series)
                n_comp = min(10, tfidf_mat.shape[1] - 1)
                if n_comp >= 1:
                    svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    svd_mat = svd.fit_transform(tfidf_mat)
                    svd_df = pd.DataFrame(svd_mat, columns=[f'{col}_svd_{i}' for i in range(n_comp)])
                    svd_df[f'{col}_char_len'] = str_series.str.len().astype(np.float32)
                    svd_df[f'{col}_word_count'] = str_series.str.split().str.len().astype(np.float32)
                    processed_features.append(svd_df)
                    continue
            except Exception:
                pass

        cat_series = col_data.astype(str).fillna('missing')
        freq = cat_series.map(cat_series.value_counts(normalize=True)).astype(np.float32)
        codes, _ = pd.factorize(cat_series)
        processed_features.append(pd.DataFrame({
            f'{col}_code': codes.astype(np.int32),
            f'{col}_freq': freq
        }))

X_all = pd.concat(processed_features, axis=1)
clean_cols = [f"f_{i}_{re.sub(r'[^\w]', '_', str(col))}" for i, col in enumerate(X_all.columns)]
X_all.columns = clean_cols

X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

test_lgb = np.zeros(n_test, dtype=np.float64)
test_cb = np.zeros(n_test, dtype=np.float64)
test_xgb = np.zeros(n_test, dtype=np.float64)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model_lgb = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    test_lgb += model_lgb.predict(X_test) / kf.n_splits

    model_cb = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        verbose=0
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    test_cb += model_cb.predict(X_test) / kf.n_splits

    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=50,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method='hist'
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    test_xgb += model_xgb.predict(X_test) / kf.n_splits

test_preds = (test_lgb + test_cb + test_xgb) / 3.0
min_target = max(0.0, float(np.min(y_train)))
test_preds = np.clip(test_preds, a_min=min_target, a_max=None)

output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'days_to_complete': test_preds
})
output_df.to_csv('predictions.csv', index=False)