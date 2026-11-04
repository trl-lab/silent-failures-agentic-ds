import re
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/national-average-drug-acquisition-cost/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/national-average-drug-acquisition-cost/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'NADAC Per Unit'
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col], errors='ignore')

n_train = len(X_train)
combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

features = pd.DataFrame(index=combined.index)

cat_cols = []
for col in combined.columns:
    col_data = combined[col]
    
    if pd.api.types.is_datetime64_any_dtype(col_data) or 'date' in col.lower():
        try:
            dt = pd.to_datetime(col_data, errors='coerce')
            if dt.notna().sum() > 0:
                features[f'{col}_year'] = dt.dt.year.fillna(-1).astype(np.float32)
                features[f'{col}_month'] = dt.dt.month.fillna(-1).astype(np.float32)
                features[f'{col}_day'] = dt.dt.day.fillna(-1).astype(np.float32)
                features[f'{col}_dayofweek'] = dt.dt.dayofweek.fillna(-1).astype(np.float32)
                ref_date = pd.Timestamp('2000-01-01')
                features[f'{col}_days'] = (dt - ref_date).dt.days.fillna(-1).astype(np.float32)
                continue
        except Exception:
            pass
            
    if pd.api.types.is_numeric_dtype(col_data):
        features[col] = col_data.fillna(col_data.median()).astype(np.float32)
    else:
        cat_cols.append(col)
        str_data = col_data.astype(str).fillna('missing')
        
        freq = str_data.value_counts(normalize=True).to_dict()
        features[f'{col}_freq'] = str_data.map(freq).fillna(0).astype(np.float32)
        
        cats = str_data.astype('category').cat.codes
        features[f'{col}_cat'] = cats.astype(np.int32)
        
        features[f'{col}_len'] = str_data.str.len().astype(np.float32)
        
        avg_words = str_data.str.split().str.len().mean()
        if avg_words > 1.5:
            try:
                tfidf = TfidfVectorizer(max_features=500, stop_words='english')
                tfidf_mat = tfidf.fit_transform(str_data)
                n_comp = min(10, tfidf_mat.shape[1] - 1)
                if n_comp > 0:
                    svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    svd_feats = svd.fit_transform(tfidf_mat)
                    for i in range(n_comp):
                        features[f'{col}_svd_{i}'] = svd_feats[:, i].astype(np.float32)
            except Exception:
                pass

X_train_proc = features.iloc[:n_train].copy()
X_test_proc = features.iloc[n_train:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in cat_cols:
    str_col_train = combined[col].iloc[:n_train].astype(str)
    str_col_test = combined[col].iloc[n_train:].astype(str)
    
    target_enc_train = np.zeros(n_train, dtype=np.float32)
    global_mean = float(y_train.mean())
    
    for tr_idx, val_idx in kf.split(X_train_proc):
        means = pd.DataFrame({'cat': str_col_train.iloc[tr_idx], 'target': y_train[tr_idx]}).groupby('cat')['target'].mean()
        target_enc_train[val_idx] = str_col_train.iloc[val_idx].map(means).fillna(global_mean)
        
    full_means = pd.DataFrame({'cat': str_col_train, 'target': y_train}).groupby('cat')['target'].mean()
    target_enc_test = str_col_test.map(full_means).fillna(global_mean)
    
    clean_col_name = re.sub(r'[^\w]+', '_', str(col))
    X_train_proc[f'{clean_col_name}_te'] = target_enc_train
    X_test_proc[f'{clean_col_name}_te'] = target_enc_test.values.astype(np.float32)

clean_cols = [re.sub(r'[^\w]+', '_', str(c)) for c in X_train_proc.columns]
X_train_proc.columns = clean_cols
X_test_proc.columns = clean_cols

oof_preds_lgb = np.zeros(n_train)
test_preds_lgb = np.zeros(len(X_test_proc))

oof_preds_cat = np.zeros(n_train)
test_preds_cat = np.zeros(len(X_test_proc))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 63,
    'n_estimators': 1500,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_proc)):
    X_tr, y_tr = X_train_proc.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y_train[val_idx]
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_preds_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test_proc) / kf.n_splits
    
    model_cat = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        eval_metric='RMSE',
        random_seed=42,
        verbose=0
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
    oof_preds_cat[val_idx] = model_cat.predict(X_va)
    test_preds_cat += model_cat.predict(X_test_proc) / kf.n_splits

final_test_preds = 0.5 * test_preds_lgb + 0.5 * test_preds_cat

if np.min(y_train) >= 0:
    final_test_preds = np.clip(final_test_preds, a_min=0, a_max=None)

submission = pd.DataFrame({
    'id': np.arange(len(final_test_preds)),
    'NADAC Per Unit': final_test_preds
})

submission.to_csv('predictions.csv', index=False)