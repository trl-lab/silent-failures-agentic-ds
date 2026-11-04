import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/meta-critic_whisky/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/meta-critic_whisky/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Meta Critic'
y = train_df[target_col].values.astype(float)
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.drop(columns=[target_col], errors='ignore')

feature_cols = [c for c in X_train_raw.columns if c in X_test_raw.columns]
if 'id' in feature_cols:
    feature_cols.remove('id')
if 'ID' in feature_cols:
    feature_cols.remove('ID')

X_train_df = X_train_raw[feature_cols].copy()
X_test_df = X_test_raw[feature_cols].copy()

n_train = len(X_train_df)
df_all = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)

num_cols = []
cat_cols = []
text_cols = []

for col in df_all.columns:
    if pd.api.types.is_numeric_dtype(df_all[col]):
        num_cols.append(col)
    else:
        df_all[col] = df_all[col].astype(str).fillna('')
        s_clean = df_all[col].str.replace('%', '', regex=False).str.replace('$', '', regex=False).str.extract(r'(\d+\.?\d*)')[0]
        converted = pd.to_numeric(s_clean, errors='coerce')
        if converted.notnull().mean() > 0.5:
            ext_col = f'{col}_extracted_num'
            df_all[ext_col] = converted
            num_cols.append(ext_col)
        
        nunique = df_all[col].nunique()
        avg_len = df_all[col].str.len().mean()
        if avg_len > 50 or nunique > 500:
            text_cols.append(col)
        else:
            cat_cols.append(col)

if text_cols:
    combined_text = df_all[text_cols].astype(str).agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    text_tfidf = tfidf.fit_transform(combined_text)
    
    svd_dim = min(30, text_tfidf.shape[1] - 1)
    if svd_dim > 0:
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        svd_feats = svd.fit_transform(text_tfidf)
        svd_cols = [f'svd_{i}' for i in range(svd_dim)]
        svd_df = pd.DataFrame(svd_feats, columns=svd_cols, index=df_all.index)
        df_all = pd.concat([df_all, svd_df], axis=1)
        num_cols.extend(svd_cols)

for col in cat_cols:
    freq = df_all[col].value_counts()
    freq_col = f'{col}_freq'
    df_all[freq_col] = df_all[col].map(freq)
    num_cols.append(freq_col)

X_all_num = df_all[num_cols].copy()
for col in X_all_num.columns:
    X_all_num[col] = X_all_num[col].fillna(X_all_num[col].median())

for col in cat_cols:
    df_all[col] = df_all[col].astype('category')

X_train_num = X_all_num.iloc[:n_train]
X_test_num = X_all_num.iloc[n_train:]

X_train_cat = df_all[cat_cols].iloc[:n_train]
X_test_cat = df_all[cat_cols].iloc[n_train:]

X_train_tree = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_tree = pd.concat([X_test_num, X_test_cat], axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
preds_lgb = np.zeros(len(X_test_df))

oof_cb = np.zeros(n_train)
preds_cb = np.zeros(len(X_test_df))

oof_xgb = np.zeros(n_train)
preds_xgb = np.zeros(len(X_test_df))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 1000
}

X_train_xgb = X_train_num.values
X_test_xgb = X_test_num.values

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tree)):
    X_tr, y_tr = X_train_tree.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_tree.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va)
    preds_lgb += model_lgb.predict(X_test_tree) / 5.0
    
    X_tr_cb = X_tr.copy()
    X_va_cb = X_va.copy()
    X_te_cb = X_test_tree.copy()
    for col in cat_cols:
        X_tr_cb[col] = X_tr_cb[col].astype(str)
        X_va_cb[col] = X_va_cb[col].astype(str)
        X_te_cb[col] = X_te_cb[col].astype(str)
        
    model_cb = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        cat_features=cat_cols if cat_cols else None,
        verbose=False
    )
    model_cb.fit(
        X_tr_cb, y_tr,
        eval_set=(X_va_cb, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    oof_cb[val_idx] = model_cb.predict(X_va_cb)
    preds_cb += model_cb.predict(X_te_cb) / 5.0

    X_tr_x, X_va_x = X_train_xgb[train_idx], X_train_xgb[val_idx]
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    model_xgb.fit(
        X_tr_x, y_tr,
        eval_set=[(X_va_x, y_va)],
        verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict(X_va_x)
    preds_xgb += model_xgb.predict(X_test_xgb) / 5.0

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

best_w = [1/3, 1/3, 1/3]
best_score = rmse(y, oof_lgb * best_w[0] + oof_cb * best_w[1] + oof_xgb * best_w[2])

for w1 in np.linspace(0, 1, 21):
    for w2 in np.linspace(0, 1 - w1, 21):
        w3 = 1.0 - w1 - w2
        score = rmse(y, oof_lgb * w1 + oof_cb * w2 + oof_xgb * w3)
        if score < best_score:
            best_score = score
            best_w = [w1, w2, w3]

final_preds = preds_lgb * best_w[0] + preds_cb * best_w[1] + preds_xgb * best_w[2]

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Meta Critic': final_preds
})

output_df.to_csv('predictions.csv', index=False)