import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/clear-corpus/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/clear-corpus/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'BT Easiness'
y = train_df[target_col].values

common_cols = [c for c in train_df.columns if c in test_df.columns and c != target_col]

text_cols = []
num_cols = []
cat_cols = []

for col in common_cols:
    if pd.api.types.is_numeric_dtype(train_df[col]):
        if train_df[col].nunique() == len(train_df) and 'id' in col.lower():
            continue
        num_cols.append(col)
    else:
        sample_str = train_df[col].dropna().astype(str)
        if len(sample_str) > 0 and sample_str.str.len().mean() > 50:
            text_cols.append(col)
        else:
            cat_cols.append(col)

if text_cols:
    train_text = train_df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
    test_text = test_df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
else:
    obj_cols = [c for c in common_cols if c not in num_cols]
    if obj_cols:
        train_text = train_df[obj_cols].fillna('').astype(str).agg(' '.join, axis=1)
        test_text = test_df[obj_cols].fillna('').astype(str).agg(' '.join, axis=1)
    else:
        train_text = pd.Series([''] * len(train_df))
        test_text = pd.Series([''] * len(test_df))

tfidf = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
X_tfidf_train = tfidf.fit_transform(train_text)
X_tfidf_test = tfidf.transform(test_text)

svd = TruncatedSVD(n_components=50, random_state=42)
X_svd_train = svd.fit_transform(X_tfidf_train)
X_svd_test = svd.transform(X_tfidf_test)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_ridge = np.zeros(len(train_df))
test_ridge = np.zeros(len(test_df))

oof_ridge2 = np.zeros(len(train_df))
test_ridge2 = np.zeros(len(test_df))

for train_idx, val_idx in kf.split(train_df):
    X_tr, y_tr = X_tfidf_train[train_idx], y[train_idx]
    X_va = X_tfidf_train[val_idx]
    
    r1 = Ridge(alpha=1.0, random_state=42)
    r1.fit(X_tr, y_tr)
    oof_ridge[val_idx] = r1.predict(X_va)
    test_ridge += r1.predict(X_tfidf_test) / n_splits
    
    r2 = Ridge(alpha=5.0, random_state=42)
    r2.fit(X_tr, y_tr)
    oof_ridge2[val_idx] = r2.predict(X_va)
    test_ridge2 += r2.predict(X_tfidf_test) / n_splits

num_train_list = []
num_test_list = []

for c in num_cols:
    med = train_df[c].median()
    if pd.isna(med):
        med = 0
    num_train_list.append(train_df[c].fillna(med).values)
    num_test_list.append(test_df[c].fillna(med).values)

if num_train_list:
    X_num_train = np.column_stack(num_train_list)
    X_num_test = np.column_stack(num_test_list)
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num_train)
    X_num_test = scaler.transform(X_num_test)
else:
    X_num_train = np.empty((len(train_df), 0))
    X_num_test = np.empty((len(test_df), 0))

cat_train_list = []
cat_test_list = []
for c in cat_cols:
    train_c = train_df[c].astype(str).fillna('missing')
    test_c = test_df[c].astype(str).fillna('missing')
    categories = list(set(train_c.unique()).union(set(test_c.unique())))
    cat_map = {cat: i for i, cat in enumerate(categories)}
    cat_train_list.append(train_c.map(cat_map).values)
    cat_test_list.append(test_c.map(cat_map).values)

if cat_train_list:
    X_cat_train = np.column_stack(cat_train_list)
    X_cat_test = np.column_stack(cat_test_list)
else:
    X_cat_train = np.empty((len(train_df), 0))
    X_cat_test = np.empty((len(test_df), 0))

X_tab_train = np.column_stack([
    X_num_train,
    X_cat_train,
    X_svd_train,
    oof_ridge.reshape(-1, 1),
    oof_ridge2.reshape(-1, 1)
])

X_tab_test = np.column_stack([
    X_num_test,
    X_cat_test,
    X_svd_test,
    test_ridge.reshape(-1, 1),
    test_ridge2.reshape(-1, 1)
])

lgb_oof = np.zeros(len(train_df))
lgb_test = np.zeros(len(test_df))

cb_oof = np.zeros(len(train_df))
cb_test = np.zeros(len(test_df))

xgb_oof = np.zeros(len(train_df))
xgb_test = np.zeros(len(test_df))

for train_idx, val_idx in kf.split(train_df):
    X_tr, y_tr = X_tab_train[train_idx], y[train_idx]
    X_va, y_va = X_tab_train[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_oof[val_idx] = model_lgb.predict(X_va)
    lgb_test += model_lgb.predict(X_tab_test) / n_splits
    
    model_cb = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0
    )
    model_cb.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    cb_oof[val_idx] = model_cb.predict(X_va)
    cb_test += model_cb.predict(X_tab_test) / n_splits

    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_oof[val_idx] = model_xgb.predict(X_va)
    xgb_test += model_xgb.predict(X_tab_test) / n_splits

oof_matrix = np.column_stack([oof_ridge, oof_ridge2, lgb_oof, cb_oof, xgb_oof])
test_matrix = np.column_stack([test_ridge, test_ridge2, lgb_test, cb_test, xgb_test])

meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(oof_matrix, y)
final_preds = meta_model.predict(test_matrix)

submission = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'BT Easiness': final_preds
})
submission.to_csv('predictions.csv', index=False)