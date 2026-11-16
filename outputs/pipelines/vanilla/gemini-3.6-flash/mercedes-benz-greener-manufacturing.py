import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, GaussianRandomProjection, SparseRandomProjection
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/mercedes-benz-greener-manufacturing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/mercedes-benz-greener-manufacturing/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

train_df = train_df[train_df['y'] < 170].reset_index(drop=True)

y_train = train_df['y'].values
X_train_raw = train_df.drop(columns=['y'])
X_test_raw = test_df.copy()

df_all = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

cat_cols = [c for c in df_all.columns if df_all[c].dtype == 'object']

for c in cat_cols:
    freq = df_all[c].value_counts()
    df_all[f'{c}_freq'] = df_all[c].map(freq)
    le = LabelEncoder()
    df_all[c] = le.fit_transform(df_all[c].astype(str))

usable_cols = [c for c in df_all.columns if df_all[c].nunique() > 1]
df_all = df_all[usable_cols]

binary_cols = [c for c in usable_cols if c.startswith('X') and c not in cat_cols]

if len(binary_cols) > 0:
    binary_data = df_all[binary_cols].values
    df_all['row_sum_ones'] = binary_data.sum(axis=1)
    df_all['row_mean_ones'] = binary_data.mean(axis=1)
    df_all['row_std_ones'] = binary_data.std(axis=1)

    n_comp = 10
    pca = PCA(n_components=n_comp, random_state=42)
    pca_results = pca.fit_transform(binary_data)

    ica = FastICA(n_components=n_comp, random_state=42, max_iter=1000)
    ica_results = ica.fit_transform(binary_data)

    tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
    tsvd_results = tsvd.fit_transform(binary_data)

    grp = GaussianRandomProjection(n_components=n_comp, random_state=42)
    grp_results = grp.fit_transform(binary_data)

    srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=42)
    srp_results = srp.fit_transform(binary_data)

    for i in range(n_comp):
        df_all[f'pca_{i}'] = pca_results[:, i]
        df_all[f'ica_{i}'] = ica_results[:, i]
        df_all[f'tsvd_{i}'] = tsvd_results[:, i]
        df_all[f'grp_{i}'] = grp_results[:, i]
        df_all[f'srp_{i}'] = srp_results[:, i]

df_all = df_all.fillna(0)

X_train = df_all.iloc[:len(train_df)].copy()
X_test = df_all.iloc[len(train_df):].copy()

kf = KFold(n_splits=10, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
test_lgb = np.zeros(len(test_df))

oof_xgb = np.zeros(len(train_df))
test_xgb = np.zeros(len(test_df))

oof_cat = np.zeros(len(train_df))
test_cat = np.zeros(len(test_df))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[trn_idx], y_train[trn_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.005,
        num_leaves=12,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42 + fold,
        verbosity=-1
    )
    model_lgb.fit(X_tr, y_tr, eval