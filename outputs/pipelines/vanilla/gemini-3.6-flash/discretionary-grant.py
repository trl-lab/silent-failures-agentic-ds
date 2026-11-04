import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/discretionary-grant/data/train.parquet"
test_path  = "/path/to/user/workspace/ds_failures/datasets/strable/discretionary-grant/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "AwardTotal"
y_train = train[target_col].values.astype(np.float64)
X_train = train.drop(columns=[target_col])
X_test = test.copy()

n_train = len(train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

num_cols = []
cat_cols = []
text_cols = []
date_cols = []

for col in df_all.columns:
    if pd.api.types.is_datetime64_any_dtype(df_all[col]):
        date_cols.append(col)
        continue
    if pd.api.types.is_numeric_dtype(df_all[col]):
        num_cols.append(col)
        continue
    
    s = df_all[col].astype(str)
    avg_len = s.str.len().mean()
    n_unique = df_all[col].nunique()
    
    if avg_len > 50 or (n_unique > 500 and avg_len > 20):
        text_cols.append(col)
    else:
        cat_cols.append(col)

for col in date_cols:
    df_all[f"{col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all.drop(columns=[col], inplace=True)

for col in list(cat_cols):
    if any(k in col.lower() for k in ['date', 'year', 'time']):
        try:
            parsed = pd.to_datetime(df_all[col], errors='coerce')
            if parsed.notna().sum() > 0.5 * len(df_all):
                df_all[f"{col}_year"] = parsed.dt.year
                df_all[f"{col}_month"] = parsed.dt.month
                df_all[f"{col}_day"] = parsed.dt.day
                df_all[f"{col}_dayofweek"] = parsed.dt.dayofweek
                cat_cols.remove(col)
                df_all.drop(columns=[col], inplace=True)
        except Exception:
            pass

df_all['missing_count'] = df_all.isnull().sum(axis=1)

for col in text_cols:
    text_data = df_all[col].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(text_data)
    n_comp = min(16, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 1
    if n_comp > 0:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        for i in range(n_comp):
            df_all[f"{col}_svd_{i}"] = svd_mat[:, i]
    df_all.drop(columns=[col], inplace=True)

for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna("missing")
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_all[col] = oe.fit_transform(df_all[[col]])

X_all_mat = df_all.values.astype(np.float32)

X_train_mat = X_all_mat[:n_train]
X_test_mat = X_all_mat[n_train:]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(test))
cat_preds = np.zeros(len(test))
xgb_preds = np.zeros(len(test))

for train_idx, val_idx in kf.split(X_train_mat, y_train):
    X_tr, y_tr = X_train_mat[train_idx], y_train[train_idx]
    X_va, y_va = X_train_mat[val_idx], y_train[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test_mat) / kf.n_splits

    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50
    )
    cat_preds += model_cat.predict(X_test_mat) / kf.n_splits

    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=50
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test_mat) / kf.n_splits

final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0
final_preds = np.maximum(0, final_preds)

sub = pd.DataFrame({
    "id": np.arange(len(test)),
    "AwardTotal": final_preds
})
sub.to_csv("predictions.csv", index=False)