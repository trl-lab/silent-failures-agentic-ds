import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

def main():
    train_path = '/path/to/user/workspace/ds_failures/datasets/strable/grant/data/train.parquet'
    test_path  = '/path/to/user/workspace/ds_failures/datasets/strable/grant/data/test.parquet'

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target_col = 'Federal Funds'

    # Remove rows with missing target in train
    train = train.dropna(subset=[target_col]).reset_index(drop=True)
    y = train[target_col].values

    X_train = train.drop(columns=[target_col])
    X_test = test.copy()

    n_train = len(X_train)
    df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    # Drop constant columns
    cols_to_drop = [c for c in df_all.columns if df_all[c].nunique(dropna=True) <= 1]
    if cols_to_drop:
        df_all = df_all.drop(columns=cols_to_drop)

    num_cols = []
    cat_cols = []
    text_cols = []

    for col in df_all.columns:
        if df_all[col].dtype == 'object' or pd.api.types.is_string_dtype(df_all[col]):
            sample = df_all[col].dropna().astype(str).head(100)
            avg_len = sample.str.len().mean() if len(sample) > 0 else 0
            
            is_date = False
            if avg_len < 30:
                try:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() > 0.5 * len(sample):
                        is_date = True
                except Exception:
                    pass
            
            if is_date:
                dt_col = pd.to_datetime(df_all[col], errors='coerce')
                df_all[f'{col}_year'] = dt_col.dt.year
                df_all[f'{col}_month'] = dt_col.dt.month
                df_all[f'{col}_day'] = dt_col.dt.day
                df_all[f'{col}_dayofweek'] = dt_col.dt.dayofweek
                df_all = df_all.drop(columns=[col])
                num_cols.extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'])
                continue

            if avg_len > 80 and df_all[col].nunique() > 100:
                text_cols.append(col)
            else:
                cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df_all[col]):
            num_cols.append(col)

    # Text extraction via TF-IDF + SVD
    svd_dfs = []
    for col in text_cols:
        text_data = df_all[col].fillna('').astype(str)
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
        tfidf_mat = tfidf.fit_transform(text_data)
        n_comp = min(15, max(1, tfidf_mat.shape[1] - 1))
        if n_comp > 0 and tfidf_mat.shape[1] > 1:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd_mat = svd.fit_transform(tfidf_mat)
            svd_cols = [f'{col}_svd_{i}' for i in range(n_comp)]
            svd_df = pd.DataFrame(svd_mat, columns=svd_cols, index=df_all.index)
            svd_dfs.append(svd_df)

    if text_cols:
        df_all = df_all.drop(columns=text_cols)
    if svd_dfs:
        df_all = pd.concat([df_all] + svd_dfs, axis=1)

    # Frequency encoding
    for col in cat_cols:
        freq = df_all[col].value_counts(dropna=False).to_dict()
        df_all[f'{col}_freq'] = df_all[col].map(freq)
        df_all[col] = df_all[col].astype(str).astype('category')

    # Split into train/test
    X_train_proc = df_all.iloc[:n_train].copy()
    X_test_proc = df_all.iloc[n_train:].copy()

    # Out-of-fold target encoding for high quality features
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = y.mean()

    for col in cat_cols:
        tr_te = pd.Series(index=X_train_proc.index, dtype=float)
        for train_idx, val_idx in kf.split(X_train_proc):
            tr_cat = X_train_proc.iloc[train_idx][col]
            tr_y = y[train_idx]
            cat_means = pd.Series(tr_y).groupby(tr_cat).mean().to_dict()
            tr_te.iloc[val_idx] = X_train_proc.iloc[val_idx][col].map(cat_means).fillna(global_mean)
        
        full_means = pd.Series(y).groupby(X_train_proc[col]).mean().to_dict()
        te_te = X_test_proc[col].map(full_means).fillna(global_mean)
        
        X_train_proc[f'{col}_te'] = tr_te
        X_test_proc[f'{col}_te'] = te_te

        # Label encode original categoricals
        X_train_proc[col] = X_train_proc[col].cat.codes
        X_test_proc[col] = X_test_proc[col].cat.codes

    test_lgb = np.zeros(len(X_test_proc))
    test_xgb = np.zeros(len(X_test_proc))
    test_cat = np.zeros(len(X_test_proc))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_proc)):
        X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

        # LightGBM
        model_lgb = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42 + fold,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=-1
        )
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        test_lgb += model_lgb.predict(X_test_proc) / kf.n_splits

        # XGBoost
        model_xgb = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            random_state=42 + fold,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50
        )
        model_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        test_xgb += model_xgb.predict(X_test_proc) / kf.n_splits

        # CatBoost
        model_cat = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.04,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50
        )
        model_cat.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            verbose=False
        )
        test_cat += model_cat.predict(X_test_proc) / kf.n_splits

    # Ensemble predictions
    final_preds = (test_lgb + test_xgb + test_cat) / 3.0

    min_y = y.min()
    if min_y >= 0:
        final_preds = np.clip(final_preds, a_min=0, a_max=None)

    sub = pd.DataFrame({
        'id': np.arange(len(final_preds)),
        'Federal Funds': final_preds
    })
    sub.to_csv('predictions.csv', index=False)

if __name__ == '__main__':
    main()