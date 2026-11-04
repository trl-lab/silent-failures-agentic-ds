import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-advisory-services-projects/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-advisory-services-projects/data/test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target_col = "Estimated Total Budget ($)"

    # Clean missing targets in train if any
    train = train.dropna(subset=[target_col]).reset_index(drop=True)
    y_train = train[target_col].values.astype(np.float64)

    X_train = train.drop(columns=[target_col])
    X_test = test.drop(columns=[target_col]) if target_col in test.columns else test.copy()

    n_train = len(X_train)
    n_test = len(X_test)

    df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

    features_df = pd.DataFrame(index=df_all.index)

    cat_cols = []
    text_cols = []

    # Process individual columns
    for col in df_all.columns:
        col_data = df_all[col]
        
        # Check datetime
        if 'date' in col.lower() or 'year' in col.lower() or pd.api.types.is_datetime64_any_dtype(col_data):
            try:
                dt_col = pd.to_datetime(col_data, errors='coerce')
                if dt_col.notnull().sum() > 0.3 * len(df_all):
                    features_df[f"{col}_year"] = dt_col.dt.year.astype(np.float32)
                    features_df[f"{col}_month"] = dt_col.dt.month.astype(np.float32)
                    features_df[f"{col}_day"] = dt_col.dt.day.astype(np.float32)
                    features_df[f"{col}_dayofweek"] = dt_col.dt.dayofweek.astype(np.float32)
                    features_df[f"{col}_quarter"] = dt_col.dt.quarter.astype(np.float32)
                    min_date = dt_col.min()
                    features_df[f"{col}_days"] = (dt_col - min_date).dt.days.astype(np.float32)
                    continue
            except Exception:
                pass

        # Check numeric
        if pd.api.types.is_numeric_dtype(col_data):
            features_df[col] = col_data.astype(np.float32)
        else:
            str_series = col_data.fillna("").astype(str)
            n_unique = str_series.nunique()
            if n_unique <= 100:
                cat_cols.append(col)
                freq = str_series.value_counts()
                features_df[f"{col}_freq"] = str_series.map(freq).astype(np.float32)
                features_df[f"{col}_cat"] = pd.factorize(str_series)[0].astype(np.int32)
            else:
                text_cols.append(col)
                features_df[f"{col}_len"] = str_series.apply(len).astype(np.float32)
                features_df[f"{col}_words"] = str_series.apply(lambda x: len(x.split())).astype(np.float32)

    # TF-IDF + SVD on string/text columns
    all_str_cols = cat_cols + text_cols
    if all_str_cols:
        combined_text = df_all[all_str_cols].fillna("").astype(str).agg(" ".join, axis=1)
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        tfidf_mat = tfidf.fit_transform(combined_text)
        svd = TruncatedSVD(n_components=20, random_state=42)
        svd_feats = svd.fit_transform(tfidf_mat)
        for i in range(20):
            features_df[f"tfidf_svd_{i}"] = svd_feats[:, i].astype(np.float32)

    # Out-of-fold target encoding for categoricals
    kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
    for col in cat_cols:
        str_series = df_all[col].fillna("").astype(str)
        train_str = str_series.iloc[:n_train]
        test_str = str_series.iloc[n_train:]
        
        te_train = np.zeros(n_train, dtype=np.float32)
        global_mean = float(np.mean(y_train))
        
        for tr_idx, val_idx in kf_te.split(train_str):
            tr_df = pd.DataFrame({"cat": train_str.iloc[tr_idx], "target": y_train[tr_idx]})
            means = tr_df.groupby("cat")["target"].mean()
            te_train[val_idx] = train_str.iloc[val_idx].map(means).fillna(global_mean)
            
        full_tr_df = pd.DataFrame({"cat": train_str, "target": y_train})
        full_means = full_tr_df.groupby("cat")["target"].mean()
        te_test = test_str.map(full_means).fillna(global_mean).values.astype(np.float32)
        
        features_df[f"{col}_te"] = np.concatenate([te_train, te_test])

    # Standardize column names to prevent LGBM/CatBoost key issues
    features_df.columns = [f"feat_{i}" for i in range(features_df.shape[1])]

    X_tr = features_df.iloc[:n_train].copy()
    X_te = features_df.iloc[n_train:].copy()

    # Modeling
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    preds_raw = np.zeros(n_test)
    preds_log = np.zeros(n_test)

    min_y = float(np.min(y_train))
    shift = 1.0 - min_y if min_y <= 0 else 0.0
    y_train_log = np.log1p(y_train + shift)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr)):
        X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y_train[tr_idx]
        X_val_f, y_val_f = X_tr.iloc[val_idx], y_train[val_idx]
        y_tr_log_f = y_train_log[tr_idx]
        y_val_log_f = y_train_log[val_idx]

        # 1. LightGBM Raw
        lgb_raw = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=-1
        )
        lgb_raw.fit(
            X_tr_f, y_tr_f,
            eval_set=[(X_val_f, y_val_f)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds_raw += lgb_raw.predict(X_te) / (n_splits * 2)

        # 2. LightGBM Log
        lgb_log = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=-1
        )
        lgb_log.fit(
            X_tr_f, y_tr_log_f,
            eval_set=[(X_val_f, y_val_log_f)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds_log += (np.expm1(lgb_log.predict(X_te)) - shift) / (n_splits * 2)

        # 3. CatBoost Raw
        cb_raw = cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1
        )
        cb_raw.fit(
            X_tr_f, y_tr_f,
            eval_set=(X_val_f, y_val_f),
            early_stopping_rounds=50
        )
        preds_raw += cb_raw.predict(X_te) / (n_splits * 2)

        # 4. CatBoost Log
        cb_log = cb.CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1
        )
        cb_log.fit(
            X_tr_f, y_tr_log_f,
            eval_set=(X_val_f, y_val_log_f),
            early_stopping_rounds=50
        )
        preds_log += (np.expm1(cb_log.predict(X_te)) - shift) / (n_splits * 2)

    final_preds = 0.5 * preds_raw + 0.5 * preds_log

    if min_y >= 0:
        final_preds = np.maximum(0, final_preds)

    sub = pd.DataFrame({
        "id": np.arange(len(final_preds)),
        target_col: final_preds
    })
    sub.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()