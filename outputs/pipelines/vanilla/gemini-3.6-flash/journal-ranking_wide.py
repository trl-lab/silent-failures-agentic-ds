import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/journal-ranking_wide/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/journal-ranking_wide/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "H index"
y = train_df[target_col].astype(float).values

train_X = train_df.drop(columns=[target_col])
test_X = test_df.copy()

n_train = len(train_X)
combined = pd.concat([train_X, test_X], axis=0, ignore_index=True)

processed_cols = {}

for col in combined.columns:
    series = combined[col]
    if pd.api.types.is_numeric_dtype(series):
        processed_cols[col] = series.astype(np.float32)
    elif pd.api.types.is_datetime64_any_dtype(series):
        processed_cols[f"{col}_year"] = series.dt.year.fillna(-1).astype(np.float32)
        processed_cols[f"{col}_month"] = series.dt.month.fillna(-1).astype(np.float32)
        processed_cols[f"{col}_day"] = series.dt.day.fillna(-1).astype(np.float32)
    else:
        str_series = series.astype(str).fillna("missing")
        cat_encoded, _ = pd.factorize(str_series)
        processed_cols[col] = cat_encoded.astype(np.float32)
        
        if str_series.nunique() > 20 or str_series.str.contains(" ").any():
            try:
                tfidf = TfidfVectorizer(max_features=5000, stop_words='english', token_pattern=r'(?u)\b\w+\b')
                tfidf_mat = tfidf.fit_transform(str_series)
                if tfidf_mat.shape[1] >= 5:
                    n_comp = min(10, tfidf_mat.shape[1] - 1)
                    svd = TruncatedSVD(n_components=n_comp, random_state=42)
                    svd_mat = svd.fit_transform(tfidf_mat)
                    for c in range(n_comp):
                        processed_cols[f"{col}_svd_{c}"] = svd_mat[:, c].astype(np.float32)
            except Exception:
                pass

X_all = pd.DataFrame(processed_cols)
X_all.columns = [f"f_{i}" for i in range(X_all.shape[1])]

X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train, y):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += model_lgb.predict(X_test) / kf.n_splits
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        tree_method="hist"
    )
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += model_xgb.predict(X_test) / kf.n_splits

    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0,
        thread_count=-1,
        early_stopping_rounds=50
    )
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    cat_preds += model_cat.predict(X_test) / kf.n_splits

final_preds = (lgb_preds + xgb_preds + cat_preds) / 3.0
final_preds = np.maximum(0, final_preds)

sub = pd.DataFrame({
    "id": np.arange(len(final_preds)),
    "H index": final_preds
})

sub.to_csv("predictions.csv", index=False)