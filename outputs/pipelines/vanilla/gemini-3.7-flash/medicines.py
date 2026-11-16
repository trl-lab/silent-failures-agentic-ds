import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/medicines/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/medicines/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Medicine status'

# 2. Clean feature column names for compatibility with all models (LightGBM, XGBoost, etc.)
def sanitize_colname(col):
    return re.sub(r'[^\w\s]', '_', str(col)).strip().replace(' ', '_')

X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

feature_cols = list(X_train_raw.columns)

# 3. Fill missing values
for col in feature_cols:
    X_train_raw[col] = X_train_raw[col].fillna('Missing').astype(str)
    X_test_raw[col] = X_test_raw[col].fillna('Missing').astype(str)

# Combine all text for TF-IDF
train_text = X_train_raw.apply(lambda row: ' '.join(row), axis=1)
test_text = X_test_raw.apply(lambda row: ' '.join(row), axis=1)

tfidf = TfidfVectorizer(max_features=500, stop_words='english', token_pattern=r'(?u)\b\w+\b')
X_train_tfidf = tfidf.fit_transform(train_text).toarray()
X_test_tfidf = tfidf.transform(test_text).toarray()

# Categorical label encoding
X_train_num = pd.DataFrame(index=X_train_raw.index)
X_test_num = pd.DataFrame(index=X_test_raw.index)

for col in feature_cols:
    safe_col = sanitize_colname(col)
    le = LabelEncoder()
    combined = pd.concat([X_train_raw[col], X_test_raw[col]], axis=0).astype(str)
    le.fit(combined)
    X_train_num[safe_col] = le.transform(X_train_raw[col])
    X_test_num[safe_col] = le.transform(X_test_raw[col])

# Combine tabular + TF-IDF with safe column names
tfidf_cols = [f'tfidf_{i}' for i in range(X_train_tfidf.shape[1])]
df_tfidf_train = pd.DataFrame(X_train_tfidf, columns=tfidf_cols, index=X_train_raw.index)
df_tfidf_test = pd.DataFrame(X_test_tfidf, columns=tfidf_cols, index=X_test_raw.index)

X_train_all = pd.concat([X_train_num, df_tfidf_train], axis=1)
X_test_all = pd.concat([X_test_num, df_tfidf_test], axis=1)

# Ensure completely safe column names
X_train_all.columns = [f'feature_{i}_{sanitize_colname(c)}' for i, c in enumerate(X_train_all.columns)]
X_test_all.columns = X_train_all.columns

# Target encoding
le_target = LabelEncoder()
y = le_target.fit_transform(train_df[target_col].astype(str))

# 4. Model Training with K-Fold Ensembling
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cat_preds = np.zeros(len(test_df))
lgb_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y)):
    X_tr, y_tr = X_train_all.iloc[train_idx], y[train_idx]
    X_va, y_val = X_train_all.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_preds += lgb_model.predict_proba(X_test_all)[:, 1] / n_splits
    
    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.04,
        depth=6,
        eval_metric='AUC',
        random_seed=42 + fold,
        verbose=False
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_val), early_stopping_rounds=50, verbose=False)
    cat_preds += cb_model.predict_proba(X_test_all)[:, 1] / n_splits

# Ensemble average
final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

# 5. Output predictions
submission = pd.DataFrame({
    'id': range(len(test_df)),
    target_col: final_preds
})

submission.to_csv('predictions.csv', index=False)