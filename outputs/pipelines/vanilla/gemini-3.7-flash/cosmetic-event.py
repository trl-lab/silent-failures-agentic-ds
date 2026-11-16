import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/cosmetic-event/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/cosmetic-event/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate target
target_col = 'outcomes'
y_raw = train_df[target_col].astype(str)

le_target = LabelEncoder()
y_train = le_target.fit_transform(y_raw)
n_classes = len(le_target.classes_)

# Combine data for consistent preprocessing
df_all = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)
n_train = len(train_df)
n_test = len(test_df)

# Feature Engineering
# Normalize age to years
age_num = pd.to_numeric(df_all['age'], errors='coerce')
age_unit = df_all['age_unit'].fillna('').astype(str).str.lower()

age_normalized = age_num.copy()
age_normalized[age_unit.str.contains('month', na=False)] /= 12.0
age_normalized[age_unit.str.contains('day', na=False)] /= 365.0
age_normalized[age_unit.str.contains('week', na=False)] /= 52.0
age_normalized[age_unit.str.contains('decade', na=False)] *= 10.0

# Process date features
def parse_date(col):
    return pd.to_datetime(col, errors='coerce', format='mixed')

df_all['event_dt'] = parse_date(df_all['event_date'])
df_all['init_recv_dt'] = parse_date(df_all['initial_received_date'])
df_all['latest_recv_dt'] = parse_date(df_all['latest_received_date'])

event_year = df_all['event_dt'].dt.year.fillna(-1)
event_month = df_all['event_dt'].dt.month.fillna(-1)
init_year = df_all['init_recv_dt'].dt.year.fillna(-1)
latest_year = df_all['latest_recv_dt'].dt.year.fillna(-1)
days_to_recv = (df_all['latest_recv_dt'] - df_all['event_dt']).dt.days.fillna(-999)

# Report version
report_ver = pd.to_numeric(df_all['report_version'], errors='coerce').fillna(-1)

# Tabular numeric features
num_features = pd.DataFrame({
    'age': age_normalized.fillna(-1),
    'has_age': (~age_normalized.isna()).astype(int),
    'event_year': event_year,
    'event_month': event_month,
    'init_year': init_year,
    'latest_year': latest_year,
    'days_to_recv': days_to_recv,
    'report_version': report_ver
})

# Categorical features
cat_cols = ['gender', 'age_unit', 'report_type']
for c in cat_cols:
    s = df_all[c].fillna('missing').astype(str)
    le = LabelEncoder()
    num_features[c] = le.fit_transform(s)

# Text features via TF-IDF
tfidf_reactions = TfidfVectorizer(max_features=250, token_pattern=r'(?u)\b\w+\b')
reactions_tfidf = tfidf_reactions.fit_transform(df_all['reactions'].fillna('').astype(str))

tfidf_products = TfidfVectorizer(max_features=250, token_pattern=r'(?u)\b\w+\b')
products_tfidf = tfidf_products.fit_transform(df_all['products'].fillna('').astype(str))

# Combine all features into a sparse matrix
X_sparse = hstack([
    csr_matrix(num_features.values),
    reactions_tfidf,
    products_tfidf
]).tocsr()

X_train = X_sparse[:n_train]
X_test = X_sparse[n_train:]

# 2. Train model with Stratified K-Fold ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((n_test, n_classes))

lgb_params = {
    'objective': 'multiclass',
    'num_class': n_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 300,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    test_preds += model.predict_proba(X_test) / skf.n_splits

# 3. Generate predictions
final_pred_indices = np.argmax(test_preds, axis=1)
final_predictions = le_target.inverse_transform(final_pred_indices)

# 4. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'outcomes': final_predictions
})
submission_df.to_csv('predictions.csv', index=False)