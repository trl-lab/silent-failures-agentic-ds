import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/lending-club-loan/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/lending-club-loan/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 2. Preprocess target
target_col = 'loan_status'
le_target = LabelEncoder()
y = le_target.fit_transform(train_df[target_col].astype(str))

# Combine for consistent feature engineering
train_features = train_df.drop(columns=[target_col])
test_features = test_df.copy()
n_train = len(train_features)
df = pd.concat([train_features, test_features], axis=0, ignore_index=True)

# 3. Feature Engineering
# Drop high-cardinality/unstructured text and identifier columns
drop_cols = ['id', 'member_id', 'url', 'desc', 'emp_title', 'title']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Parse percentages
for col in ['int_rate', 'revol_util']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.rstrip('%').replace({'None': np.nan, 'nan': np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Parse term (e.g. " 36 months" -> 36)
if 'term' in df.columns:
    df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype(float)

# Parse emp_length (e.g. "10+ years" -> 10, "< 1 year" -> 0)
if 'emp_length' in df.columns:
    emp_len = df['emp_length'].astype(str).str.extract(r'(\d+)').astype(float)
    df['emp_length_num'] = emp_len[0]

# Parse date columns
for date_col in ['issue_d', 'earliest_cr_line']:
    if date_col in df.columns:
        parsed_date = pd.to_datetime(df[date_col], format='%b-%y', errors='coerce')
        df[f'{date_col}_year'] = parsed_date.dt.year
        df[f'{date_col}_month'] = parsed_date.dt.month
        df.drop(columns=[date_col], inplace=True)

# Handle categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype('category')

# Split back into train and test
X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

# 4. Model Training with K-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

# Prepare for CatBoost (convert category to str/int codes or handle NaNs)
cat_cols_list = [col for col in X_train.select_dtypes(include=['category']).columns]
X_train_cat = X_train.copy()
X_test_cat = X_test.copy()
for col in cat_cols_list:
    X_train_cat[col] = X_train_cat[col].astype(str).fillna('missing')
    X_test_cat[col] = X_test_cat[col].astype(str).fillna('missing')

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train.iloc[train_idx], y[train_idx],
        eval_set=[(X_train.iloc[val_idx], y[val_idx])],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict_proba(X_test)[:, 1] / n_splits

    # CatBoost
    cb_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_cols_list,
        random_seed=42 + fold,
        verbose=False
    )
    cb_model.fit(
        X_train_cat.iloc[train_idx], y[train_idx],
        eval_set=(X_train_cat.iloc[val_idx], y[val_idx]),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cb_model.predict_proba(X_test_cat)[:, 1] / n_splits

# Ensemble predictions
final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

# 5. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'loan_status': final_preds
})
submission_df.to_csv('predictions.csv', index=False)