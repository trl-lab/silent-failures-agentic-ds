import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/child-adult-healthcare-quality/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/child-adult-healthcare-quality/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'State Rate'

# Drop rows where target is NaN if any
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y_train = train_df[target_col].values

X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Feature classification
cat_cols = [
    'State', 'Domain', 'Reporting Program', 'Measure Abbreviation',
    'Measure Type', 'Population', 'Methodology', 
    'Rate Used in Calculating State Mean and Median'
]
text_cols = ['Measure Name', 'Rate Definition', 'Source', 'State Specific Comments']
num_cols = ['Number of States Reporting', 'Core Set Year']

# Clean and prepare CatBoost dataset
cb_train = X_train.copy()
cb_test = X_test.copy()

for col in cat_cols:
    cb_train[col] = cb_train[col].fillna('Missing').astype(str)
    cb_test[col] = cb_test[col].fillna('Missing').astype(str)

for col in text_cols:
    cb_train[col] = cb_train[col].fillna('').astype(str)
    cb_test[col] = cb_test[col].fillna('').astype(str)

for col in num_cols:
    median_val = cb_train[col].median()
    cb_train[col] = cb_train[col].fillna(median_val)
    cb_test[col] = cb_test[col].fillna(median_val)

# Feature engineering for LightGBM
# Text TF-IDF
combined_text_train = cb_train['Measure Name'] + " " + cb_train['Rate Definition'] + " " + cb_train['State Specific Comments']
combined_text_test = cb_test['Measure Name'] + " " + cb_test['Rate Definition'] + " " + cb_test['State Specific Comments']

tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_train = tfidf.fit_transform(combined_text_train).toarray()
tfidf_test = tfidf.transform(combined_text_test).toarray()

# Ordinal encode categoricals
lgb_train = pd.DataFrame(index=cb_train.index)
lgb_test = pd.DataFrame(index=cb_test.index)

for col in cat_cols + ['Measure Name', 'Rate Definition', 'Source']:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    lgb_train[col] = oe.fit_transform(cb_train[[col]]).astype(int)
    lgb_test[col] = oe.transform(cb_test[[col]]).astype(int)

for col in num_cols:
    lgb_train[col] = cb_train[col]
    lgb_test[col] = cb_test[col]

for i in range(tfidf_train.shape[1]):
    lgb_train[f'tfidf_{i}'] = tfidf_train[:, i]
    lgb_test[f'tfidf_{i}'] = tfidf_test[:, i]

# Cross-validation and ensemble training
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

cb_test_preds = np.zeros(len(cb_test))
lgb_test_preds = np.zeros(len(lgb_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(cb_train, y_train)):
    # CatBoost
    cb_X_tr, cb_y_tr = cb_train.iloc[train_idx], y_train[train_idx]
    cb_X_va, cb_y_va = cb_train.iloc[val_idx], y_train[val_idx]
    
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        verbose=False,
        cat_features=cat_cols,
        text_features=text_cols
    )
    cb_model.fit(cb_X_tr, cb_y_tr, eval_set=(cb_X_va, cb_y_va), early_stopping_rounds=50, verbose=False)
    cb_test_preds += cb_model.predict(cb_test) / n_splits

    # LightGBM
    lgb_X_tr, lgb_y_tr = lgb_train.iloc[train_idx], y_train[train_idx]
    lgb_X_va, lgb_y_va = lgb_train.iloc[val_idx], y_train[val_idx]
    
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        lgb_X_tr, lgb_y_tr,
        eval_set=[(lgb_X_va, lgb_y_va)],
        callbacks=[]
    )
    lgb_test_preds += lgb_model.predict(lgb_test) / n_splits

# Blend predictions
final_preds = 0.7 * cb_test_preds + 0.3 * lgb_test_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'State Rate': final_preds
})

output_df.to_csv('predictions.csv', index=False)