import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier

# 1. Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/animalandveterinary-event/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/animalandveterinary-event/data/test.parquet')

# 2. Extract target
y_train = (
    train_df['serious_ae']
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(['true', '1', 't', 'yes'])
    .astype(int)
    .values
)

# 3. Feature engineering
def preprocess_tabular(df):
    feat = pd.DataFrame(index=df.index)
    
    # Numeric extractions
    feat['num_affected'] = pd.to_numeric(df['number_of_animals_affected'], errors='coerce')
    feat['num_treated'] = pd.to_numeric(df['number_of_animals_treated'], errors='coerce')
    
    # Age extraction
    age_val = df['age'].astype(str).str.extract(r'min:\s*([\d\.]+)')[0].astype(float)
    age_unit = df['age'].astype(str).str.extract(r'unit:\s*([A-Za-z]+)')[0].fillna('Unknown')
    # Convert age to approximate years
    multiplier = age_unit.map({'Year': 1.0, 'Month': 1.0/12.0, 'Week': 1.0/52.0, 'Day': 1.0/365.0, 'Hour': 1.0/8760.0}).fillna(1.0)
    feat['age_years'] = age_val * multiplier
    
    # Weight extraction
    weight_val = df['weight'].astype(str).str.extract(r'min:\s*([\d\.]+)')[0].astype(float)
    weight_unit = df['weight'].astype(str).str.extract(r'unit:\s*([A-Za-z]+)')[0].fillna('Unknown')
    w_multiplier = weight_unit.map({'Kilogram': 1.0, 'Gram': 0.001, 'Pound': 0.453592, 'Ounce': 0.0283495}).fillna(1.0)
    feat['weight_kg'] = weight_val * w_multiplier
    
    # Date extractions
    feat['receive_date_num'] = pd.to_numeric(df['original_receive_date'].astype(str).str.slice(0, 8), errors='coerce')
    feat['onset_date_num'] = pd.to_numeric(df['onset_date'].astype(str).str.slice(0, 8), errors='coerce')
    feat['date_diff'] = feat['receive_date_num'] - feat['onset_date_num']
    
    # Text length features
    feat['reaction_len'] = df['reaction'].astype(str).str.len()
    feat['drug_len'] = df['drug'].astype(str).str.len()
    feat['num_reactions'] = df['reaction'].astype(str).str.count('veddra_term_code')
    feat['num_drugs'] = df['drug'].astype(str).str.count('active_ingredients')
    
    # Breed crossbred flag
    feat['is_crossbred'] = df['breed'].astype(str).str.extract(r'is_crossbred:\s*([a-zA-Z]+)')[0].str.lower().map({'true': 1, 'false': 0}).fillna(-1)
    
    return feat

X_train_num = preprocess_tabular(train_df)
X_test_num = preprocess_tabular(test_df)

# Categorical features
cat_cols = [
    'primary_reporter', 'secondary_reporter', 'species', 'gender',
    'reproductive_status', 'female_animal_physiological_status',
    'type_of_information', 'condition'
]

X_train_cat = pd.DataFrame(index=train_df.index)
X_test_cat = pd.DataFrame(index=test_df.index)

for col in cat_cols:
    train_c = train_df[col].astype(str).fillna('missing')
    test_c = test_df[col].astype(str).fillna('missing')
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat[col] = oe.fit_transform(train_c.values.reshape(-1, 1)).ravel()
    X_test_cat[col] = oe.transform(test_c.values.reshape(-1, 1)).ravel()

# Text features via TF-IDF
vec_reaction = TfidfVectorizer(max_features=600, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
X_train_rx = vec_reaction.fit_transform(train_df['reaction'].astype(str).fillna(''))
X_test_rx = vec_reaction.transform(test_df['reaction'].astype(str).fillna(''))

vec_drug = TfidfVectorizer(max_features=400, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
X_train_drug = vec_drug.fit_transform(train_df['drug'].astype(str).fillna(''))
X_test_drug = vec_drug.transform(test_df['drug'].astype(str).fillna(''))

vec_breed = TfidfVectorizer(max_features=100, token_pattern=r'(?u)\b\w+\b')
X_train_breed = vec_breed.fit_transform(train_df['breed'].astype(str).fillna(''))
X_test_breed = vec_breed.transform(test_df['breed'].astype(str).fillna(''))

# Combine all features into sparse matrices
X_train_dense = pd.concat([X_train_num, X_train_cat], axis=1).values
X_test_dense = pd.concat([X_test_num, X_test_cat], axis=1).values

X_train_all = hstack([X_train_dense, X_train_rx, X_train_drug, X_train_breed]).tocsr().astype(np.float32)
X_test_all = hstack([X_test_dense, X_test_rx, X_test_drug, X_test_breed]).tocsr().astype(np.float32)

# 4. K-Fold Training and Ensembling
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_lgb = np.zeros(test_df.shape[0])
test_preds_cat = np.zeros(test_df.shape[0])

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_all, y_train)):
    X_tr, y_tr = X_train_all[train_idx], y_train[train_idx]
    X_va, y_va = X_train_all[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )
    test_preds_lgb += lgb_model.predict_proba(X_test_all)[:, 1] / cv.n_splits
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=40,
        verbose=False
    )
    test_preds_cat += cat_model.predict_proba(X_test_all)[:, 1] / cv.n_splits

# Ensemble predictions
final_predictions = 0.5 * test_preds_lgb + 0.5 * test_preds_cat

# 5. Output predictions
submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'serious_ae': final_predictions
})
submission.to_csv('predictions.csv', index=False)