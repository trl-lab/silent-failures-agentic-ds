import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-drugsfda/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-drugsfda/data/test.parquet')

# Encode target
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df['marketing_status'])

# Feature engineering from submissions text
def extract_submission_features(submissions_text):
    if pd.isna(submissions_text):
        return {
            'n_submissions': 0,
            'n_orig': 0,
            'n_suppl': 0,
            'n_ap': 0,
            'n_ta': 0,
            'n_labeling': 0,
            'n_manuf': 0,
            'n_unknown': 0,
            'n_review_priority': 0,
            'n_application_docs': 0,
            'earliest_date': 0,
            'latest_date': 0,
            'date_range': 0,
            'text_length': 0
        }
    
    text = str(submissions_text)
    
    n_submissions = len(re.findall(r'submission_type:', text))
    n_orig = len(re.findall(r'submission_type:\s*ORIG', text))
    n_suppl = len(re.findall(r'submission_type:\s*SUPPL', text))
    n_ap = len(re.findall(r'submission_status:\s*AP', text))
    n_ta = len(re.findall(r'submission_status:\s*TA', text))
    n_labeling = len(re.findall(r'submission_class_code:\s*LABELING', text))
    n_manuf = len(re.findall(r'submission_class_code:\s*MANUF', text))
    n_unknown = len(re.findall(r'submission_class_code:\s*UNKNOWN', text))
    n_review_priority = len(re.findall(r'review_priority:', text))
    n_application_docs = len(re.findall(r'application_docs:', text))
    
    dates = re.findall(r'submission_status_date:\s*(\d{8})', text)
    if dates:
        dates = [int(d) for d in dates]
        earliest_date = min(dates)
        latest_date = max(dates)
        date_range = latest_date - earliest_date
    else:
        earliest_date = 0
        latest_date = 0
        date_range = 0
    
    return {
        'n_submissions': n_submissions,
        'n_orig': n_orig,
        'n_suppl': n_suppl,
        'n_ap': n_ap,
        'n_ta': n_ta,
        'n_labeling': n_labeling,
        'n_manuf': n_manuf,
        'n_unknown': n_unknown,
        'n_review_priority': n_review_priority,
        'n_application_docs': n_application_docs,
        'earliest_date': earliest_date,
        'latest_date': latest_date,
        'date_range': date_range,
        'text_length': len(text)
    }

print("Extracting submission features...")
train_subm_feats = pd.json_normalize(train_df['submissions'].apply(extract_submission_features))
test_subm_feats = pd.json_normalize(test_df['submissions'].apply(extract_submission_features))

# Additional features
def extract_app_number(app_num):
    if pd.isna(app_num):
        return 0
    match = re.search(r'(\d+)', str(app_num))
    return int(match.group(1)) if match else 0

def extract_reference_drug(ref):
    if pd.isna(ref):
        return 0
    return 0 if str(ref).strip().lower() == 'no' else 1

def extract_n_strengths(strengths):
    if pd.isna(strengths):
        return 0
    return len(str(strengths).split(','))

def extract_dosage_form_count(df):
    return df['dosage_form'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

train_df['app_number'] = train_df['application_number'].apply(extract_app_number)
test_df['app_number'] = test_df['application_number'].apply(extract_app_number)

train_df['has_reference_drug'] = train_df['reference_drug'].apply(extract_reference_drug)
test_df['has_reference_drug'] = test_df['reference_drug'].apply(extract_reference_drug)

train_df['n_strengths'] = train_df['strengths'].apply(extract_n_strengths)
test_df['n_strengths'] = test_df['strengths'].apply(extract_n_strengths)

train_df['n_dosage_forms'] = extract_dosage_form_count(train_df)
test_df['n_dosage_forms'] = extract_dosage_form_count(test_df)

train_df['sponsor_name_len'] = train_df['sponsor_name'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
test_df['sponsor_name_len'] = test_df['sponsor_name'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

train_df['brand_name_len'] = train_df['brand_name'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
test_df['brand_name_len'] = test_df['brand_name'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

# Combine features
train_features = pd.concat([train_df[['app_number', 'has_reference_drug', 'n_strengths', 'n_dosage_forms', 'sponsor_name_len', 'brand_name_len']], train_subm_feats], axis=1)
test_features = pd.concat([test_df[['app_number', 'has_reference_drug', 'n_strengths', 'n_dosage_forms', 'sponsor_name_len', 'brand_name_len']], test_subm_feats], axis=1)

# Add categorical features as label-encoded
cat_cols = ['application_number', 'sponsor_name', 'reference_drug', 'strengths', 'brand_name', 'dosage_form']

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
    le.fit(combined)
    train_features[col] = le.transform(train_df[col].astype(str))
    test_features[col] = le.transform(test_df[col].astype(str))

# Fill any remaining NaN
train_features = train_features.fillna(0)
test_features = test_features.fillna(0)

print(f"Training features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")

# Validation split
X_train, X_val, y_train_split, y_val_split = train_test_split(
    train_features, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Train models
print("Training LightGBM...")
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_train, y_train_split)
lgbm_val_auc = roc_auc_score(y_val_split, lgbm_model.predict_proba(X_val)[:, 1])
print(f"LightGBM Val AUC: {lgbm_val_auc:.4f}")

print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=5,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train_split)
xgb_val_auc = roc_auc_score(y_val_split, xgb_model.predict_proba(X_val)[:, 1])
print(f"XGBoost Val AUC: {xgb_val_auc:.4f}")

print("Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0
)
cat_model.fit(X_train, y_train_split)
cat_val_auc = roc_auc_score(y_val_split, cat_model.predict_proba(X_val)[:, 1])
print(f"CatBoost Val AUC: {cat_val_auc:.4f}")

# Retrain on full training data
print("Retraining on full training data...")
lgbm_model.fit(train_features, y_train)
xgb_model.fit(train_features, y_train)
cat_model.fit(train_features, y_train)

# Ensemble predictions
lgbm_pred = lgbm_model.predict_proba(test_features)[:, 1]
xgb_pred = xgb_model.predict_proba(test_features)[:, 1]
cat_pred = cat_model.predict_proba(test_features)[:, 1]

# Weighted ensemble based on validation AUC
weights = np.array([lgbm_val_auc, xgb_val_auc, cat_val_auc])
weights = weights / weights.sum()
print(f"Ensemble weights: LGBM={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}")

ensemble_pred = weights[0] * lgbm_pred + weights[1] * xgb_pred + weights[2] * cat_pred

# Save predictions as probability values (ROC AUC expects scores)
output_df = pd.DataFrame({
    'id': range(len(ensemble_pred)),
    'marketing_status': ensemble_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")
print(f"Prediction stats: mean={ensemble_pred.mean():.4f}, min={ensemble_pred.min():.4f}, max={ensemble_pred.max():.4f}")