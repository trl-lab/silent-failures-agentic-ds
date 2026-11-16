import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-shortages/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-shortages/data/test.parquet')

def extract_features(df):
    """Extract features from raw text columns."""
    features = pd.DataFrame(index=df.index)
    
    # Date features
    dates = pd.to_datetime(df['initial_posting_date'], format='%m/%d/%Y', errors='coerce')
    features['year'] = dates.dt.year.fillna(0).astype(int)
    features['month'] = dates.dt.month.fillna(0).astype(int)
    features['day'] = dates.dt.day.fillna(0).astype(int)
    
    # NDC features
    ndc = df['package_ndc'].fillna('')
    features['ndc_manufacturer'] = ndc.str[:5].astype(str)
    features['ndc_product'] = ndc.str[6:10].astype(str)
    features['ndc_package'] = ndc.str[11:14].astype(str)
    
    # Dosage form
    features['dosage_form'] = df['dosage_form'].fillna('Unknown').astype(str)
    
    # Therapeutic category - take first category
    features['therapeutic_category'] = df['therapeutic_category'].fillna('Unknown').str.split(',').str[0].str.strip().astype(str)
    features['n_therapeutic_categories'] = df['therapeutic_category'].fillna('').str.count(',').astype(int) + 1
    
    # Company name
    features['company_name'] = df['company_name'].fillna('Unknown').astype(str)
    
    # Generic name - extract key drug name
    features['generic_name_first'] = df['generic_name'].fillna('Unknown').str.split().str[0].astype(str)
    features['generic_name_first2'] = df['generic_name'].fillna('Unknown').str.split().str[:2].str.join(' ').astype(str)
    
    # OpenFDA features
    openfda = df['openfda'].fillna('')
    
    # Extract route
    route_match = openfda.str.extract(r'route:\s*([^,]+)')
    features['route'] = route_match[0].fillna('Unknown').str.strip().astype(str)
    
    # Extract product type
    ptype_match = openfda.str.extract(r'product_type:\s*([^,]+)')
    features['product_type'] = ptype_match[0].fillna('Unknown').str.strip().astype(str)
    
    # Extract substance name
    substance_match = openfda.str.extract(r'substance_name:\s*([^,]+)')
    features['substance_name'] = substance_match[0].fillna('Unknown').str.strip().astype(str)
    
    # Extract brand name
    brand_match = openfda.str.extract(r'brand_name:\s*([^,]+)')
    features['brand_name'] = brand_match[0].fillna('Unknown').str.strip().astype(str)
    
    # Extract manufacturer from openfda
    mfr_match = openfda.str.extract(r'manufacturer_name:\s*([^,]+)')
    features['openfda_manufacturer'] = mfr_match[0].fillna('Unknown').str.strip().astype(str)
    
    # Number of package NDCs
    pkg_ndc_match = openfda.str.extract(r'package_ndc:\s*([^,]+(?:,\s*[^,]+)*)')
    features['n_package_ndcs'] = pkg_ndc_match[0].fillna('').str.count(',').astype(int) + 1
    
    # Number of rxcuis
    rxcui_match = openfda.str.extract(r'rxcui:\s*([^,]+(?:,\s*[^,]+)*)')
    features['n_rxcuis'] = rxcui_match[0].fillna('').str.count(',').astype(int) + 1
    
    # Number of spl_ids
    spl_match = openfda.str.extract(r'spl_id:\s*([^,]+(?:,\s*[^,]+)*)')
    features['n_spl_ids'] = spl_match[0].fillna('').str.count(',').astype(int) + 1
    
    # Presentation features
    presentation = df['presentation'].fillna('')
    features['presentation_length'] = presentation.str.len().astype(int)
    
    # Extract dosage from presentation
    dosage_match = presentation.str.extract(r'(\d+\.?\d*)\s*(mg|mcg|g|ml|mL|IU|units?)')
    features['dosage_value'] = pd.to_numeric(dosage_match[0], errors='coerce').fillna(0).astype(float)
    features['dosage_unit'] = dosage_match[1].fillna('Unknown').astype(str)
    
    # Contact info - extract area code
    contact = df['contact_info'].fillna('')
    features['contact_prefix'] = contact.str.extract(r'(\d{3})')[0].fillna('Unknown').astype(str)
    
    return features

# Extract features
X_train = extract_features(train_df)
X_test = extract_features(test_df)
y_train = train_df['availability'].astype(str)

# Encode target
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train)

# Encode all string columns as integers
str_cols = X_train.select_dtypes(include=['object']).columns.tolist()
for col in str_cols:
    le = LabelEncoder()
    all_values = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(all_values)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Fill any remaining NaN
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

# Define models
models = {
    'LightGBM': LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        random_state=42,
        verbose=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0
    )
}

# Train and evaluate on validation set
val_predictions = []

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"{name} Validation Accuracy: {val_acc:.4f}")
    val_predictions.append(model.predict_proba(X_val))

# Ensemble on validation
val_proba_avg = np.mean(val_predictions, axis=0)
val_pred_ensemble = np.argmax(val_proba_avg, axis=1)
val_acc_ensemble = accuracy_score(y_val, val_pred_ensemble)
print(f"Ensemble Validation Accuracy: {val_acc_ensemble:.4f}")

# Retrain on full training data and predict test
test_predictions = []
for name, model in models.items():
    model.fit(X_train, y_train_encoded)
    test_predictions.append(model.predict_proba(X_test))

# Ensemble prediction
test_proba_avg = np.mean(test_predictions, axis=0)
test_pred = np.argmax(test_proba_avg, axis=1)
test_pred_labels = le_target.inverse_transform(test_pred)

# Save predictions
submission_df = pd.DataFrame({
    'id': range(len(test_pred_labels)),
    'availability': test_pred_labels
})
submission_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {submission_df.shape}")
print(submission_df.head())