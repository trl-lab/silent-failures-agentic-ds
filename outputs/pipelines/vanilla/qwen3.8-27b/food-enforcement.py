import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/food-enforcement/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/food-enforcement/data/test.parquet')

# Separate target
y_train = train_df['classification']
X_train_full = train_df.drop('classification', axis=1)
X_test = test_df.drop('classification', axis=1) if 'classification' in test_df.columns else test_df

# Encode target
target_le = LabelEncoder()
y_train_encoded = target_le.fit_transform(y_train)

# Combine for consistent feature engineering
combined = pd.concat([X_train_full, X_test], axis=0, ignore_index=True)

# Feature engineering
def extract_features(df):
    features = pd.DataFrame(index=df.index)
    
    # Categorical features to keep as-is
    cat_cols = ['status', 'state', 'country', 'voluntary_mandated', 'initial_firm_notification', 'distribution_pattern']
    for col in cat_cols:
        if col in df.columns:
            features[col] = df[col].fillna('MISSING')
    
    # Text length features
    text_cols = ['product_description', 'reason_for_recall', 'code_info', 'recalling_firm', 'address_1', 'address_2']
    for col in text_cols:
        if col in df.columns:
            features[f'{col}_len'] = df[col].fillna('').str.len()
            features[f'{col}_word_count'] = df[col].fillna('').str.split().str.len().fillna(0)
    
    # Date features - extract year
    date_cols = ['recall_initiation_date', 'center_classification_date', 'report_date', 'termination_date']
    for col in date_cols:
        if col in df.columns:
            features[f'{col}_year'] = pd.to_numeric(df[col].fillna('').str[:4], errors='coerce').fillna(0)
    
    # Distribution pattern features
    if 'distribution_pattern' in df.columns:
        features['dist_domestic'] = df['distribution_pattern'].fillna('').str.contains('Domestic', case=False).astype(int)
        features['dist_international'] = df['distribution_pattern'].fillna('').str.contains('International', case=False).astype(int)
        features['dist_num_states'] = df['distribution_pattern'].fillna('').str.split(',').str.len().fillna(0)
    
    # Reason for recall - keyword features
    if 'reason_for_recall' in df.columns:
        reason = df['reason_for_recall'].fillna('').str.lower()
        features['reason_allergen'] = reason.str.contains('allergen|undeclared|soy|peanut|milk|egg|wheat|tree nut|fish|shellfish|sesame').astype(int)
        features['reason_salmonella'] = reason.str.contains('salmonella').astype(int)
        features['reason_listeria'] = reason.str.contains('listeria').astype(int)
        features['reason_e_coli'] = reason.str.contains('e.coli|e coli|ecoli').astype(int)
        features['reason_undeclared'] = reason.str.contains('undeclared|failed to declare|lacked').astype(int)
        features['reason_mislabeled'] = reason.str.contains('mislabeled|mislabel|incorrect label|wrong label').astype(int)
        features['reason_contamination'] = reason.str.contains('contaminat').astype(int)
        features['reason_foreign'] = reason.str.contains('foreign|fingertip|hair|glass|metal|plastic').astype(int)
        features['reason_lead'] = reason.str.contains('lead').astype(int)
        features['reason_bacteria'] = reason.str.contains('bacteria|bacterial').astype(int)
        features['reason_packing'] = reason.str.contains('packing|packaging|unsealed|broken seal').astype(int)
    
    # Product description features
    if 'product_description' in df.columns:
        prod = df['product_description'].fillna('').str.lower()
        features['prod_food'] = prod.str.contains('bread|meat|chicken|beef|pork|fish|shrimp|egg|milk|cheese|yogurt|ice cream|candy|chocolate|cookie|cake|pastry|soup|sauce|juice|drink|beverage|wine|beer|spice|herb|vitamin|supplement|formula|infant|baby').astype(int)
        features['prod_pet'] = prod.str.contains('pet|dog|cat|animal').astype(int)
        features['prod_supplement'] = prod.str.contains('supplement|vitamin|mineral|herbal|capsule|tablet').astype(int)
        features['prod_baby'] = prod.str.contains('infant|baby|toddler').astype(int)
    
    # Status feature
    if 'status' in df.columns:
        features['status_terminated'] = (df['status'].fillna('') == 'Terminated').astype(int)
        features['status_open'] = (df['status'].fillna('') == 'Open').astype(int)
    
    # Voluntary vs mandated
    if 'voluntary_mandated' in df.columns:
        features['voluntary'] = df['voluntary_mandated'].fillna('').str.contains('Voluntary', case=False).astype(int)
        features['mandated'] = df['voluntary_mandated'].fillna('').str.contains('Mandated', case=False).astype(int)
    
    # Notification method
    if 'initial_firm_notification' in df.columns:
        notif = df['initial_firm_notification'].fillna('')
        features['notif_phone'] = notif.str.contains('Telephone', case=False).astype(int)
        features['notif_email'] = notif.str.contains('Email', case=False).astype(int)
        features['notif_letter'] = notif.str.contains('Letter', case=False).astype(int)
        features['notif_visit'] = notif.str.contains('Visit', case=False).astype(int)
        features['notif_multiple'] = notif.str.contains('Two or more', case=False).astype(int)
    
    # Recall number year
    if 'recall_number' in df.columns:
        features['recall_year'] = pd.to_numeric(df['recall_number'].fillna('').str.extract(r'(\d{4})')[0], errors='coerce').fillna(0)
    
    # Product quantity - extract numeric
    if 'product_quantity' in df.columns:
        qty = df['product_quantity'].fillna('')
        features['qty_numeric'] = pd.to_numeric(qty.str.extract(r'(\d+)')[0], errors='coerce').fillna(0)
        features['qty_has_cases'] = qty.str.contains('case', case=False).astype(int)
        features['qty_has_lbs'] = qty.str.contains('lb|pound', case=False).astype(int)
        features['qty_has_units'] = qty.str.contains('unit|bottle|can|bag|box|loaf|dozen', case=False).astype(int)
    
    # City length as proxy for city size
    if 'city' in df.columns:
        features['city_len'] = df['city'].fillna('').str.len()
    
    # Postal code - extract first 2 digits (region)
    if 'postal_code' in df.columns:
        features['postal_region'] = df['postal_code'].fillna('').str[:2]
    
    return features

features = extract_features(combined)

# Encode remaining categorical columns
cat_feature_cols = [c for c in features.columns if features[c].dtype == 'object']
for col in cat_feature_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col].astype(str))

# Split back
X_train = features.iloc[:len(X_train_full)].reset_index(drop=True)
X_test = features.iloc[len(X_train_full):].reset_index(drop=True)

# Validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded)

# Train models and evaluate
models = {
    'LightGBM': LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
    'XGBoost': XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(iterations=500, learning_rate=0.05, random_state=42, verbose=0)
}

val_scores = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    acc = (val_pred == y_val).mean()
    val_scores[name] = acc
    print(f"{name}: Validation Accuracy = {acc:.4f}")

# Use best model
best_model_name = max(val_scores, key=val_scores.get)
print(f"\nBest model: {best_model_name} with accuracy {val_scores[best_model_name]:.4f}")

# Retrain best model on full training data and predict test
best_model = models[best_model_name]
best_model.fit(X_train, y_train_encoded)
test_predictions_encoded = best_model.predict(X_test)

# Decode predictions back to original class names
test_predictions = target_le.inverse_transform(test_predictions_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_predictions)),
    'classification': test_predictions
})
output_df.to_csv('predictions.csv', index=False)
print(f"\nPredictions saved. Shape: {output_df.shape}")