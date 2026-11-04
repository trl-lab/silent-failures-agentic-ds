import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/food-event/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/food-event/data/test.parquet')

def extract_features(df):
    features = pd.DataFrame(index=df.index)
    
    # Age: convert to numeric
    features['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Age unit
    features['age_unit'] = df['age_unit'].fillna('unknown')
    
    # Gender
    features['gender'] = df['gender'].fillna('unknown')
    
    # Date created: extract year, month, day
    date_created = pd.to_datetime(df['date_created'], format='%Y%m%d', errors='coerce')
    features['date_created_year'] = date_created.dt.year
    features['date_created_month'] = date_created.dt.month
    features['date_created_day'] = date_created.dt.day
    
    # Date started: extract year, month, day
    date_started = pd.to_datetime(df['date_started'], format='%Y%m%d', errors='coerce')
    features['date_started_year'] = date_started.dt.year
    features['date_started_month'] = date_started.dt.month
    features['date_started_day'] = date_started.dt.day
    
    # Days between date_started and date_created
    features['days_between'] = (date_created - date_started).dt.days
    
    # Reactions: count and specific features
    reactions = df['reactions'].fillna('')
    features['num_reactions'] = reactions.apply(lambda x: len([r for r in x.split(',') if r.strip()]))
    
    # Extract specific reaction keywords
    reaction_keywords = ['NAUSEA', 'VOMITING', 'DIARRHOEA', 'HEADACHE', 'DIZZINESS', 
                         'FATIGUE', 'PAIN', 'FEVER', 'RASH', 'ALLERGY', 'HYPERTENSION',
                         'TACHYCARDIA', 'CHEST', 'ABDOMINAL', 'MUSCLE', 'SLEEP',
                         'ANXIETY', 'DEPRESSION', 'WEIGHT', 'BLOOD', 'LIVER', 'KIDNEY',
                         'CARDIAC', 'RESPIRATORY', 'NEUROLOGICAL', 'GASTROINTESTINAL',
                         'SKIN', 'EYE', 'EAR', 'THROAT', 'MOUTH', 'BONE', 'JOINT']
    for kw in reaction_keywords:
        features[f'react_{kw}'] = reactions.str.contains(kw, case=False, na=False).astype(int)
    
    # Products: extract features
    products = df['products'].fillna('')
    features['num_products'] = products.str.count('role:').fillna(0).astype(int)
    features['num_suspect'] = products.str.count('role: SUSPECT').fillna(0).astype(int)
    features['num_concomitant'] = products.str.count('role: CONCOMITANT').fillna(0).astype(int)
    
    # Industry codes
    for i in range(1, 30):
        features[f'has_industry_{i}'] = products.str.contains(f'industry_code: {i}', na=False).astype(int)
    
    # Product name length
    features['product_text_length'] = products.str.len().fillna(0)
    
    # Reactions text length
    features['reactions_text_length'] = reactions.str.len().fillna(0)
    
    return features

# Extract features
X_train = extract_features(train_df)
X_test = extract_features(test_df)
y_train = train_df['outcomes']

# Encode categorical features
categorical_cols = ['age_unit', 'gender']
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Encode target
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train)

# Ensure all columns are numeric
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Fill missing values
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

# Train models
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_tr, y_tr)
lgbm_val_pred = lgbm_model.predict(X_val)
lgbm_val_acc = accuracy_score(y_val, lgbm_val_pred)
print(f"LightGBM Validation Accuracy: {lgbm_val_acc:.4f}")

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)
xgb_model.fit(X_tr, y_tr)
xgb_val_pred = xgb_model.predict(X_val)
xgb_val_acc = accuracy_score(y_val, xgb_val_pred)
print(f"XGBoost Validation Accuracy: {xgb_val_acc:.4f}")

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0
)
cat_model.fit(X_tr, y_tr)
cat_val_pred = cat_model.predict(X_val)
cat_val_acc = accuracy_score(y_val, cat_val_pred)
print(f"CatBoost Validation Accuracy: {cat_val_acc:.4f}")

# Ensemble: use soft voting with probability averaging
lgbm_proba = lgbm_model.predict_proba(X_test)
xgb_proba = xgb_model.predict_proba(X_test)
cat_proba = cat_model.predict_proba(X_test)

# Average probabilities
avg_proba = (lgbm_proba + xgb_proba + cat_proba) / 3.0
final_pred_indices = np.argmax(avg_proba, axis=1)
final_pred = le_target.inverse_transform(final_pred_indices)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'outcomes': final_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")