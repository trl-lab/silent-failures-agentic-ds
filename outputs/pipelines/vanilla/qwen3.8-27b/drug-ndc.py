import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-ndc/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-ndc/data/test.parquet')

# Separate target
y_train = train_df['dea_schedule'].copy()
X_train = train_df.drop('dea_schedule', axis=1)
X_test = test_df.copy()

# Drop high-cardinality ID columns that won't generalize
drop_cols = ['product_ndc', 'spl_id', 'product_id', 'application_number', 'packaging', 'openfda', 'listing_expiration_date', 'marketing_start_date', 'marketing_end_date']
drop_cols = [c for c in drop_cols if c in X_train.columns]
X_train = X_train.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

# Feature engineering from active_ingredients
def extract_ingredients(text):
    if pd.isna(text):
        return '', 0
    names = []
    parts = str(text).split(', ')
    for part in parts:
        if 'name:' in part:
            name_part = part.split('name:')[1].strip()
            names.append(name_part)
    return ' | '.join(names), len(names)

X_train['ingredient_names'] = X_train['active_ingredients'].apply(lambda x: extract_ingredients(x)[0])
X_train['n_ingredients'] = X_train['active_ingredients'].apply(lambda x: extract_ingredients(x)[1])
X_test['ingredient_names'] = X_test['active_ingredients'].apply(lambda x: extract_ingredients(x)[0])
X_test['n_ingredients'] = X_test['active_ingredients'].apply(lambda x: extract_ingredients(x)[1])

X_train['pharm_class_str'] = X_train['pharm_class'].fillna('MISSING').astype(str)
X_test['pharm_class_str'] = X_test['pharm_class'].fillna('MISSING').astype(str)

# Drop original text columns that we've processed
X_train = X_train.drop(columns=['active_ingredients', 'pharm_class'])
X_test = X_test.drop(columns=['active_ingredients', 'pharm_class'])

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()

# Convert bool to int
for col in bool_cols:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# Fill missing values and label encode all categorical columns
for col in categorical_cols:
    X_train[col] = X_train[col].fillna('MISSING').astype(str)
    X_test[col] = X_test[col].fillna('MISSING').astype(str)
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Ensure all columns are numeric
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

# Label encode the target
target_le = LabelEncoder()
y_train_encoded = target_le.fit_transform(y_train)
classes = target_le.classes_

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded)

# Model 1: LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_tr, y_tr)
lgbm_val_pred = lgbm_model.predict(X_val)
lgbm_val_acc = (lgbm_val_pred == y_val).mean()
print(f"LightGBM validation accuracy: {lgbm_val_acc:.4f}")

# Model 2: XGBoost
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_tr, y_tr)
xgb_val_pred = xgb_model.predict(X_val)
xgb_val_acc = (xgb_val_pred == y_val).mean()
print(f"XGBoost validation accuracy: {xgb_val_acc:.4f}")

# Model 3: CatBoost
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0
)
cat_model.fit(X_tr, y_tr)
cat_val_pred = cat_model.predict(X_val)
cat_val_acc = (cat_val_pred == y_val).mean()
print(f"CatBoost validation accuracy: {cat_val_acc:.4f}")

# Retrain all models on full training data
lgbm_model_full = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    random_state=42,
    verbose=-1
)
lgbm_model_full.fit(X_train, y_train_encoded)

xgb_model_full = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model_full.fit(X_train, y_train_encoded)

cat_model_full = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0
)
cat_model_full.fit(X_train, y_train_encoded)

# Ensemble: use probability averaging
lgbm_proba = lgbm_model_full.predict_proba(X_test)
xgb_proba = xgb_model_full.predict_proba(X_test)
cat_proba = cat_model_full.predict_proba(X_test)

# Align columns to the same class order
n_classes = len(classes)
lgbm_proba_aligned = np.zeros((len(X_test), n_classes))
for i, c in enumerate(lgbm_model_full.classes_):
    lgbm_proba_aligned[:, int(c)] = lgbm_proba[:, i]

xgb_proba_aligned = np.zeros((len(X_test), n_classes))
for i, c in enumerate(xgb_model_full.classes_):
    xgb_proba_aligned[:, int(c)] = xgb_proba[:, i]

cat_proba_aligned = np.zeros((len(X_test), n_classes))
for i, c in enumerate(cat_model_full.classes_):
    cat_proba_aligned[:, int(c)] = cat_proba[:, i]

# Weighted ensemble based on validation accuracy
weights = np.array([lgbm_val_acc, xgb_val_acc, cat_val_acc])
weights = weights / weights.sum()
print(f"Ensemble weights: LGBM={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}")

ensemble_proba = weights[0] * lgbm_proba_aligned + weights[1] * xgb_proba_aligned + weights[2] * cat_proba_aligned
ensemble_pred_encoded = np.argmax(ensemble_proba, axis=1)
ensemble_pred = target_le.inverse_transform(ensemble_pred_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(ensemble_pred)),
    'dea_schedule': ensemble_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")
print(f"Class distribution:\n{output_df['dea_schedule'].value_counts()}")