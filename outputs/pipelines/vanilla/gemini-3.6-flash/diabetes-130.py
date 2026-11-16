import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/diabetes-130/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/diabetes-130/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'readmitted'
y_train_raw = train_df[target_col]
train_df = train_df.drop(columns=[target_col])

# Encode target variable
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

n_train = len(train_df)
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Replace missing indicator '?' with NaN
df = df.replace('?', np.nan)

# Feature engineering
age_map = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}
if 'age' in df.columns:
    df['age_num'] = df['age'].map(age_map).fillna(-1)

visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
existing_visit_cols = [c for c in visit_cols if c in df.columns]
if existing_visit_cols:
    df['total_visits'] = df[existing_visit_cols].sum(axis=1)

med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone']
existing_med_cols = [c for c in med_cols if c in df.columns]
if existing_med_cols:
    df['num_med_active'] = 0
    df['num_med_changed'] = 0
    for col in existing_med_cols:
        col_str = df[col].astype(str).str.lower()
        df['num_med_active'] += col_str.isin(['up', 'down', 'steady']).astype(int)
        df['num_med_changed'] += col_str.isin(['up', 'down']).astype(int)

if 'patient_nbr' in df.columns:
    patient_counts = df['patient_nbr'].value_counts()
    df['patient_encounter_count'] = df['patient_nbr'].map(patient_counts)

if 'encounter_id' in df.columns:
    df = df.drop(columns=['encounter_id'])

# Convert ID columns to categorical strings
categorical_ids = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
for col in categorical_ids:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Categorical processing
cat_cols = []
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        cat_cols.append(col)
        df[col] = df[col].fillna('Missing').astype(str)

# Separate train and test sets
X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

# Cast categoricals to category dtype for LightGBM
for c in cat_cols:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = X_test[c].astype('category')

num_classes = len(le.classes_)
test_preds = np.zeros((len(X_test), num_classes))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_params = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'random_state': 42,
    'n_estimators': 500,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
    
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )
    test_preds += model.predict_proba(X_test) / skf.n_splits

final_preds = np.argmax(test_preds, axis=1)
final_labels = le.inverse_transform(final_preds)

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'readmitted': final_labels
})
sub.to_csv('predictions.csv', index=False)