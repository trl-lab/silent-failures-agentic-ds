import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/cosmetic-event/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/cosmetic-event/data/test.parquet')

# Drop ID columns
id_cols = ['report_number', 'legacy_report_id']
for col in id_cols:
    if col in train_df.columns:
        train_df.drop(col, axis=1, inplace=True)
    if col in test_df.columns:
        test_df.drop(col, axis=1, inplace=True)

# Separate target
y_train = train_df['outcomes']
X_train = train_df.drop('outcomes', axis=1)
X_test = test_df.copy()

# Encode target
target_le = LabelEncoder()
y_train_encoded = target_le.fit_transform(y_train)

# Feature engineering
def parse_age(df):
    age = pd.to_numeric(df['age'].replace('None', np.nan), errors='coerce')
    return age

def parse_date_year(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    dates = pd.to_numeric(df[col].str.replace('-', '').str.replace('/', ''), errors='coerce')
    years = (dates // 10000).astype(float)
    mask = years.isna()
    if mask.any():
        try:
            parsed = pd.to_datetime(df[col][mask], errors='coerce', format='mixed')
            years[mask] = parsed.dt.year
        except:
            pass
    return years

def parse_date_month(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    dates = pd.to_numeric(df[col].str.replace('-', '').str.replace('/', ''), errors='coerce')
    months = ((dates % 10000) // 100).astype(float)
    mask = months.isna()
    if mask.any():
        try:
            parsed = pd.to_datetime(df[col][mask], errors='coerce', format='mixed')
            months[mask] = parsed.dt.month
        except:
            pass
    return months

# Parse age
X_train['age_num'] = parse_age(X_train)
X_test['age_num'] = parse_age(X_test)
X_train.drop('age', axis=1, inplace=True)
X_test.drop('age', axis=1, inplace=True)

# Parse dates
for col in ['event_date', 'latest_received_date', 'initial_received_date']:
    X_train[f'{col}_year'] = parse_date_year(X_train, col)
    X_test[f'{col}_year'] = parse_date_year(X_test, col)
    X_train[f'{col}_month'] = parse_date_month(X_train, col)
    X_test[f'{col}_month'] = parse_date_month(X_test, col)
    X_train.drop(col, axis=1, inplace=True)
    X_test.drop(col, axis=1, inplace=True)

# Parse report_version
X_train['report_version_num'] = pd.to_numeric(X_train['report_version'], errors='coerce')
X_test['report_version_num'] = pd.to_numeric(X_test['report_version'], errors='coerce')
X_train.drop('report_version', axis=1, inplace=True)
X_test.drop('report_version', axis=1, inplace=True)

# Encode categorical columns
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Fill remaining NaN
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

# Validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded)

# Train models and evaluate
models = {
    'LightGBM': LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1),
    'XGBoost': XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(iterations=500, learning_rate=0.05, random_state=42, verbose=0)
}

val_predictions = []
test_predictions = []

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"{name} validation accuracy: {acc:.4f}")
    val_predictions.append(model.predict_proba(X_val))
    test_predictions.append(model.predict_proba(X_test))

# Ensemble: average probabilities
val_ensemble = np.mean(val_predictions, axis=0)
test_ensemble = np.mean(test_predictions, axis=0)

# Convert ensemble probabilities to class predictions
val_pred_ensemble = val_ensemble.argmax(axis=1)
test_pred_ensemble = test_ensemble.argmax(axis=1)

print(f"Ensemble validation accuracy: {accuracy_score(y_val, val_pred_ensemble):.4f}")

# Decode predictions back to strings
test_pred_labels = target_le.inverse_transform(test_pred_ensemble)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_pred_labels)),
    'outcomes': test_pred_labels
})
output_df.to_csv('predictions.csv', index=False)