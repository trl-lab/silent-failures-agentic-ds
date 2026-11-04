import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/community-banking_wide/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/community-banking_wide/data/test.parquet')

# Identify columns to drop (identifiers, names, addresses - not useful for prediction)
drop_cols = ['Id', 'NAMEHCR', 'NAMEFULL', 'ADDRESS', 'CITY', 'STALP', 'ZIP', 'STCNTY', 'CERT']

# Parse string columns that contain numeric data
def parse_numeric_string(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace('%', '').replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

for col in ['FORASSET', 'CORERATIO', 'MAX_DEPOSITS', 'OFFICE_COUNT']:
    if col in train_df.columns:
        train_df[col] = train_df[col].apply(parse_numeric_string)
    if col in test_df.columns:
        test_df[col] = test_df[col].apply(parse_numeric_string)

# Separate target
y_train = train_df['LOANTOASSET'].copy()
train_df = train_df.drop('LOANTOASSET', axis=1)

# Drop identifier columns
train_df = train_df.drop([c for c in drop_cols if c in train_df.columns], axis=1)
test_df = test_df.drop([c for c in drop_cols if c in test_df.columns], axis=1)

# Encode categorical columns
categorical_cols = ['BUSINESS_LINE', 'STALP']
label_encoders = {}
for col in categorical_cols:
    if col in train_df.columns:
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le

# Ensure all columns are numeric
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

# Align columns
common_cols = [c for c in train_df.columns if c in test_df.columns]
X_train = train_df[common_cols]
X_test = test_df[common_cols]

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train models and evaluate
models = {
    'lgbm': LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
    'xgb': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='rmse'),
    'cat': CatBoostRegressor(iterations=500, learning_rate=0.05, random_state=42, verbose=0)
}

val_preds = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    print(f"{name}: Val RMSE = {rmse:.4f}")
    val_preds.append(val_pred)

# Ensemble validation
ensemble_val = np.mean(val_preds, axis=0)
ensemble_rmse = np.sqrt(np.mean((ensemble_val - y_val) ** 2))
print(f"Ensemble: Val RMSE = {ensemble_rmse:.4f}")

# Retrain on full training data and predict test
test_preds = []
for name, model in models.items():
    model.fit(X_train, y_train)
    test_preds.append(model.predict(X_test))

final_pred = np.mean(test_preds, axis=0)

# Save predictions
output_df = pd.DataFrame({'id': range(len(final_pred)), 'LOANTOASSET': final_pred})
output_df.to_csv('predictions.csv', index=False)