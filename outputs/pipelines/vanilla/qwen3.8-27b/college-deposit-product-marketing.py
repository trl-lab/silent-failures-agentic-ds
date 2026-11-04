import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/college-deposit-product-marketing/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/college-deposit-product-marketing/data/test.parquet')

# Define columns to drop (identifiers, URLs, high-cardinality names)
drop_cols = [
    'INSTITUTION_NAME',
    'UNIQUE_ID_(OPEID)',
    'CASH_MANAGEMENT_DISCLOSURE_URL',
    'ACCOUNT_TERM_DISCLOSURE_URL',
    'PARTNERSHIP_AGREEMENT_URL',
    'DATE_ACCESSED_BY_CFPB'
]

# Drop target from features
target = 'TOTAL_ACTIVE_ACCOUNTS_'

# Drop unnecessary columns
train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

# Separate target
y_train = train_df[target].copy()
X_train_full = train_df.drop(columns=[target])
X_test = test_df.copy()

# Replace '.' with NaN for all object columns
for df in [X_train_full, X_test]:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace('.', np.nan)

# Identify numeric and categorical columns
numeric_cols = []
categorical_cols = []

for col in X_train_full.columns:
    if X_train_full[col].dtype in ['float64', 'float32', 'int64', 'int32']:
        numeric_cols.append(col)
    else:
        # Try to convert to numeric
        converted = pd.to_numeric(X_train_full[col], errors='coerce')
        if converted.notna().sum() > X_train_full[col].notna().sum() * 0.5:
            # Mostly numeric, convert it
            X_train_full[col] = converted
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

# Fill missing numeric values with median
for col in numeric_cols:
    median_val = X_train_full[col].median()
    X_train_full[col] = X_train_full[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Combine train and test to fit encoder
    combined = pd.concat([X_train_full[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42
)

# Train models and evaluate
models = {
    'LGBM': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1
    ),
    'XGB': XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='rmse'
    ),
    'CatBoost': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0
    )
}

val_predictions = []
trained_models = {}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    print(f"{name} Validation RMSE: {rmse:.4f}")
    val_predictions.append(val_pred)
    trained_models[name] = model

# Ensemble: average predictions from all models on test set
test_predictions = []
for name, model in trained_models.items():
    test_pred = model.predict(X_test)
    test_predictions.append(test_pred)

final_predictions = np.mean(test_predictions, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'TOTAL_ACTIVE_ACCOUNTS_': final_predictions
})
output_df.to_csv('predictions.csv', index=False)