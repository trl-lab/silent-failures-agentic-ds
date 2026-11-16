import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/corporate-procurement-contract-awards/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/corporate-procurement-contract-awards/data/test.parquet')

target = 'Contract Award Amount (USD)'

# Drop high-cardinality / non-informative columns
drop_cols = ['Contract Description', 'Selection Number', 'Supplier']
for col in drop_cols:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

# Parse Award Date for temporal features
train_df['Award Date'] = pd.to_datetime(train_df['Award Date'], errors='coerce')
test_df['Award Date'] = pd.to_datetime(test_df['Award Date'], errors='coerce')

train_df['Award_Year'] = train_df['Award Date'].dt.year
train_df['Award_Month'] = train_df['Award Date'].dt.month
test_df['Award_Year'] = test_df['Award Date'].dt.year
test_df['Award_Month'] = test_df['Award Date'].dt.month

train_df = train_df.drop(columns=['Award Date'])
test_df = test_df.drop(columns=['Award Date'])

# Identify categorical and numeric columns
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()

# Remove target from numeric cols if present
if target in numeric_cols:
    numeric_cols.remove(target)

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Combine train and test to fit encoder (avoid unknown categories)
    combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Fill missing numeric values with median
for col in numeric_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    test_df[col] = test_df[col].fillna(median_val)

# Separate features and target
X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_test = test_df

# Validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define models
models = {
    'lgbm': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    'xgb': XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    ),
    'catboost': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        random_state=42,
        verbose=0
    )
}

# Train models and get validation RMSE
val_predictions = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    print(f"{name} validation RMSE: {rmse:.6f}")
    val_predictions.append(val_pred)

# Ensemble: average predictions from all models on test set
test_predictions = []
for name, model in models.items():
    test_pred = model.predict(X_test)
    test_predictions.append(test_pred)

# Average ensemble
final_predictions = np.mean(test_predictions, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'Contract Award Amount (USD)': final_predictions
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")