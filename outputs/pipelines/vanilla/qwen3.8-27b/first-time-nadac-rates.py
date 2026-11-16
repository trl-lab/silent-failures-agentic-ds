import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/first-time-nadac-rates/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/first-time-nadac-rates/data/test.parquet')

# Feature engineering
def preprocess(df):
    df = df.copy()
    
    # Convert Package Size to numeric
    df['Package Size'] = pd.to_numeric(df['Package Size'], errors='coerce')
    
    # Parse As of Date
    df['As of Date'] = pd.to_datetime(df['As of Date'], errors='coerce', format='mixed')
    df['Date_Year'] = df['As of Date'].dt.year
    df['Date_Month'] = df['As of Date'].dt.month
    df['Date_Quarter'] = df['As of Date'].dt.quarter
    
    # Drop original date and high-cardinality text columns
    df.drop(['As of Date', 'Drug Product'], axis=1, inplace=True, errors='ignore')
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        elif df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype in ['int64', 'int32']:
            df[col] = df[col].fillna(df[col].median())
    
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Separate target
y_train = train_df['NADAC Rate']
X_train = train_df.drop('NADAC Rate', axis=1)

# Identify categorical and numeric columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Encode categoricals
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train models
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
    'cat': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        random_state=42,
        verbose=0
    )
}

# Evaluate on validation set
val_preds = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - pred) ** 2))
    print(f"{name}: Validation RMSE = {rmse:.6f}")
    val_preds.append(pred)

# Ensemble prediction on test set
test_preds = []
for name, model in models.items():
    test_preds.append(model.predict(test_df))

# Average ensemble
final_pred = np.mean(test_preds, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'NADAC Rate': final_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")