import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/financial-management/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/financial-management/data/test.parquet')

# Extract latitude and longitude from Location column
def extract_coords(location_str):
    if pd.isna(location_str):
        return np.nan, np.nan
    try:
        # Format: "State (lat, lon)"
        paren_start = location_str.rfind('(')
        paren_end = location_str.rfind(')')
        if paren_start != -1 and paren_end != -1:
            coords = location_str[paren_start+1:paren_end].split(',')
            lat = float(coords[0].strip())
            lon = float(coords[1].strip())
            return lat, lon
    except:
        pass
    return np.nan, np.nan

for df in [train_df, test_df]:
    coords = df['Location'].apply(extract_coords)
    df['Latitude'] = [c[0] for c in coords]
    df['Longitude'] = [c[1] for c in coords]

# Define feature columns
categorical_cols = ['State', 'Program', 'Service Category']
numeric_cols = ['Federal Share', 'Federal Share Medicaid', 'Federal Share ARRA', 'Federal Share BIPP', 'Latitude', 'Longitude']

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Combine train and test to fit encoder
    combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Prepare features
X_train = train_df[numeric_cols + categorical_cols].copy()
y_train = train_df['State Share'].copy()
X_test = test_df[numeric_cols + categorical_cols].copy()

# Fill missing numeric values with median
for col in numeric_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train models and evaluate
models = {
    'LGBM': LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1),
    'XGB': XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='rmse'),
    'CatBoost': CatBoostRegressor(iterations=1000, learning_rate=0.05, random_state=42, verbose=0)
}

val_predictions = []
trained_models = {}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"{name} Validation RMSE: {rmse:.6f}")
    val_predictions.append(val_pred)
    trained_models[name] = model

# Ensemble validation prediction
ensemble_val_pred = np.mean(val_predictions, axis=0)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
print(f"Ensemble Validation RMSE: {ensemble_rmse:.6f}")

# Retrain on full training data and predict test
test_predictions = []
for name, model in models.items():
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_predictions.append(test_pred)

# Ensemble test predictions
final_predictions = np.mean(test_predictions, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'State Share': final_predictions
})
output_df.to_csv('predictions.csv', index=False)