import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/electric-generating-plants/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/electric-generating-plants/data/test.parquet')

# Define columns to drop (irrelevant text/ID columns)
drop_cols = ['OBJECTID', 'PLANT_CODE', 'NAME', 'ADDRESS', 'TELEPHONE', 'WEBSITE',
             'OPERATOR', 'OPERAT_ID', 'SUB_1', 'SUB_2', 'ZIP', 'NAICS_DESC']

# Keep useful categorical columns
cat_cols = ['STATE', 'TYPE', 'COUNTY', 'NAICS_CODE', 'PRIM_FUEL', 'SEC_FUEL', 'SOURCE', 'VAL_METHOD']

# Keep numeric columns
num_cols = ['COUNTYFIPS', 'LATITUDE', 'LONGITUDE', 'GEN_UNITS', 'PLAN_UNIT',
            'RETIR_UNIT', 'COAL_USED', 'NGAS_USED', 'OIL_USED', 'LINES',
            'SOURCE_LAT', 'SOURC_LONG', 'x', 'y']

# Date columns - extract year
date_cols = ['SOURCEDATE', 'VAL_DATE']

# Drop target and irrelevant columns
target = 'NET_GEN'
y_train = train_df[target].values

# Process train
X_train = train_df.drop(columns=[target] + drop_cols, errors='ignore')

# Process test
X_test = test_df.drop(columns=drop_cols, errors='ignore')

# Handle date columns - extract year
for col in date_cols:
    if col in X_train.columns:
        X_train[col + '_year'] = pd.to_datetime(X_train[col], errors='coerce').dt.year
        X_train = X_train.drop(columns=[col])
    if col in X_test.columns:
        X_test[col + '_year'] = pd.to_datetime(X_test[col], errors='coerce').dt.year
        X_test = X_test.drop(columns=[col])

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    if col in X_train.columns:
        le = LabelEncoder()
        # Combine train and test to fit encoder
        combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

# Ensure all columns are numeric
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str) if col in X_test.columns else pd.Series(['unknown'] * len(X_test))
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

# Fill remaining NaN values
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)

# Ensure same columns
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train models and evaluate
models = {
    'lgbm': LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=7,
                           num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbose=-1),
    'xgb': XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=7,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, eval_metric='rmse'),
    'cat': CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=7,
                             random_state=42, verbose=0)
}

val_preds = []
test_preds = []

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"{name} validation RMSE: {val_rmse:.4f}")
    val_preds.append(val_pred)
    test_preds.append(model.predict(X_test))

# Ensemble: average predictions
val_ensemble = np.mean(val_preds, axis=0)
test_ensemble = np.mean(test_preds, axis=0)

print(f"Ensemble validation RMSE: {np.sqrt(mean_squared_error(y_val, val_ensemble)):.4f}")

# Save predictions
predictions = pd.DataFrame({
    'id': range(len(test_ensemble)),
    'NET_GEN': test_ensemble
})
predictions.to_csv('predictions.csv', index=False)