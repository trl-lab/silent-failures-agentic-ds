import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/colleges-and-universities/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/colleges-and-universities/data/test.parquet')

# Define features to use (drop identifiers, text, and non-predictive columns)
drop_cols = [
    'OBJECTID', 'IPEDSID', 'NAME', 'ADDRESS', 'CITY', 'ZIP', 'ZIP4',
    'TELEPHONE', 'COUNTY', 'COUNTYFIPS', 'COUNTRY', 'NAICS_DESC',
    'SOURCE', 'SOURCEDATE', 'VAL_METHOD', 'VAL_DATE', 'WEBSITE',
    'STFIPS', 'COFIPS', 'ALIAS', 'SHELTER_ID', 'CLOSE_DATE', 'RATIO_ENROLL_EMP'
]

# Identify columns to keep
all_cols = train_df.columns.tolist()
feature_cols = [c for c in all_cols if c not in drop_cols]

# Separate features and target
X = train_df[feature_cols].copy()
y = train_df['RATIO_ENROLL_EMP'].copy()
X_test = test_df[feature_cols].copy()

# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Combine train and test to fit encoder
    combined = pd.concat([X[col].astype(str), X_test[col].astype(str)], axis=0)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Fill missing numeric values with median
for col in num_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Create validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models and evaluate
models = {
    'lgbm': LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
    'xgb': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='rmse'),
    'cat': CatBoostRegressor(iterations=500, learning_rate=0.05, random_state=42, verbose=0)
}

val_preds = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - pred) ** 2))
    print(f"{name}: Val RMSE = {rmse:.6f}")
    val_preds.append(pred)

# Ensemble prediction on test
test_preds = []
for name, model in models.items():
    test_preds.append(model.predict(X_test))

# Average ensemble
final_pred = np.mean(test_preds, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'RATIO_ENROLL_EMP': final_pred
})
output_df.to_csv('predictions.csv', index=False)