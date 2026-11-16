import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/electric-retail-service-territories/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/electric-retail-service-territories/data/test.parquet')

# Identify target and features
target = 'TOTAL_MWH'
y_train = train_df[target].values

# Drop non-predictive columns
drop_cols = ['OBJECTID', 'ID', 'NAME', 'ADDRESS', 'SOURCE', 'VAL_DATE', 'WEBSITE', target]
drop_cols = [c for c in drop_cols if c in train_df.columns]

X_train_full = train_df.drop(columns=drop_cols)
X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

# Identify categorical and numeric columns
cat_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train_full.select_dtypes(include=['number']).columns.tolist()

# Replace "NOT AVAILABLE" with NaN for categorical columns
for col in cat_cols:
    X_train_full[col] = X_train_full[col].replace('NOT AVAILABLE', np.nan)
    X_test[col] = X_test[col].replace('NOT AVAILABLE', np.nan)

# Encode categorical columns using LabelEncoder
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined train+test to handle unseen categories
    combined = pd.concat([X_train_full[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Fill missing numeric values with median
for col in num_cols:
    median_val = X_train_full[col].median()
    X_train_full[col] = X_train_full[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train, test_size=0.2, random_state=42)

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
    'cat': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        random_state=42,
        verbose=0
    )
}

# Train models and evaluate on validation set
val_predictions = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"{name}: Validation RMSE = {rmse:.4f}")
    val_predictions.append(val_pred)

# Ensemble validation
ensemble_val = np.mean(val_predictions, axis=0)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))
print(f"Ensemble: Validation RMSE = {ensemble_rmse:.4f}")

# Retrain on full training data and predict test
test_predictions = []
for name, model in models.items():
    model.fit(X_train_full, y_train)
    test_pred = model.predict(X_test)
    test_predictions.append(test_pred)

# Ensemble prediction
final_predictions = np.mean(test_predictions, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'TOTAL_MWH': final_predictions
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")