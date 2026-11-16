import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/china-overseas-finance-inventory/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/china-overseas-finance-inventory/data/test.parquet')

# Define columns to drop (IDs and non-predictive)
drop_cols = [
    'unique_id', 'power_plant_name', 'compid', 'unit_id', 'location_id.x',
    'gppd_id', 'fdi_id', 'rma_id', 'bu_id', 'sais_id', 'iad_id', 'r_id',
    'ij_id', 'cgp_id', 'equity_investor_name_1', 'equity_investor_name_2',
    'parent_company_of_investor', 'city', 'province'
]

# Separate target
y_train = train_df['installed_capacity'].copy()
X_train_full = train_df.drop('installed_capacity', axis=1)
X_test_full = test_df.copy()

# Drop ID/non-predictive columns
drop_cols = [c for c in drop_cols if c in X_train_full.columns]
X_train_full = X_train_full.drop(columns=drop_cols)
X_test_full = X_test_full.drop(columns=drop_cols)

# Identify categorical and numeric columns
cat_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train_full.select_dtypes(include=['number']).columns.tolist()

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Combine train and test to fit encoder
    combined = pd.concat([X_train_full[col].astype(str), X_test_full[col].astype(str)], axis=0)
    combined = combined.fillna('MISSING')
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str).fillna('MISSING'))
    X_test_full[col] = le.transform(X_test_full[col].astype(str).fillna('MISSING'))
    label_encoders[col] = le

# Fill missing numeric values with -999 (a sentinel value that tree models can handle)
for col in num_cols:
    X_train_full[col] = X_train_full[col].fillna(-999)
    X_test_full[col] = X_test_full[col].fillna(-999)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42
)

# Define models
models = {
    'lgbm': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    'xgb': XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    ),
    'cat': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )
}

# Train and evaluate models
val_predictions = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    print(f"{name}: Val RMSE = {rmse:.4f}")
    val_predictions.append(val_pred)

# Ensemble validation score
ensemble_val = np.mean(val_predictions, axis=0)
ensemble_rmse = np.sqrt(np.mean((ensemble_val - y_val) ** 2))
print(f"Ensemble: Val RMSE = {ensemble_rmse:.4f}")

# Retrain on full training data and predict test
test_predictions = []
for name, model in models.items():
    model.fit(X_train_full, y_train)
    test_pred = model.predict(X_test_full)
    test_predictions.append(test_pred)

# Ensemble prediction
final_predictions = np.mean(test_predictions, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'installed_capacity': final_predictions
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")