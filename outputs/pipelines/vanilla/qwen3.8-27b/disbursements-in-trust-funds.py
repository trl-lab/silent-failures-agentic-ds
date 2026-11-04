import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/disbursements-in-trust-funds/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/disbursements-in-trust-funds/data/test.parquet')

target = 'Cash Disbursements (US$)'

# Separate features and target
y_train = train_df[target].values
X_train_full = train_df.drop(target, axis=1)
X_test = test_df.copy()

# Identify categorical and numeric columns
categorical_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train_full.select_dtypes(include=['number']).columns.tolist()

# Fill missing values
for col in categorical_cols:
    train_df[col] = train_df[col].fillna('MISSING')
    test_df[col] = test_df[col].fillna('MISSING')

for col in numeric_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(test_df[col].median())

# Re-extract after filling
X_train_full = train_df.drop(target, axis=1)
X_test = test_df.copy()

# Label encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train_full[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train, test_size=0.2, random_state=42)

# Train models and evaluate
models = {
    'lgbm': LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
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
        random_state=42,
        verbose=0
    )
}

val_predictions = []
trained_models = {}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    print(f"{name} validation RMSE: {rmse:.6f}")
    val_predictions.append(val_pred)
    trained_models[name] = model

# Ensemble validation
ensemble_val = np.mean(val_predictions, axis=0)
ensemble_rmse = np.sqrt(np.mean((ensemble_val - y_val) ** 2))
print(f"Ensemble validation RMSE: {ensemble_rmse:.6f}")

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
    'Cash Disbursements (US$)': final_predictions
})
output_df.to_csv('predictions.csv', index=False)