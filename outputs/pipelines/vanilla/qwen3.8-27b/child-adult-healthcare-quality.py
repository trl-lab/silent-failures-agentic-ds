import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/child-adult-healthcare-quality/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/child-adult-healthcare-quality/data/test.parquet')

# Separate target
y_train = train_df['State Rate'].copy()
X_train = train_df.drop('State Rate', axis=1)
X_test = test_df.copy()

# Identify column types
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Fill missing values
for col in numeric_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(train_df[col].median())

for col in categorical_cols:
    train_df[col] = train_df[col].fillna('MISSING')
    test_df[col] = test_df[col].fillna('MISSING')

# Re-extract after filling
X_train = train_df.drop('State Rate', axis=1)
X_test = test_df.copy()

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train models and get validation scores
models = {
    'lgbm': LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1
    ),
    'xgb': XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='rmse'
    ),
    'cat': CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0
    )
}

val_predictions = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    print(f"{name}: Val RMSE = {val_rmse:.4f}")
    val_predictions.append(val_pred)

# Ensemble validation
ensemble_val = np.mean(val_predictions, axis=0)
ensemble_rmse = np.sqrt(np.mean((y_val - ensemble_val) ** 2))
print(f"Ensemble: Val RMSE = {ensemble_rmse:.4f}")

# Retrain on full training data and predict test
test_predictions = []
for name, model in models.items():
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_predictions.append(test_pred)

# Ensemble prediction
final_predictions = np.mean(test_predictions, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'State Rate': final_predictions
})
output_df.to_csv('predictions.csv', index=False)