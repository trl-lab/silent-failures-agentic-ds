import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/external-clinician-dashboard/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/external-clinician-dashboard/data/test.parquet')

# Separate target
y = train_df['Score'].values
train_df = train_df.drop('Score', axis=1)

# Identify categorical and numeric columns
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()

# Encode categorical columns using label encoding (works well with tree-based models)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Combine train and test to fit encoder (avoid unknown categories in test)
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

# Fill missing values
for col in train_df.columns:
    if train_df[col].isna().any():
        train_df[col].fillna(train_df[col].median() if col in numeric_cols else -1, inplace=True)
    if test_df[col].isna().any():
        test_df[col].fillna(train_df[col].median() if col in numeric_cols else -1, inplace=True)

# Create validation split
X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size=0.2, random_state=42)

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
    'catboost': CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
}

# Train and evaluate on validation set
val_predictions = []
for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"{name}: Validation RMSE = {rmse:.4f}")
    val_predictions.append(val_pred)

# Ensemble validation score
ensemble_val_pred = np.mean(val_predictions, axis=0)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
print(f"Ensemble: Validation RMSE = {ensemble_rmse:.4f}")

# Retrain on full training data and predict test
test_predictions = []
for name, model in models.items():
    model.fit(train_df, y)
    test_pred = model.predict(test_df)
    test_predictions.append(test_pred)

# Ensemble prediction
final_predictions = np.mean(test_predictions, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_predictions)),
    'Score': final_predictions
})
output_df.to_csv('predictions.csv', index=False)