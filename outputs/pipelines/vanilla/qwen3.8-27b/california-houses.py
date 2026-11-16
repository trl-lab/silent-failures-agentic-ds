import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/california-houses/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/california-houses/data/test.parquet')

# Define target
target = 'Total interior livable area'

# Drop the target and id from features
train_df = train_df.drop(columns=['Id'], errors='ignore')
test_df = test_df.drop(columns=['Id'], errors='ignore')

# Drop text columns that are not useful for tabular models
text_cols = ['Summary', 'Address']
for col in text_cols:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

# Drop the target from train
y_train = train_df[target].copy()
X_train = train_df.drop(columns=[target])

# Identify feature columns (intersection of train and test)
feature_cols = [col for col in X_train.columns if col in test_df.columns]
X_train = X_train[feature_cols]
X_test = test_df[feature_cols]

# Identify categorical and numeric columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Combine train and test to fit encoder
    combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Fill missing numeric values with median
for col in num_cols:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Define models
models = {
    'lgbm': LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    ),
    'xgb': XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='rmse'
    ),
    'cat': CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=0
    )
}

# Train models and evaluate
val_preds = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    print(f"{name}: Val RMSE = {rmse:.4f}")
    val_preds.append(pred)

# Ensemble validation
ensemble_val = np.mean(val_preds, axis=0)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))
print(f"Ensemble: Val RMSE = {ensemble_rmse:.4f}")

# Retrain on full training data and predict test
test_preds = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    test_preds.append(pred)

# Ensemble test predictions
final_pred = np.mean(test_preds, axis=0)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'Total interior livable area': final_pred
})
output_df.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {output_df.shape}")