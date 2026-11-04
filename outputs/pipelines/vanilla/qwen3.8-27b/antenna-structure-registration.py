import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/antenna-structure-registration/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/antenna-structure-registration/data/test.parquet')

# Separate target
y_train = train_df['Strucht'].copy()
X_train_full = train_df.drop('Strucht', axis=1)
X_test = test_df.copy()

# Drop high-cardinality / non-informative columns
drop_cols = ['Entity', 'ContAdd', 'ContPO', 'ContName', 'LocAdd', 'url', 'FAAstudy', 'FAAcirc']
drop_cols = [c for c in drop_cols if c in X_train_full.columns]
X_train_full = X_train_full.drop(columns=drop_cols)
X_test = X_test.drop(columns=drop_cols)

# Identify categorical columns
cat_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train_full.select_dtypes(include=['number']).columns.tolist()

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train_full[col].astype(str), X_test[col].astype(str)], axis=0)
    combined = combined.fillna('MISSING')
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str).fillna('MISSING'))
    X_test[col] = le.transform(X_test[col].astype(str).fillna('MISSING'))
    label_encoders[col] = le

# Fill missing numeric values with median
for col in num_cols:
    median_val = X_train_full[col].median()
    X_train_full[col] = X_train_full[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train, test_size=0.15, random_state=42)

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
        subsample=0.8,
        colsample_bylevel=0.8,
        random_state=42,
        verbose=0
    )
}

# Train models and get validation scores
val_preds = []
test_preds = []

for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    print(f"{name}: Val RMSE = {val_rmse:.4f}")
    val_preds.append(val_pred)
    test_preds.append(model.predict(X_test))

# Ensemble: average predictions
val_ensemble = np.mean(val_preds, axis=0)
test_ensemble = np.mean(test_preds, axis=0)

val_rmse_ensemble = np.sqrt(np.mean((y_val - val_ensemble) ** 2))
print(f"Ensemble: Val RMSE = {val_rmse_ensemble:.4f}")

# Save predictions
predictions = pd.DataFrame({
    'id': range(len(test_ensemble)),
    'Strucht': test_ensemble
})
predictions.to_csv('predictions.csv', index=False)