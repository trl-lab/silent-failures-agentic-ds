import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/covid-clinical-trials/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/covid-clinical-trials/data/test.parquet')

def extract_features(df):
    features = pd.DataFrame(index=df.index)
    
    # Numeric features
    features['Rank'] = df['Rank'].astype(float)
    features['Enrollment'] = df['Enrollment'].fillna(df['Enrollment'].median())
    
    # Count features from pipe-separated text columns
    pipe_cols = ['Conditions', 'Interventions', 'Outcome Measures', 
                 'Sponsor/Collaborators', 'Locations', 'Funded Bys', 'Other IDs']
    for col in pipe_cols:
        if col in df.columns:
            features[f'{col}_count'] = df[col].fillna('').apply(
                lambda x: len([s for s in str(x).split('|') if s.strip()])
            )
    
    # Binary features
    features['Has_Acronym'] = (df['Acronym'].fillna('None') != 'None').astype(int)
    features['Has_Study_Documents'] = (df['Study Documents'].fillna('None') != 'None').astype(int)
    features['Has_Results_First_Posted'] = (df['Results First Posted'].fillna('None') != 'None').astype(int)
    
    # Categorical features
    cat_cols = ['Status', 'Study Results', 'Gender', 'Age', 'Phases', 
                'Study Type', 'Study Designs']
    for col in cat_cols:
        if col in df.columns:
            features[col] = df[col].fillna('Unknown').astype(str)
    
    # Date features - extract year and month
    date_cols = ['Primary Completion Date', 'First Posted', 'Last Update Posted']
    for col in date_cols:
        if col in df.columns:
            dates = pd.to_datetime(df[col].fillna(''), errors='coerce')
            features[f'{col}_year'] = dates.dt.year.fillna(0).astype(int)
            features[f'{col}_month'] = dates.dt.month.fillna(0).astype(int)
    
    # Title length as a feature
    features['Title_length'] = df['Title'].fillna('').str.len()
    
    return features

# Extract features
X_train = extract_features(train_df)
X_test = extract_features(test_df)
y_train = train_df['days_to_complete'].values

# Encode categorical columns with safe handling of unseen labels
cat_feature_cols = ['Status', 'Study Results', 'Gender', 'Age', 'Phases', 
                    'Study Type', 'Study Designs']
label_mappings = {}
for col in cat_feature_cols:
    if col in X_train.columns:
        unique_vals = sorted(X_train[col].unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        label_mappings[col] = mapping
        X_train[col] = X_train[col].map(mapping).fillna(-1).astype(int)
        X_test[col] = X_test[col].map(mapping).fillna(-1).astype(int)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train models
lgbm_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_tr, y_tr)
lgbm_val_pred = lgbm_model.predict(X_val)

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    random_state=42,
    eval_metric='rmse'
)
xgb_model.fit(X_tr, y_tr)
xgb_val_pred = xgb_model.predict(X_val)

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0
)
cat_model.fit(X_tr, y_tr)
cat_val_pred = cat_model.predict(X_val)

# Evaluate
lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_val_pred))
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
cat_rmse = np.sqrt(mean_squared_error(y_val, cat_val_pred))

print(f"LightGBM RMSE: {lgbm_rmse:.4f}")
print(f"XGBoost RMSE: {xgb_rmse:.4f}")
print(f"CatBoost RMSE: {cat_rmse:.4f}")

# Retrain on full training data and predict test
lgbm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)

# Ensemble prediction
pred_lgbm = lgbm_model.predict(X_test)
pred_xgb = xgb_model.predict(X_test)
pred_cat = cat_model.predict(X_test)

# Weighted ensemble based on validation performance
weights = np.array([1/lgbm_rmse, 1/xgb_rmse, 1/cat_rmse])
weights = weights / weights.sum()

final_pred = weights[0] * pred_lgbm + weights[1] * pred_xgb + weights[2] * pred_cat

# Save predictions
predictions = pd.DataFrame({
    'id': range(len(final_pred)),
    'days_to_complete': final_pred
})
predictions.to_csv('predictions.csv', index=False)