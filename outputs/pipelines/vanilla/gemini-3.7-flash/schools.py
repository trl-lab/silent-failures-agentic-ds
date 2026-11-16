import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/schools/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/schools/data/test.parquet')

target_col = 'Enrollment_FT_Teachers_Ratio'
y = train_df[target_col].values

# Combine for feature engineering
train_df = train_df.drop(columns=[target_col])
n_train = len(train_df)
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Parse date
if 'Validation Date' in df.columns:
    df['Validation_Date_dt'] = pd.to_datetime(df['Validation Date'], errors='coerce')
    df['Validation_Year'] = df['Validation_Date_dt'].dt.year.fillna(-1).astype(int)
    df['Validation_Month'] = df['Validation_Date_dt'].dt.month.fillna(-1).astype(int)
    df.drop(columns=['Validation Date', 'Validation_Date_dt'], inplace=True)

# Drop redundant or very high cardinality uninformative text columns
cols_to_drop = ['OBJECTID', 'NCES ID', 'Name', 'Address', 'Telephone', 'Source', 'Website']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Categorical columns
cat_cols = ['City', 'State', 'Zip4', 'County', 'Validation Method', 'Level', 'Start Grade', 'End Grade', 'Shelter ID']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('missing')
        # Frequency encoding
        freq = df[col].value_counts()
        df[f'{col}_freq'] = df[col].map(freq)
        df[col] = df[col].astype('category')

# Numerical columns
num_cols = ['Zip', 'Type', 'Status', 'County FIPS', 'Latitude', 'Longitude', 'x', 'y', 'District ID']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Split back to train and test
X = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

# Cross-validation training with LightGBM and CatBoost
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

# Prepare for CatBoost
cb_cat_cols = [col for col in X.columns if X[col].dtype.name == 'category']
X_cb = X.copy()
X_test_cb = X_test.copy()
for col in cb_cat_cols:
    X_cb[col] = X_cb[col].astype(str).fillna('missing')
    X_test_cb[col] = X_test_cb[col].astype(str).fillna('missing')

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits
    
    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        cat_features=cb_cat_cols,
        random_seed=42,
        verbose=0
    )
    cb_model.fit(
        X_cb.iloc[train_idx], y_tr,
        eval_set=(X_cb.iloc[val_idx], y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cb_model.predict(X_test_cb) / kf.n_splits

# Ensemble predictions
final_preds = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_preds)),
    'Enrollment_FT_Teachers_Ratio': final_preds
})
output_df.to_csv('predictions.csv', index=False)