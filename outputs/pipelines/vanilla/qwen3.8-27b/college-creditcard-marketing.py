import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/college-creditcard-marketing/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/college-creditcard-marketing/data/test.parquet')

# Clean column names (remove BOM and extra spaces)
train_df.columns = [c.replace('\ufeff', '').strip() for c in train_df.columns]
test_df.columns = [c.replace('\ufeff', '').strip() for c in test_df.columns]

# Separate target
y_train = train_df['PAYMENTS_BY_ISSUER'].values
train_df = train_df.drop('PAYMENTS_BY_ISSUER', axis=1)

# Identify columns to drop (high cardinality identifiers)
drop_cols = ['INSTITUTION_OR_ORGANIZATION', 'OPEID', 'UNIVERSITY_NAME']
drop_cols = [c for c in drop_cols if c in train_df.columns]
train_df = train_df.drop(columns=drop_cols, errors='ignore')
test_df = test_df.drop(columns=drop_cols, errors='ignore')

# Handle NEW_ACCOUNTS_OPENED_IN_REPORTING_YEAR - convert to numeric
for df in [train_df, test_df]:
    df['NEW_ACCOUNTS_OPENED_IN_REPORTING_YEAR'] = pd.to_numeric(
        df['NEW_ACCOUNTS_OPENED_IN_REPORTING_YEAR'].replace(['.', 'None', ''], np.nan), errors='coerce'
    )

# Normalize string columns (lowercase, strip)
string_cols = train_df.select_dtypes(include='object').columns.tolist()
for col in string_cols:
    train_df[col] = train_df[col].astype(str).str.strip().str.lower().replace('nan', np.nan)
    test_df[col] = test_df[col].astype(str).str.strip().str.lower().replace('nan', np.nan)

# Label encode categorical columns
label_encoders = {}
for col in string_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col].dropna(), test_df[col].dropna()], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = train_df[col].map(lambda x: le.transform([x])[0] if pd.notna(x) and str(x) in le.classes_ else -1)
    test_df[col] = test_df[col].map(lambda x: le.transform([x])[0] if pd.notna(x) and str(x) in le.classes_ else -1)
    label_encoders[col] = le

# Fill remaining NaNs
train_df = train_df.fillna(-999)
test_df = test_df.fillna(-999)

# Ensure all columns are numeric
train_df = train_df.apply(pd.to_numeric, errors='coerce').fillna(-999)
test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(-999)

# Validation split
X_train, X_val, y_train_split, y_val_split = train_test_split(
    train_df, y_train, test_size=0.2, random_state=42
)

# Train models
lgbm_model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_train, y_train_split)

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    eval_metric='rmse',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train_split)

cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0
)
cat_model.fit(X_train, y_train_split)

# Evaluate on validation set
lgbm_val = lgbm_model.predict(X_val)
xgb_val = xgb_model.predict(X_val)
cat_val = cat_model.predict(X_val)

lgbm_rmse = np.sqrt(np.mean((lgbm_val - y_val_split) ** 2))
xgb_rmse = np.sqrt(np.mean((xgb_val - y_val_split) ** 2))
cat_rmse = np.sqrt(np.mean((cat_val - y_val_split) ** 2))

print(f"LightGBM RMSE: {lgbm_rmse:.4f}")
print(f"XGBoost RMSE: {xgb_rmse:.4f}")
print(f"CatBoost RMSE: {cat_rmse:.4f}")

# Ensemble prediction on test set
pred_lgbm = lgbm_model.predict(test_df)
pred_xgb = xgb_model.predict(test_df)
pred_cat = cat_model.predict(test_df)

# Weighted ensemble (inverse RMSE weighting)
weights = np.array([1/lgbm_rmse, 1/xgb_rmse, 1/cat_rmse])
weights = weights / weights.sum()

final_pred = weights[0] * pred_lgbm + weights[1] * pred_xgb + weights[2] * pred_cat

# Save predictions
submission_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'PAYMENTS_BY_ISSUER': final_pred
})
submission_df.to_csv('predictions.csv', index=False)