import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/wine-quality/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/wine-quality/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'quality'
X = train_df.drop(columns=[target_col])
y = train_df[target_col].astype(float)
X_test = test_df.copy()

# Ensure same columns in test as in train features
if 'id' in X_test.columns and 'id' not in X.columns:
    X_test = X_test.drop(columns=['id'])

# Identify categorical and numerical features
categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or isinstance(X[col].dtype, pd.CategoricalDtype)]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Handle categorical features
if categorical_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols].astype(str))
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols].astype(str))

# Handle missing values
if numerical_cols:
    imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
    X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])

# K-Fold Cross Validation Setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test) / n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test) / n_splits

# Ensemble average
final_predictions = (lgb_preds + xgb_preds + cat_preds) / 3.0

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'quality': final_predictions
})
output_df.to_csv('predictions.csv', index=False)