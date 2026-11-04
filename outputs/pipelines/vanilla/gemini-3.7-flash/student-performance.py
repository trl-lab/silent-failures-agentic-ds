import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/student-performance/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/student-performance/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'G3'
X = train_df.drop(columns=[target_col])
y = train_df[target_col].values
X_test = test_df.copy()

# Identify column types
categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Handle any missing values
for col in numeric_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

for col in categorical_cols:
    mode_val = X[col].mode()[0] if not X[col].mode().empty else 'missing'
    X[col] = X[col].fillna(mode_val).astype(str)
    X_test[col] = X_test[col].fillna(mode_val).astype(str)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

X_proc = preprocessor.fit_transform(X)
X_test_proc = preprocessor.transform(X_test)

# K-Fold Ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_proc, y)):
    X_tr, y_tr = X_proc[train_idx], y[train_idx]
    X_va, y_va = X_proc[val_idx], y[val_idx]

    # Models
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    
    cb_model = CatBoostRegressor(
        iterations=400,
        learning_rate=0.03,
        depth=5,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(X_tr, y_tr)
    
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42 + fold,
        n_jobs=-1
    )
    rf_model.fit(X_tr, y_tr)
    
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_tr, y_tr)
    
    fold_pred = (
        0.35 * lgb_model.predict(X_test_proc) +
        0.35 * cb_model.predict(X_test_proc) +
        0.15 * rf_model.predict(X_test_proc) +
        0.15 * ridge_model.predict(X_test_proc)
    )
    
    test_preds += fold_pred / kf.n_splits

# Clip predictions to reasonable domain range for grades [0, 20]
test_preds = np.clip(test_preds, 0, 20)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'G3': test_preds
})

output_df.to_csv('predictions.csv', index=False)