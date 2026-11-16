import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/mercedes-benz-greener-manufacturing/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/mercedes-benz-greener-manufacturing/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate target and drop ID
y_train = train_df['y'].values
X_train = train_df.drop(columns=['ID', 'y'], errors='ignore')
X_test = test_df.drop(columns=['ID'], errors='ignore')

# Identify categorical and numerical columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object' or str(X_train[col].dtype) == 'category']
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

# Remove constant columns (zero variance in train)
usable_numerical_cols = []
for col in numerical_cols:
    if X_train[col].nunique() > 1:
        usable_numerical_cols.append(col)

X_train = X_train[categorical_cols + usable_numerical_cols].copy()
X_test = X_test[categorical_cols + usable_numerical_cols].copy()

# Ordinal encode categorical columns
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

# Handle any missing values in numerical columns
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Decomposition features
n_comp = 10
decompositions = [
    ('pca', PCA(n_components=n_comp, random_state=42)),
    ('ica', FastICA(n_components=n_comp, max_iter=500, random_state=42)),
    ('tsvd', TruncatedSVD(n_components=n_comp, random_state=42)),
    ('grp', GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=42)),
    ('srp', SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=42))
]

X_train_decomp = []
X_test_decomp = []

for name, model in decompositions:
    train_trans = model.fit_transform(X_train)
    test_trans = model.transform(X_test)
    for i in range(n_comp):
        X_train[f'{name}_{i}'] = train_trans[:, i]
        X_test[f'{name}_{i}'] = test_trans[:, i]

# Prepare CV setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_preds += lgb_model.predict(X_test) / n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold
    )
    xgb_model.fit(X_tr, y_tr)
    xgb_preds += xgb_model.predict(X_test) / n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=350,
        learning_rate=0.01,
        depth=4,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(X_tr, y_tr)
    cat_preds += cat_model.predict(X_test) / n_splits

# Ensemble average
final_predictions = 0.35 * lgb_preds + 0.35 * xgb_preds + 0.30 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'y': final_predictions
})
output_df.to_csv('predictions.csv', index=False)