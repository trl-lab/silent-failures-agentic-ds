import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/managed-care-enrollment/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/managed-care-enrollment/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Total Enrollment'
y = train_df[target_col].values.astype(np.float64)

X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

n_train = len(X_train)
n_test = len(X_test)

# Combine for uniform preprocessing
combined_df = pd.concat([X_train, X_test], ignore_index=True)

categorical_cols = ['State', 'Notes', 'Program Name', 'Plan Name', 'Geographic Region', 'Parent Organization']
numeric_cols = ['Year']

for col in categorical_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].fillna('Unknown').astype(str).str.strip()

for col in numeric_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].fillna(combined_df[col].median()).astype(float)

# Feature Engineering
features_df = pd.DataFrame(index=combined_df.index)

# Add numeric columns
for col in numeric_cols:
    features_df[col] = combined_df[col]

# Frequency encoding and character lengths
for col in categorical_cols:
    freq = combined_df[col].value_counts()
    features_df[f'{col}_freq'] = combined_df[col].map(freq).astype(float)
    features_df[f'{col}_len'] = combined_df[col].apply(len).astype(float)
    features_df[f'{col}_word_count'] = combined_df[col].apply(lambda x: len(x.split())).astype(float)

# Ordinal encoding for LightGBM
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoded_cats = ordinal_encoder.fit_transform(combined_df[categorical_cols])
for i, col in enumerate(categorical_cols):
    features_df[f'{col}_encoded'] = encoded_cats[:, i]

# TF-IDF + SVD on combined text representation
combined_text = combined_df[categorical_cols].apply(lambda row: ' '.join(row), axis=1)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b')
tfidf_matrix = tfidf.fit_transform(combined_text)

n_svd_components = min(30, tfidf_matrix.shape[1] - 1)
svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
svd_features = svd.fit_transform(tfidf_matrix)

for i in range(n_svd_components):
    features_df[f'svd_{i}'] = svd_features[:, i]

X_train_final = features_df.iloc[:n_train].values
X_test_final = features_df.iloc[n_train:].values

# 5-Fold Cross Validation Modeling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(n_test)
cat_preds = np.zeros(n_test)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False
}

for train_idx, val_idx in kf.split(X_train_final):
    X_tr, y_tr = X_train_final[train_idx], y[train_idx]
    X_va, y_va = X_train_final[val_idx], y[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test_final) / kf.n_splits

    # CatBoost
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cat_preds += cat_model.predict(X_test_final) / kf.n_splits

# Ensemble predictions
final_predictions = 0.5 * lgb_preds + 0.5 * cat_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Total Enrollment': final_predictions
})
output_df.to_csv('predictions.csv', index=False)