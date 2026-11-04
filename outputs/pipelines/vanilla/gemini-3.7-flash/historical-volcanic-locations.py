import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/historical-volcanic-locations/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/historical-volcanic-locations/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'DEATHS_TOTAL_DESCRIPTION'

# Filter out missing targets if any
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

# Encode target
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].astype(str))

X_train = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col], errors='ignore').copy()

# Feature Engineering: Comments TF-IDF
comments_train = X_train['COMMENTS'].fillna('').astype(str)
comments_test = X_test['COMMENTS'].fillna('').astype(str)

tfidf = TfidfVectorizer(max_features=50, stop_words='english', token_pattern=r'(?u)\b\w+\b')
tfidf_train = tfidf.fit_transform(comments_train).toarray()
tfidf_test = tfidf.transform(comments_test).toarray()

tfidf_cols = [f'tfidf_{i}' for i in range(tfidf_train.shape[1])]
tfidf_train_df = pd.DataFrame(tfidf_train, columns=tfidf_cols, index=X_train.index)
tfidf_test_df = pd.DataFrame(tfidf_test, columns=tfidf_cols, index=X_test.index)

X_train = pd.concat([X_train.drop(columns=['COMMENTS']), tfidf_train_df], axis=1)
X_test = pd.concat([X_test.drop(columns=['COMMENTS']), tfidf_test_df], axis=1)

# Handle categorical features
cat_cols = [col for col in X_train.columns if X_train[col].dtype == 'object' or str(X_train[col].dtype) == 'category']

for col in cat_cols:
    X_train[col] = X_train[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
X_test[cat_cols] = oe.transform(X_test[cat_cols])

# Ensure all columns are numeric
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Model training with Stratified K-Fold ensemble
n_classes = len(le.classes_)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_lgb = np.zeros((len(X_test), n_classes))
test_preds_cb = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    test_preds_lgb += lgb.predict_proba(X_test) / n_splits
    
    # CatBoost
    cb = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb.fit(X_tr, y_tr)
    test_preds_cb += cb.predict_proba(X_test) / n_splits

# Blend predictions
final_probs = 0.5 * test_preds_lgb + 0.5 * test_preds_cb
final_class_indices = np.argmax(final_probs, axis=1)
final_predictions = le.inverse_transform(final_class_indices)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'DEATHS_TOTAL_DESCRIPTION': final_predictions
})
output_df.to_csv('predictions.csv', index=False)