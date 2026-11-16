import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/historical-earthquake-locations/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/historical-earthquake-locations/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Damage Description (total)'

# Filter out rows where target is null
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

# Encode target
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df[target_col].astype(str))

# Combine features for unified preprocessing
combined_df = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0).reset_index(drop=True)
n_train = len(train_df)

# Text processing on Comments
combined_df['Comments'] = combined_df['Comments'].fillna('').astype(str)
tfidf_comments = TfidfVectorizer(max_features=200, stop_words='english')
svd_comments = TruncatedSVD(n_components=20, random_state=42)
comments_svd = svd_comments.fit_transform(tfidf_comments.fit_transform(combined_df['Comments']))
comments_svd_df = pd.DataFrame(comments_svd, columns=[f'comments_svd_{i}' for i in range(20)])

# Text processing on Location Name
combined_df['Location Name'] = combined_df['Location Name'].fillna('').astype(str)
tfidf_loc = TfidfVectorizer(max_features=100, stop_words='english')
svd_loc = TruncatedSVD(n_components=10, random_state=42)
loc_svd = svd_loc.fit_transform(tfidf_loc.fit_transform(combined_df['Location Name']))
loc_svd_df = pd.DataFrame(loc_svd, columns=[f'loc_svd_{i}' for i in range(10)])

# Drop unneeded text / id columns
cols_to_drop = ['Comments', 'URL', 'Date (YYYY/MM/DD)', 'OBJECTID', 'ID']
combined_df = combined_df.drop(columns=[col for col in cols_to_drop if col in combined_df.columns])

# Handle categorical features
cat_cols = combined_df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    combined_df[col] = combined_df[col].astype(str).astype('category')

# Concatenate SVD features
combined_df = pd.concat([combined_df, comments_svd_df, loc_svd_df], axis=1)

# Split back to train and test
X_train = combined_df.iloc[:n_train].copy().reset_index(drop=True)
X_test = combined_df.iloc[n_train:].copy().reset_index(drop=True)

# Train model with cross-validation
num_classes = len(le_target.classes_)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    test_preds += model.predict_proba(X_test) / skf.n_splits

# Generate final predictions
pred_classes = np.argmax(test_preds, axis=1)
pred_labels = le_target.inverse_transform(pred_classes)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    target_col: pred_labels
})
output_df.to_csv('predictions.csv', index=False)