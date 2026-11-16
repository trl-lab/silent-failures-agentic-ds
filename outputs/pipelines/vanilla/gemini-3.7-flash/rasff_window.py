import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from lightgbm import LGBMClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasff_window/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasff_window/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'risk_decision'
y_raw = train_df[target_col].astype(str)

le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)

# 2. Feature Engineering
def extract_tabular_and_text_features(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Process date features
    for df in [df_train, df_test]:
        dt = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        df['year'] = dt.dt.year.fillna(-1)
        df['month'] = dt.dt.month.fillna(-1)
        df['day'] = dt.dt.day.fillna(-1)
        df['dayofweek'] = dt.dt.dayofweek.fillna(-1)
        df['hour'] = dt.dt.hour.fillna(-1)
        
        # Reference breakdown
        df['ref_year'] = df['reference'].apply(lambda x: int(x) if pd.notnull(x) else -1)
        df['ref_num'] = df['reference'].apply(lambda x: int(round((x - int(x)) * 10000)) if pd.notnull(x) else -1)

    num_cols = ['year', 'month', 'day', 'dayofweek', 'hour', 'ref_year', 'ref_num', 'reference']
    for col in num_cols:
        df_train[col] = df_train[col].fillna(-1)
        df_test[col] = df_test[col].fillna(-1)

    X_train_num = df_train[num_cols].values
    X_test_num = df_test[num_cols].values

    # Categorical / text fields for TF-IDF extraction
    text_fields = {
        'subject': 500,
        'hazards': 200,
        'category': 100,
        'type': 30,
        'notifying_country': 100,
        'classification': 50,
        'distribution': 150,
        'forAttention': 50,
        'forFollowUp': 100,
        'operator': 150,
        'origin': 150
    }

    train_sparse_list = [csr_matrix(X_train_num)]
    test_sparse_list = [csr_matrix(X_test_num)]

    for field, max_feat in text_fields.items():
        if field in df_train.columns:
            tfidf = TfidfVectorizer(max_features=max_feat, token_pattern=r'(?u)\b\w+\b', min_df=2)
            train_text = df_train[field].fillna('missing').astype(str)
            test_text = df_test[field].fillna('missing').astype(str)

            train_vec = tfidf.fit_transform(train_text)
            test_vec = tfidf.transform(test_text)

            train_sparse_list.append(train_vec)
            test_sparse_list.append(test_vec)

    X_train = hstack(train_sparse_list).tocsr()
    X_test = hstack(test_sparse_list).tocsr()

    return X_train, X_test

X_train, X_test = extract_tabular_and_text_features(train_df, test_df)

# 3. Model Training with Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_classes = len(le_target.classes_)
test_preds_proba = np.zeros((X_test.shape[0], n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_va = X_train[val_idx], y[val_idx]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_tr, y_tr)
    test_preds_proba += model.predict_proba(X_test) / skf.n_splits

final_preds_indices = np.argmax(test_preds_proba, axis=1)
final_preds = le_target.inverse_transform(final_preds_indices)

# 4. Save Predictions
submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'risk_decision': final_preds
})

submission.to_csv('predictions.csv', index=False)