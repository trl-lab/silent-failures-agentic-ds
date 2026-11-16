import pandas as pd
import numpy as np
import catboost as cb
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasff_window/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/rasff_window/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'risk_decision'

train = train.dropna(subset=[target_col]).reset_index(drop=True)

le = LabelEncoder()
y = le.fit_transform(train[target_col].astype(str))

X_train = train.drop(columns=[target_col])
X_test = test.drop(columns=[target_col]) if target_col in test.columns else test.copy()

if 'id' in X_train.columns:
    X_train = X_train.drop(columns=['id'])
if 'id' in X_test.columns:
    X_test = X_test.drop(columns=['id'])

for col in list(X_train.columns):
    if pd.api.types.is_datetime64_any_dtype(X_train[col]) or 'date' in col.lower() or 'time' in col.lower():
        try:
            d_train = pd.to_datetime(X_train[col], errors='coerce')
            d_test = pd.to_datetime(X_test[col], errors='coerce')
            if d_train.notna().sum() > 0:
                X_train[col + '_year'] = d_train.dt.year.fillna(-1)
                X_train[col + '_month'] = d_train.dt.month.fillna(-1)
                X_train[col + '_day'] = d_train.dt.day.fillna(-1)
                X_train[col + '_dayofweek'] = d_train.dt.dayofweek.fillna(-1)
                
                X_test[col + '_year'] = d_test.dt.year.fillna(-1)
                X_test[col + '_month'] = d_test.dt.month.fillna(-1)
                X_test[col + '_day'] = d_test.dt.day.fillna(-1)
                X_test[col + '_dayofweek'] = d_test.dt.dayofweek.fillna(-1)
                
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
        except Exception:
            pass

cat_cols = []
num_cols = []
text_cols = []

for col in list(X_train.columns):
    if pd.api.types.is_numeric_dtype(X_train[col]):
        num_cols.append(col)
    else:
        X_train[col] = X_train[col].fillna('').astype(str)
        X_test[col] = X_test[col].fillna('').astype(str)
        
        avg_len = X_train[col].str.len().mean()
        n_unique = X_train[col].nunique()
        
        if avg_len > 50 and n_unique > 100:
            text_cols.append(col)
        else:
            cat_cols.append(col)

for col in text_cols:
    try:
        tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
        svd = TruncatedSVD(n_components=15, random_state=42)
        
        tr_vec = tfidf.fit_transform(X_train[col])
        te_vec = tfidf.transform(X_test[col])
        
        tr_svd = svd.fit_transform(tr_vec)
        te_svd = svd.transform(te_vec)
        
        for i in range(tr_svd.shape[1]):
            col_name = f'{col}_svd_{i}'
            X_train[col_name] = tr_svd[:, i]
            X_test[col_name] = te_svd[:, i]
            num_cols.append(col_name)
    except Exception:
        pass
    X_train = X_train.drop(columns=[col])
    X_test = X_test.drop(columns=[col])

for col in cat_cols:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

for col in num_cols:
    if col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(-999)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(-999)

cat_cols = [c for c in cat_cols if c in X_train.columns]

model = cb.CatBoostClassifier(
    iterations=800,
    learning_rate=0.08,
    depth=6,
    cat_features=cat_cols if len(cat_cols) > 0 else None,
    random_seed=42,
    verbose=0
)

model.fit(X_train, y)

test_preds = model.predict(X_test)
test_preds = np.asarray(test_preds).ravel().astype(int)

test_preds_labels = le.inverse_transform(test_preds)

sub = pd.DataFrame({
    'id': np.arange(len(test_preds_labels)),
    'risk_decision': test_preds_labels
})

sub.to_csv('predictions.csv', index=False)