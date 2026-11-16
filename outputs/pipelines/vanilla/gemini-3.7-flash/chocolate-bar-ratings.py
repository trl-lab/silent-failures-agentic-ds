import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# 1. Load data
train_path = (
    '/path/to/user/workspace/ds_failures/datasets/strable/chocolate-bar-ratings/data/train.parquet'
)
test_path = (
    '/path/to/user/workspace/ds_failures/datasets/strable/chocolate-bar-ratings/data/test.parquet'
)

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)


# 2. Feature Engineering & Preprocessing
def preprocess_features(df):
  df = df.copy()

  # Clean Cocoa Percent into a float
  if 'Cocoa Percent' in df.columns:
    df['Cocoa_Percent_Num'] = (
        df['Cocoa Percent']
        .astype(str)
        .str.replace('%', '', regex=False)
        .astype(float)
    )

  # Extract ingredient count and ingredient list
  if 'Ingredients' in df.columns:
    df['n_ingredients'] = pd.to_numeric(
        df['Ingredients'].astype(str).str.extract(r'(\d+)')[0], errors='coerce'
    )
    df['Ingredients_Clean'] = (
        df['Ingredients']
        .fillna('')
        .astype(str)
        .apply(lambda x: x.split('-')[1].strip() if '-' in x else x.strip())
    )

  # Text features fillna
  df['Characteristics'] = (
      df['Most Memorable Characteristics'].fillna('').astype(str)
  )
  df['Bean_Origin_Detail'] = (
      df['Specific Bean Origin or Bar Name'].fillna('').astype(str)
  )
  df['Company'] = df['Company (Manufacturer)'].fillna('Unknown').astype(str)
  df['Location'] = df['Company Location'].fillna('Unknown').astype(str)
  df['Country_Origin'] = (
      df['Country of Bean Origin'].fillna('Unknown').astype(str)
  )

  return df


X_train_raw = preprocess_features(train_df.drop(columns=['Rating']))
y_train_raw = train_df['Rating'].astype(str)
X_test_raw = preprocess_features(test_df)

# Encode target
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)

# Define column transformations
numeric_features = ['Review Date', 'Cocoa_Percent_Num', 'n_ingredients']
categorical_features = ['Company', 'Location', 'Country_Origin']

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            'cat',
            Pipeline([
                (
                    'imputer',
                    SimpleImputer(strategy='constant', fill_value='Unknown'),
                ),
                (
                    'onehot',
                    OneHotEncoder(handle_unknown='ignore', sparse_output=True),
                ),
            ]),
            categorical_features,
        ),
        (
            'text_char',
            TfidfVectorizer(max_features=250, ngram_range=(1, 2)),
            'Characteristics',
        ),
        (
            'text_bean',
            TfidfVectorizer(max_features=150, ngram_range=(1, 2)),
            'Bean_Origin_Detail',
        ),
        ('text_ing', TfidfVectorizer(max_features=50), 'Ingredients_Clean'),
    ]
)

# 3. K-Fold Training & Prediction Ensembling
min_class_count = np.min(np.bincount(y_train))
n_splits = min(5, max(2, min_class_count))

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_preds_proba = np.zeros((len(test_df), len(label_encoder.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
  X_tr, y_tr = X_train_raw.iloc[train_idx], y_train[train_idx]
  X_val, y_val = X_train_raw.iloc[val_idx], y_train[val_idx]

  # Fit preprocessor
  X_tr_proc = preprocessor.fit_transform(X_tr)
  X_test_proc = preprocessor.transform(X_test_raw)

  # Train multiple models
  model_lgb = LGBMClassifier(
      n_estimators=150,
      learning_rate=0.05,
      max_depth=6,
      num_leaves=31,
      random_state=42 + fold,
      verbose=-1,
  )
  model_rf = RandomForestClassifier(
      n_estimators=200, max_depth=12, random_state=42 + fold, n_jobs=-1
  )
  model_lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42 + fold)

  model_lgb.fit(X_tr_proc.astype(np.float32), y_tr)
  model_rf.fit(X_tr_proc, y_tr)
  model_lr.fit(X_tr_proc, y_tr)

  # Average predicted probabilities across ensemble models
  fold_proba = (
      model_lgb.predict_proba(X_test_proc.astype(np.float32)) * 0.45
      + model_rf.predict_proba(X_test_proc) * 0.35
      + model_lr.predict_proba(X_test_proc) * 0.20
  )

  test_preds_proba += fold_proba / n_splits

# 4. Generate Final Predictions & Output File
final_preds_indices = np.argmax(test_preds_proba, axis=1)
final_preds_labels = label_encoder.inverse_transform(final_preds_indices)

submission_df = pd.DataFrame(
    {'id': np.arange(len(final_preds_labels)), 'Rating': final_preds_labels}
)

submission_df.to_csv('predictions.csv', index=False)