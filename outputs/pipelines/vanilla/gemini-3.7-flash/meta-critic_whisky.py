import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load datasets
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/meta-critic_whisky/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/meta-critic_whisky/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Meta Critic'
y_train = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])


def extract_features(df):
    df = df.copy()

    # Cost feature: ordinal mapping
    cost_map = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, '$$$$$': 5, '$$$$$+': 6}
    df['cost_num'] = df['Cost'].map(cost_map).fillna(0)

    # Text features from Whisky name
    def extract_age(name):
        if not isinstance(name, str):
            return np.nan
        match = re.search(r'(\d+)\s*(?:yo|year|yr)', name, re.IGNORECASE)
        if match:
            age = float(match.group(1))
            if 3 <= age <= 80:
                return age
        return np.nan

    def extract_year(name):
        if not isinstance(name, str):
            return np.nan
        match = re.search(r'\b(19\d{2}|20\d{2})\b', name)
        if match:
            year = float(match.group(1))
            if 1900 <= year <= 2030:
                return year
        return np.nan

    df['whisky_name'] = df['Whisky'].fillna('').astype(str)
    df['extracted_age'] = df['whisky_name'].apply(extract_age).fillna(-1)
    df['extracted_year'] = df['whisky_name'].apply(extract_year).fillna(-1)
    df['name_len'] = df['whisky_name'].apply(len)
    df['name_word_count'] = df['whisky_name'].apply(lambda x: len(x.split()))

    # Categorical columns
    cat_cols = ['Cost', 'Class', 'Super Cluster', 'Cluster', 'Country', 'Type']
    for col in cat_cols:
        df[col] = df[col].fillna('Missing').astype(str)

    return df


X_train_full = extract_features(train_df)
X_test_full = extract_features(test_df)

cat_cols = ['Cost', 'Class', 'Super Cluster', 'Cluster', 'Country', 'Type']
num_cols = ['cost_num', 'extracted_age', 'extracted_year', 'name_len', 'name_word_count']
text_col = 'whisky_name'

# Feature preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('text', TfidfVectorizer(ngram_range=(1, 2), max_features=300), text_col),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train_full)
X_test_proc = preprocessor.transform(X_test_full)

# Cross-validated ensemble modeling
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

models = {
    'ridge': Ridge(alpha=1.0),
    'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'et': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'lgb': LGBMRegressor(n_estimators=200, learning_rate=0.03, max_depth=5, num_leaves=15, random_state=42, verbose=-1),
    'cb': CatBoostRegressor(iterations=300, learning_rate=0.03, depth=5, random_seed=42, verbose=0)
}

test_preds = np.zeros(len(test_df))
total_weights = 0

for name, model in models.items():
    model_test_pred = np.zeros(len(test_df))
    for train_idx, val_idx in kf.split(X_train_proc):
        X_tr, y_tr = X_train_proc[train_idx], y_train[train_idx]
        model.fit(X_tr, y_tr)
        model_test_pred += model.predict(X_test_proc) / n_splits
    test_preds += model_test_pred
    total_weights += 1

final_predictions = test_preds / total_weights

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Meta Critic': final_predictions
})
output_df.to_csv('predictions.csv', index=False)