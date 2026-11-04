import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/strable/food-enforcement/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/food-enforcement/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'classification'
y_labels = train_df[target_col].astype(str)
le = LabelEncoder()
y = le.fit_transform(y_labels)
num_classes = len(le.classes_)

X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

def preprocess_dates(df):
    df_clean = df.copy()
    date_cols = [c for c in df_clean.columns if 'date' in c.lower()]
    for c in date