```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/paediatric-investigation-plan/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/paediatric-investigation-plan/data/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Decision type"
    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

    y_raw = train_df[target_col].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    feature_cols = [c for c in train_df.columns if c != target_col]

    text_cols = []
    num_cols = []

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            num_cols.append(c)
        else:
            try:
                parsed = pd.to_datetime(train_df[c], errors='coerce')
                if parsed.notna().sum() > 0.5 * len(train_df):
                    for df in [train_df, test_df]:
                        dt = pd.to_datetime(df[c], errors='coerce')
                        df[f'{c}_year'] = dt.dt.year.fillna(-1)
                        df[f'{c}_month'] = dt.dt.month.fillna(-1)
                        df[f'{c}_day'] = dt.dt.day.fillna(-1)
                        df[f'{c}_dayofweek'] = dt.dt.dayofweek.fillna(-1)
                    num_cols.extend([f'{c}_year', f'{c}_month', f'{c}_day', f'{c}_dayofweek'])
                else:
                    text_cols.append(c)
            except Exception:
                text_cols.append(c)

    def build_text(df, cols):
        if not cols:
            return pd.Series(["document"] * len(df), index=df.index)
        text_series = pd.Series([""] * len(df), index=df.index)
        for c in cols:
            text_series = text_series + f" {c} " + df[c].fillna("").astype(str)
        return text_series

    train_text = build_text(train_df, text_cols)
    test_text = build_text(test_df, text_cols)

    cat_dense_train = []
    cat_dense_test = []
    for c in text_cols:
        freq = train_df[c].value_counts().to_dict()
        cat_dense_train.append(train_df[c].map(freq).fillna(0).values.reshape(-1, 1))
        cat_dense_test.append(test_df[c].map(freq).fillna(0).values.reshape(-1, 1))

    num_train_list = []
    num_test_list =