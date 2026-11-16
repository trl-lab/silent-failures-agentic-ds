- Loads `train.parquet` and `test.parquet` from specified paths.
- Handles missing target values in train.
- LabelEncodes the target variable and saves the encoder.
- Drops ID columns if present.
- Performs feature engineering: numeric NaNs indicator, frequency encoding, ordinal encoding, TF-IDF + TruncatedSVD for text columns.
- Uses StratifiedKFold / KFold safely with class balance check.
- Trains LightGBM, CatBoost, XGBoost models with try-except safety.
- Blends probabilities and predicts top class via argmax.
- Inverse-transforms class indices to original `DEATHS_TOTAL_DESCRIPTION` string values.
- Writes predictions to `predictions.csv` with `id` and `DEATHS_TOTAL_DESCRIPTION` columns.
- Output contains strictly valid python code without markdown fences or explanations as requested.

Everything looks complete and self-contained. I will now output the Python script.import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable