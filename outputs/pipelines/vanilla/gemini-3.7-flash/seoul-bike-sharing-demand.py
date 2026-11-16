import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/seoul-bike-sharing-demand/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/seoul-bike-sharing-demand/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Rented Bike Count'

# Prepare target variable
y_raw = train_df[target_col].astype(str)
target_mapping = {'No': 0, 'Yes': 1, '0': 0, '1': 1, 'no': 0, 'yes': 1}
if set(y_raw.unique()).issubset(set(target_mapping.keys())):
    y_train = y_raw.map(target_mapping).astype(int)
else:
    le_target = LabelEncoder()
    y_train = pd.Series(le_target.fit_transform(y_raw))

# Features
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Feature engineering on Date
for df in [X_train, X_test]:
    if 'Date' in df.columns:
        date_series = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df['Month'] = date_series.dt.month.fillna(-1).astype(int)
        df['Day'] = date_series.dt.day.fillna(-1).astype(int)
        df['DayOfWeek'] = date_series.dt.dayofweek.fillna(-1).astype(int)
        df['Year'] = date_series.dt.year.fillna(-1).astype(int)
        df.drop(columns=['Date'], inplace=True)

# Encode categorical variables
for col in X_train.columns:
    if X_train[col].dtype == 'object' or str(X_train[col].dtype) == 'category':
        le = LabelEncoder()
        combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Fill missing values
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_train.median(numeric_only=True))

# Cross-validation and ensemble modeling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=-1
    )