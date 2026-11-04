import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/transmission-lines/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/transmission-lines/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'VOLT_CLASS'
y_raw = train_df[target_col]
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)

# Combine for uniform feature engineering
combined_df = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# Date processing
for date_col in ['SOURCEDATE', 'VAL_DATE']:
    if date_col in combined_df.columns:
        dt_series = pd.to_datetime(combined_df[date_col], errors='coerce')
        combined_df[f'{date_col}_year'] = dt_series.dt.year.fillna(-1).astype(int)
        combined_df[f'{date_col}_month'] = dt_series.dt.month.fillna(-1).astype(int)
        combined_df[f'{date_col}_day'] = dt_series.dt.day.fillna(-1).astype(int)
        combined_df[f'{date_col}_dayofweek'] = dt_series.dt.dayofweek.fillna(-1).astype(int)
        combined_df[f'{date_col}_epoch'] = dt_series.astype('int64') // 10**9
        combined_df[f'{date_col}_epoch'] = combined_df[f'{date_col}_epoch'].replace(-9223372037, np.nan)
        combined_df.drop(columns=[date_col], inplace=True)

# Categorical columns encoding
cat_cols = ['TYPE', 'STATUS', 'VAL_METHOD', 'OWNER', 'INFERRED']
for col in cat_cols:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].astype(str).fillna('missing')
        freq = combined_df[col].value_counts()
        combined_df[f'{col}_freq'] = combined_df[col].map(freq)
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
        combined_df[col] = combined_df[col].astype('category')

# Separate train and test features
n_train = len(train_df)
X = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# Cross-validated training and test prediction
n_classes = len(le_target.classes_)
test_preds_proba = np.zeros((len(test_df), n_classes))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    test_preds_proba += model.predict_proba(X_test) / skf.n_splits

# Generate final predictions
best_preds = np.argmax(test_preds_proba, axis=1)
predicted_classes = le_target.inverse_transform(best_preds)

# Output predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(predicted_classes)),
    'VOLT_CLASS': predicted_classes
})
submission_df.to_csv('predictions.csv', index=False)