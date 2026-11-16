import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/mobile-home-parks/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/mobile-home-parks/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'SIZE'

# Encode target
target_le = LabelEncoder()
y = target_le.fit_transform(train_df[target_col].astype(str))

# Combine for uniform preprocessing
n_train = len(train_df)
combined_df = pd.concat([train_df.drop(columns=[target_col]), test_df], axis=0).reset_index(drop=True)

# Feature engineering
date_cols = ['SOURCEDATE', 'VAL_DATE']
for col in date_cols:
    if col in combined_df.columns:
        parsed_date = pd.to_datetime(combined_df[col], errors='coerce')
        combined_df[f'{col}_YEAR'] = parsed_date.dt.year
        combined_df[f'{col}_MONTH'] = parsed_date.dt.month
        combined_df.drop(columns=[col], inplace=True)

# Identify categorical and numeric columns
cat_cols = []
num_cols = []

for col in combined_df.columns:
    if combined_df[col].dtype == 'object' or combined_df[col].dtype.name == 'category':
        cat_cols.append(col)
        combined_df[col] = combined_df[col].astype(str).fillna('MISSING')
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
    else:
        num_cols.append(col)
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

X = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()

# Stratified K-Fold Training
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

num_classes = len(target_le.classes_)
test_preds = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_val = X.iloc[val_idx], y[val_idx]

    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        n_estimators=500,
        learning_rate=0.05,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    test_preds += clf.predict_proba(X_test) / n_splits

final_class_indices = np.argmax(test_preds, axis=1)
final_predictions = target_le.inverse_transform(final_class_indices)

submission = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'SIZE': final_predictions
})

submission.to_csv('predictions.csv', index=False)