import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

# 1. Load the data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/adult/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/adult/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 2. Clean target and prepare features
y_raw = train_df['income'].astype(str).str.strip().str.rstrip('.')
y = (y_raw == '>50K').astype(int)

X = train_df.drop(columns=['income']).copy()
X_test = test_df.copy()

categorical_cols = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

for col in categorical_cols:
    X[col] = X[col].astype(str).str.strip().replace('?', 'Unknown')
    X_test[col] = X_test[col].astype(str).str.strip().replace('?', 'Unknown')
    X[col] = X[col].astype('category')
    X_test[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories)

# 3. Train models using Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)]
    )
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

# 4. Generate final class predictions
pred_labels = np.where(test_preds >= 0.5, '>50K', '<=50K')

# 5. Save predictions
submission_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'income': pred_labels
})
submission_df.to_csv('predictions.csv', index=False)