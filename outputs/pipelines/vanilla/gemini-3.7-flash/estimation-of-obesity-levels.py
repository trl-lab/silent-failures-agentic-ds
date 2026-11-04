import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

train_path = '/path/to/user/workspace/ds_failures/datasets/uci/estimation-of-obesity-levels/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/estimation-of-obesity-levels/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'NObeyesdad'

# Feature engineering
for df in [train_df, test_df]:
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

X_train = train_df.drop(columns=[target_col]).copy()
y_train_raw = train_df[target_col]
X_test = test_df.copy()

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

num_classes = len(le.classes_)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=num_classes,
        n_estimators=400,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=-1
    )
    clf.fit(X_tr, y_tr)
    test_preds += clf.predict_proba(X_test) / n_splits

final_indices = np.argmax(test_preds, axis=1)
final_predictions = le.inverse_transform(final_indices)

submission = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'NObeyesdad': final_predictions
})
submission.to_csv('predictions.csv', index=False)