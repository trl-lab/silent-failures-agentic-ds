import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/har/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/har/data/test.parquet'

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = 'class'

y_train = train[target_col]
X_train = train.drop(columns=[target_col])

if target_col in test.columns:
    X_test = test.drop(columns=[target_col])
else:
    X_test = test.copy()

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

X_train.columns = [str(col) for col in X_train.columns]
X_test.columns = [str(col) for col in X_test.columns]

X_test = X_test[X_train.columns]

if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
else:
    X_train_imp = X_train.values
    X_test_imp = X_test.values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

clf1 = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
clf2 = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf3 = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
clf4 = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')

clf1.fit(X_train_imp, y_train_encoded)
clf2.fit(X_train_imp, y_train_encoded)
clf3.fit(X_train_scaled, y_train_encoded)
clf4.fit(X_train_imp, y_train_encoded)

p1 = clf1.predict_proba(X_test_imp)
p2 = clf2.predict_proba(X_test_imp)
p3 = clf3.predict_proba(X_test_scaled)
p4 = clf4.predict_proba(X_test_imp)

p_blend = 0.3 * p1 + 0.3 * p2 + 0.2 * p3 + 0.2 * p4
final_preds_encoded = np.argmax(p_blend, axis=1)

final_preds = le.inverse_transform(final_preds_encoded)

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'class': final_preds
})

sub.to_csv('predictions.csv', index=False)