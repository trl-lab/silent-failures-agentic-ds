import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/madelon/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/madelon/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Class"

X_train = train_df.drop(columns=[target_col])
y_train_raw = train_df[target_col]

X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

if len(cat_cols) > 0:
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

num_imputer = SimpleImputer(strategy='median')
X_train_imp = num_imputer.fit_transform(X_train)
X_test_imp = num_imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

selector_model = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
selector_model.fit(X_train_scaled, y_train)

importances = selector_model.feature_importances_
top_indices = np.argsort(importances)[::-1][:25]

X_train_sel = X_train_scaled[:, top_indices]
X_test_sel = X_test_scaled[:, top_indices]

clf1 = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)
clf2 = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
clf3 = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')
clf4 = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
clf5 = cb.CatBoostClassifier(iterations=400, depth=5, learning_rate=0.05, random_state=42, verbose=0)
clf6 = SVC(C=10, gamma='scale', probability=True, random_state=42)

ensemble = VotingClassifier(
    estimators=[
        ('et', clf1),
        ('rf', clf2),
        ('xgb', clf3),
        ('lgb', clf4),
        ('cb', clf5),
        ('svc', clf6)
    ],
    voting='soft'
)

ensemble.fit(X_train_sel, y_train)
preds_encoded = ensemble.predict(X_test_sel)

preds = label_encoder.inverse_transform(preds_encoded)

sub = pd.DataFrame({
    'id': np.arange(len(preds)),
    'Class': preds
})

sub.to_csv('predictions.csv', index=False)