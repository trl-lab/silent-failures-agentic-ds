import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/estimation-of-obesity-levels/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/estimation-of-obesity-levels/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "NObeyesdad"

y = train[target_col]
X = train.drop(columns=[target_col])
X_test = test.copy()

if "id" in X.columns:
    X = X.drop(columns=["id"])
if "id" in X_test.columns:
    X_test = X_test.drop(columns=["id"])

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

combined = pd.concat([X, X_test], axis=0, ignore_index=True)

combined['BMI'] = combined['Weight'] / (combined['Height'] ** 2)
combined['Weight_Height_ratio'] = combined['Weight'] / combined['Height']
combined['Age_BMI'] = combined['Age'] * combined['BMI']
combined['CH2O_Weight'] = combined['CH2O'] / combined['Weight']
combined['FAF_Age'] = combined['FAF'] / (combined['Age'] + 1e-5)
combined['FCVC_NCP'] = combined['FCVC'] / (combined['NCP'] + 1e-5)
combined['TUE_Age'] = combined['TUE'] / (combined['Age'] + 1e-5)

ordinal_maps = {
    'family_history_with_overweight': {'no': 0, 'yes': 1},
    'FAVC': {'no': 0, 'yes': 1},
    'SMOKE': {'no': 0, 'yes': 1},
    'SCC': {'no': 0, 'yes': 1},
    'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'Gender': {'Female': 0, 'Male': 1}
}

for col, m in ordinal_maps.items():
    if col in combined.columns:
        combined[col] = combined[col].map(m).fillna(-1)

cat_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    combined[col] = combined[col].astype('category').cat.codes

X_proc = combined.iloc[:len(X)].copy().apply(pd.to_numeric)
X_test_proc = combined.iloc[len(X):].copy().apply(pd.to_numeric)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros((len(X_test_proc), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y_encoded)):
    X_tr, y_tr = X_proc.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X_proc.iloc[val_idx], y_encoded[val_idx]
    
    lgbm = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        objective='multiclass',
        num_class=num_classes,
        verbose=-1
    )
    lgbm.fit(X_tr, y_tr)
    
    xgb = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss'
    )
    xgb.fit(X_tr, y_tr)
    
    cat = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        loss_function='MultiClass',
        verbose=False
    )
    cat.fit(X_tr, y_tr)
    
    fold_preds = (
        0.4 * lgbm.predict_proba(X_test_proc) +
        0.3 * xgb.predict_proba(X_test_proc) +
        0.3 * cat.predict_proba(X_test_proc)
    )
    test_preds += fold_preds / n_splits

final_class_indices = test_preds.argmax(axis=1)
final_labels = le.inverse_transform(final_class_indices)

sub = pd.DataFrame({
    'id': np.arange(len(final_labels)),
    'NObeyesdad': final_labels
})
sub.to_csv('predictions.csv', index=False)