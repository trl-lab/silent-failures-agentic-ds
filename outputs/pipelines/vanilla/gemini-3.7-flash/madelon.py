import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/madelon/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/madelon/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Class'

X = train_df.drop(columns=[target_col]).copy()
y_raw = train_df[target_col].values
X_test = test_df.copy()

if 'id' in X_test.columns and 'id' not in X.columns:
    X_test = X_test.drop(columns=['id'])

if X.isna().sum().sum() > 0:
    medians = X.median()
    X = X.fillna(medians)
    X_test = X_test.fillna(medians)

# Map target classes to 0 and 1
classes = np.unique(y_raw)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for i, c in enumerate(classes)}
y = np.array([class_to_idx[c] for c in y_raw])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
selector.fit(X_scaled, y)
sfm = SelectFromModel(selector, threshold='1.25*median', prefit=True)

X_sel = sfm.transform(X_scaled)
X_test_sel = sfm.transform(X_test_scaled)

if X_sel.shape[1] < 10:
    X_sel = X_scaled
    X_test_sel = X_test_scaled

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_proba = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_sel, y)):
    X_tr, y_tr = X_sel[train_idx], y[train_idx]

    clf_et = ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_split=4, random_state=42 + fold, n_jobs=-1)
    clf_et.fit(X_tr, y_tr)

    clf_cb = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, verbose=0, random_seed=42 + fold)
    clf_cb.fit(X_tr, y_tr)

    clf_lgb = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42 + fold, verbose=-1)
    clf_lgb.fit(X_tr, y_tr)

    p_et = clf_et.predict_proba(X_test_sel)[:, 1]
    p_cb = clf_cb.predict_proba(X_test_sel)[:, 1]
    p_lgb = clf_lgb.predict_proba(X_test_sel)[:, 1]

    fold_pred = 0.45 * p_et + 0.30 * p_cb + 0.25 * p_lgb
    test_preds_proba += fold_pred / skf.n_splits

pred_indices = (test_preds_proba >= 0.5).astype(int)
final_preds = [idx_to_class[i] for i in pred_indices]

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Class': final_preds
})

output_df.to_csv('predictions.csv', index=False)