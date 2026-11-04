import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/steel-plates-fault/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/steel-plates-fault/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 2. Prepare features and target
target_col = 'target'
X = train_df.drop(columns=[target_col])
y_raw = train_df[target_col]

# Match columns for test
X_test = test_df.copy()
if target_col in X_test.columns:
    X_test = X_test.drop(columns=[target_col])

# Encode target
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(le.classes_)

# 3. Stratified K-Fold Cross-Validation Ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_lgb = np.zeros((len(X_test), n_classes))
test_preds_xgb = np.zeros((len(X_test), n_classes))
test_preds_cat = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y[val_idx]
    
    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_train_f, y_train_f)
    test_preds_lgb += lgb_model.predict_proba(X_test) / n_splits
    
    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42 + fold,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train_f, y_train_f)
    test_preds_xgb += xgb_model.predict_proba(X_test) / n_splits
    
    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(X_train_f, y_train_f)
    test_preds_cat += cat_model.predict_proba(X_test) / n_splits

# 4. Blend predictions
blended_probs = (test_preds_lgb + test_preds_xgb + test_preds_cat) / 3.0
final_preds_encoded = np.argmax(blended_probs, axis=1)
final_preds = le.inverse_transform(final_preds_encoded)

# 5. Save predictions to CSV
submission_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'target': final_preds
})

submission_df.to_csv('predictions.csv', index=False)