import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/electricity/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/electricity/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].copy()
X_test = test_df.copy()

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Preprocessing: convert categorical columns to category dtype or integer
categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Cross-validation and ensemble predictions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_lgb = np.zeros((len(X_test), len(le.classes_)))
test_preds_cat = np.zeros((len(X_test), len(le.classes_)))

cat_indices = [X.columns.get_loc(c) for c in categorical_cols]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]
    
    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    test_preds_lgb += lgb_model.predict_proba(X_test) / skf.n_splits

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        cat_features=cat_indices,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=0
    )
    test_preds_cat += cat_model.predict_proba(X_test) / skf.n_splits

# Blend predictions
final_probs = 0.5 * test_preds_lgb + 0.5 * test_preds_cat
final_preds_encoded = np.argmax(final_probs, axis=1)
final_preds = le.inverse_transform(final_preds_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'class': final_preds
})
output_df.to_csv('predictions.csv', index=False)