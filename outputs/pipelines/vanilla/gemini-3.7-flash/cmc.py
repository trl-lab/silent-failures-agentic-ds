import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/cmc/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/cmc/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Contraceptive_method_used'

X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].copy()
X_test = test_df.copy()

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
n_classes = len(target_encoder.classes_)

# Identify feature types
cat_cols = [c for c in X.columns if X[c].dtype.name == 'category' or X[c].dtype == 'object']
num_cols = [c for c in X.columns if c not in cat_cols]

# Cast categorical columns to string to handle any unseen categories uniformly
for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_proc = preprocessor.fit_transform(X)
X_test_proc = preprocessor.transform(X_test)

# Stratified K-Fold Ensemble
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_proba = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_proc, y_encoded)):
    X_tr, y_tr = X_proc[train_idx], y_encoded[train_idx]
    X_va, y_va = X_proc[val_idx], y_encoded[val_idx]
    
    # Define models
    lgb = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42 + fold,
        verbose=-1
    )
    
    cb = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=5,
        random_seed=42 + fold,
        verbose=0
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1
    )
    
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1
    )
    
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42 + fold
    )
    
    models = [lgb, cb, rf, et, hgb]
    
    for model in models:
        model.fit(X_tr, y_tr)
        test_preds_proba += model.predict_proba(X_test_proc) / (n_splits * len(models))

# Final predictions
final_pred_indices = np.argmax(test_preds_proba, axis=1)
final_predictions = target_encoder.inverse_transform(final_pred_indices)

# Save output
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    target_col: final_predictions
})
output_df.to_csv('predictions.csv', index=False)