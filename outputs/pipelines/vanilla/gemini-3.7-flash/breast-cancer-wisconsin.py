import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/breast-cancer-wisconsin/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/breast-cancer-wisconsin/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Diagnosis'
X = train_df.drop(columns=[target_col])
y_raw = train_df[target_col]
X_test = test_df.copy()

# Encode target
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Identify feature columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Handle categoricals if any
for col in categorical_cols:
    cat_le = LabelEncoder()
    full_series = pd.concat([X[col], X_test[col]]).astype(str)
    cat_le.fit(full_series)
    X[col] = cat_le.transform(X[col].astype(str))
    X_test[col] = cat_le.transform(X_test[col].astype(str))

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Stratified K-Fold CV and Ensembling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test_imputed))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_imputed, y)):
    X_tr, y_tr = X_imputed.iloc[train_idx], y[train_idx]
    X_va, y_va = X_imputed.iloc[val_idx], y[val_idx]
    
    # Model 1: Logistic Regression with Scaling
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    lr_pipe.fit(X_tr, y_tr)
    pred_lr = lr_pipe.predict_proba(X_test_imputed)[:, 1]
    
    # Model 2: LightGBM
    lgb = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        random_state=42,
        verbose=-1
    )
    lgb.fit(X_tr, y_tr)
    pred_lgb = lgb.predict_proba(X_test_imputed)[:, 1]
    
    # Model 3: CatBoost
    cb = CatBoostClassifier(
        iterations=150,
        learning_rate=0.05,
        depth=4,
        verbose=0,
        random_state=42
    )
    cb.fit(X_tr, y_tr)
    pred_cb = cb.predict_proba(X_test_imputed)[:, 1]
    
    # Model 4: Extra Trees
    et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et.fit(X_tr, y_tr)
    pred_et = et.predict_proba(X_test_imputed)[:, 1]
    
    # Average fold predictions
    fold_pred = 0.35 * pred_lr + 0.25 * pred_lgb + 0.25 * pred_cb + 0.15 * pred_et
    test_preds += fold_pred / skf.n_splits

# Format submission
submission = pd.DataFrame({
    'id': np.arange(len(test_preds)),
    'Diagnosis': test_preds
})

submission.to_csv('predictions.csv', index=False)