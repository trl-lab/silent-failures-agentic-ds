import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/phoneme/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/phoneme/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "class"
X = train_df.drop(columns=[target_col])
y = train_df[target_col]

X_test = test_df.drop(columns=[target_col], errors='ignore')
X_test = X_test[X.columns]

for col in X.columns:
    if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[col] = oe.fit_transform(X[[col]].astype(str))
        X_test[col] = oe.transform(X_test[[col]].astype(str))

le = LabelEncoder()
y_encoded = le.fit_transform(y)

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
    X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        random_state=42 + fold,
        verbosity=0,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        random_seed=42 + fold,
        verbose=False
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    
    fold_pred = (
        lgb_model.predict_proba(X_test)[:, 1] +
        xgb_model.predict_proba(X_test)[:, 1] +
        cat_model.predict_proba(X_test)[:, 1]
    ) / 3.0
    
    test_preds += fold_pred / n_splits

final_encoded = (test_preds >= 0.5).astype(int)
final_labels = le.inverse_transform(final_encoded)

submission = pd.DataFrame({
    'id': np.arange(len(X_test)),
    'class': final_labels
})

submission.to_csv('predictions.csv', index=False)