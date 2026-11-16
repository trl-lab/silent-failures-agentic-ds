import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/dry-bean-dataset/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/dry-bean-dataset/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Class"
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

def create_features(df):
    df = df.copy()
    df['Area_ConvexArea_ratio'] = df['Area'] / (df['ConvexArea'] + 1e-6)
    df['Major_Minor_ratio'] = df['MajorAxisLength'] / (df['MinorAxisLength'] + 1e-6)
    df['Log_Area'] = np.log1p(df['Area'])
    df['Log_Perimeter'] = np.log1p(df['Perimeter'])
    df['Perimeter_Area_ratio'] = df['Perimeter'] / (np.sqrt(df['Area']) + 1e-6)
    return df

X_fe = create_features(X_imputed)
X_test_fe = create_features(X_test_imputed)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_probs = np.zeros((len(X_test_fe), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_fe, y_encoded)):
    X_train, y_train = X_fe.iloc[train_idx], y_encoded[train_idx]
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    test_probs += lgb_model.predict_proba(X_test_fe) * 0.35
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42 + fold,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    test_probs += xgb_model.predict_proba(X_test_fe) * 0.35
    
    cat_model = CatBoostClassifier(
        iterations=700,
        learning_rate=0.04,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cat_model.fit(X_train, y_train)
    test_probs += cat_model.predict_proba(X_test_fe) * 0.30

final_preds_idx = np.argmax(test_probs, axis=1)
final_preds_labels = le.inverse_transform(final_preds_idx)

submission = pd.DataFrame({
    'id': range(len(test_df)),
    'Class': final_preds_labels
})
submission.to_csv('predictions.csv', index=False)