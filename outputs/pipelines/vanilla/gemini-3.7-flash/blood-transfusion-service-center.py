import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/blood-transfusion-service-center/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/blood-transfusion-service-center/data/test.parquet')

# Encode target: '2' -> 1 (donated), '1' -> 0 (not donated)
y_train = (train_df['Class'].astype(str) == '2').astype(int).values

X_train = train_df.drop(columns=['Class']).copy()
X_test = test_df.copy()

# Feature engineering
def create_features(df):
    df_feat = df.copy()
    # Donation span
    df_feat['time_since_first_minus_recency'] = df_feat['V4'] - df_feat['V1']
    # Donation frequency rate
    df_feat['donation_rate'] = df_feat['V2'] / (df_feat['V4'] + 1.0)
    # Recency to time ratio
    df_feat['recency_ratio'] = df_feat['V1'] / (df_feat['V4'] + 1.0)
    # Log transforms for skewed features
    for col in ['V1', 'V2', 'V3', 'V4']:
        df_feat[f'log_{col}'] = np.log1p(df_feat[col].clip(lower=0))
    return df_feat

X_train_fe = create_features(X_train)
X_test_fe = create_features(X_test)

# Cross-validation ensemble
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

test_preds = np.zeros(len(X_test_fe))

for train_idx, val_idx in skf.split(X_train_fe, y_train):
    X_tr, y_tr = X_train_fe.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train_fe.iloc[val_idx], y_train[val_idx]
    
    # Scale for linear & distance-based models
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_test_fe)
    
    # Logistic Regression
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_tr_scaled, y_tr)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42)
    rf.fit(X_tr, y_tr)
    
    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42)
    et.fit(X_tr, y_tr)
    
    # LightGBM
    lgb = LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr)
    
    # CatBoost
    cb = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
    cb.fit(X_tr, y_tr)
    
    # Fold predictions
    fold_preds = (
        0.20 * lr.predict_proba(X_te_scaled)[:, 1] +
        0.20 * rf.predict_proba(X_test_fe)[:, 1] +
        0.20 * et.predict_proba(X_test_fe)[:, 1] +
        0.20 * lgb.predict_proba(X_test_fe)[:, 1] +
        0.20 * cb.predict_proba(X_test_fe)[:, 1]
    )
    test_preds += fold_preds / skf.n_splits

# Save predictions
predictions_df = pd.DataFrame({
    'id': range(len(test_preds)),
    'Class': test_preds
})

predictions_df.to_csv('predictions.csv', index=False)