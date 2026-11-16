import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 1. Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/openml/vehicle/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/vehicle/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
feature_cols = [c for c in train_df.columns if c != target_col]

X_train_raw = train_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()

# Target encoding
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].astype(str))
n_classes = len(le.classes_)

# 2. Preprocessing & Feature Engineering
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feature_cols)
X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=feature_cols)

def add_features(df):
    df_feat = df.copy()
    # Aspect ratio and shape ratios
    df_feat['aspect_ratio_mult'] = df['PR.AXIS_ASPECT_RATIO'] * df['MAX.LENGTH_ASPECT_RATIO']
    df_feat['var_ratio'] = (df['SCALED_VARIANCE_MAJOR'] + 1e-5) / (df['SCALED_VARIANCE_MINOR'] + 1e-5)
    df_feat['skew_ratio'] = (df['SKEWNESS_ABOUT_MAJOR'] + 1e-5) / (df['SKEWNESS_ABOUT_MINOR'] + 1e-5)
    df_feat['kurt_ratio'] = (df['KURTOSIS_ABOUT_MAJOR'] + 1e-5) / (df['KURTOSIS_ABOUT_MINOR'] + 1e-5)
    df_feat['circ_compact'] = df['CIRCULARITY'] * df['COMPACTNESS']
    return df_feat

X_train = add_features(X_train_imp)
X_test = add_features(X_test_imp)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Cross-validated Ensemble Modeling
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_total = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    X_tr_sc, X_va_sc = X_train_scaled[train_idx], X_train_scaled[val_idx]
    
    # Model 1: LightGBM
    lgb = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    lgb.fit(X_tr, y_tr)
    
    # Model 2: XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=42 + fold
    )
    xgb.fit(X_tr, y_tr)
    
    # Model 3: CatBoost
    cat = CatBoostClassifier(
        iterations=300,
        learning_rate=0.04,
        depth=5,
        verbose=False,
        random_seed=42 + fold
    )
    cat.fit(X_tr, y_tr)
    
    # Model 4: Extra Trees
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42 + fold,
        n_jobs=-1
    )
    et.fit(X_tr_sc, y_tr)
    
    # Model 5: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42 + fold,
        n_jobs=-1
    )
    rf.fit(X_tr_sc, y_tr)
    
    fold_preds = (
        0.25 * lgb.predict_proba(X_test) +
        0.25 * xgb.predict_proba(X_test) +
        0.25 * cat.predict_proba(X_test) +
        0.15 * et.predict_proba(X_test_scaled) +
        0.10 * rf.predict_proba(X_test_scaled)
    )
    
    test_preds_total += fold_preds / n_splits

# 4. Generate Final Predictions
final_pred_classes = np.argmax(test_preds_total, axis=1)
final_labels = le.inverse_transform(final_pred_classes)

submission = pd.DataFrame({
    'id': np.arange(len(final_labels)),
    'class': final_labels
})

submission.to_csv('predictions.csv', index=False)