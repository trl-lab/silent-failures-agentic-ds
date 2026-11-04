import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

# Load data
train_path = "/path/to/user/workspace/ds_failures/datasets/openml/MiceProtein/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/MiceProtein/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "class"
X_train = train_df.drop(columns=[target_col]).copy()
y_train = train_df[target_col].copy()

X_test = test_df.drop(columns=[target_col], errors="ignore").copy()

# Encode target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
n_classes = len(le.classes_)

# Separate categorical and numerical features
cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

# Preprocess categorical features
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

if cat_cols:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32)
    X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])

# Preprocess numeric features
for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').astype(np.float32)
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').astype(np.float32)

# Imputed versions for algorithms requiring complete data
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# K-Fold Ensemble
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_probs = np.zeros((len(X_test), n_classes), dtype=np.float64)

for train_idx, val_idx in skf.split(X_train, y_train_encoded):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_encoded[train_idx]
    X_tr_imp = X_train_imp.iloc[train_idx]

    # 1. LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    model_lgb.fit(X_tr, y_tr)
    test_probs += model_lgb.predict_proba(X_test)

    # 2. CatBoost
    model_cb = cb.CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    model_cb.fit(X_tr, y_tr)
    test_probs += model_cb.predict_proba(X_test)

    # 3. XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    model_xgb.fit(X_tr, y_tr)
    test_probs += model_xgb.predict_proba(X_test)

    # 4. ExtraTrees
    model_et = ExtraTreesClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model_et.fit(X_tr_imp, y_tr)
    test_probs += model_et.predict_proba(X_test_imp)

    # 5. RandomForest
    model_rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model_rf.fit(X_tr_imp, y_tr)
    test_probs += model_rf.predict_proba(X_test_imp)

# Final predictions
final_pred_indices = np.argmax(test_probs, axis=1)
preds = le.inverse_transform(final_pred_indices)

# Save output
df_out = pd.DataFrame({
    'id': np.arange(len(preds)),
    'class': preds
})
df_out.to_csv('predictions.csv', index=False)