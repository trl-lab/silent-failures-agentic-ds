import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/vehicle/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/vehicle/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
y_raw = train_df[target_col].values

X_train = train_df.drop(columns=[target_col])
X_test = test_df.drop(columns=[target_col]) if target_col in test_df.columns else test_df.copy()

le = LabelEncoder()
y = le.fit_transform(y_raw)
num_classes = len(le.classes_)

n_train = len(X_train)
df_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)

cat_cols = df_all.select_dtypes(exclude=[np.number]).columns.tolist()
if cat_cols:
    df_all = pd.get_dummies(df_all, columns=cat_cols, dummy_na=True)

imputer = SimpleImputer(strategy='median')
df_all_imputed = pd.DataFrame(imputer.fit_transform(df_all), columns=df_all.columns)

X_all_num = df_all_imputed.values
row_mean = X_all_num.mean(axis=1, keepdims=True)
row_std = X_all_num.std(axis=1, keepdims=True)
row_min = X_all_num.min(axis=1, keepdims=True)
row_max = X_all_num.max(axis=1, keepdims=True)

X_all_feat = np.hstack([X_all_num, row_mean, row_std, row_min, row_max])

X_tr = X_all_feat[:n_train]
X_te = X_all_feat[n_train:]

scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_te), num_classes))

def get_models(seed):
    return [
        ('et1', ExtraTreesClassifier(n_estimators=500, max_depth=None, max_features='sqrt', random_state=seed, n_jobs=-1)),
        ('et2', ExtraTreesClassifier(n_estimators=500, min_samples_split=3, max_features=0.7, random_state=seed+1, n_jobs=-1)),
        ('rf1', RandomForestClassifier(n_estimators=500, max_depth=None, max_features='sqrt', random_state=seed+2, n_jobs=-1)),
        ('rf2', RandomForestClassifier(n_estimators=500, min_samples_split=3, max_features=0.7, random_state=seed+3, n_jobs=-1)),
        ('hgb', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, l2_regularization=1.0, random_state=seed+4)),
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=seed+5)),
        ('cb', CatBoostClassifier(iterations=500, learning_rate=0.04, depth=6, verbose=0, random_seed=seed+6)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=seed+7, eval_metric='mlogloss', n_jobs=-1)),
        ('lgb', LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=seed+8, verbose=-1, n_jobs=-1)),
        ('svc', SVC(C=10.0, gamma='scale', probability=True, random_state=seed+9)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed+10, early_stopping=True))
    ]

total_models = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y)):
    X_fold_tr, y_fold_tr = X_tr[train_idx], y[train_idx]
    X_fold_tr_sc = X_tr_scaled[train_idx]
    
    models = get_models(fold * 100)
    
    for name, model in models:
        if name in ['svc', 'mlp']:
            model.fit(X_fold_tr_sc, y_fold_tr)
            preds = model.predict_proba(X_te_scaled)
        else:
            model.fit(X_fold_tr, y_fold_tr)
            preds = model.predict_proba(X_te)
        
        test_preds += preds
        total_models += 1

test_preds /= total_models
final_classes = le.inverse_transform(np.argmax(test_preds, axis=1))

output_df = pd.DataFrame({
    'id': np.arange(len(final_classes)),
    'class': final_classes
})

output_df.to_csv('predictions.csv', index=False)