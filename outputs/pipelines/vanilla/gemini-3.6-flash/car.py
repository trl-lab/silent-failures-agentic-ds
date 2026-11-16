import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/car/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/car/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'class'
X_train_raw = train_df.drop(columns=[target_col])
y_train_raw = train_df[target_col]

if target_col in test_df.columns:
    X_test_raw = test_df.drop(columns=[target_col])
else:
    X_test_raw = test_df.copy()

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw.astype(str))

ordinal_maps = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3, '5': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2}
}

feature_cols = list(X_train_raw.columns)

def preprocess(df):
    df_out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        col_str = df[col].astype(str)
        if col in ordinal_maps:
            df_out[col + '_ord'] = col_str.map(ordinal_maps[col]).fillna(-1).astype(float)
        df_out[col] = col_str
    return df_out

X_train_proc = preprocess(X_train_raw)
X_test_proc = preprocess(X_test_raw)

cat_cols = [c for c in X_train_proc.columns if X_train_proc[c].dtype == 'object']

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_encoded = X_train_proc.copy()
X_test_encoded = X_test_proc.copy()

X_train_encoded[cat_cols] = oe.fit_transform(X_train_proc[cat_cols].astype(str))
X_test_encoded[cat_cols] = oe.transform(X_test_proc[cat_cols].astype(str))

num_classes = len(np.unique(y_train))

models = [
    CatBoostClassifier(verbose=0, random_state=42, iterations=500),
    LGBMClassifier(random_state=42, verbose=-1, n_estimators=300),
    XGBClassifier(random_state=42, eval_metric='mlogloss', n_estimators=300),
    RandomForestClassifier(n_estimators=300, random_state=42),
    ExtraTreesClassifier(n_estimators=300, random_state=42),
    HistGradientBoostingClassifier(random_state=42, max_iter=300)
]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_list = []

for model in models:
    model_test_probs = np.zeros((len(X_test_encoded), num_classes))
    for train_idx, val_idx in skf.split(X_train_encoded, y_train):
        X_tr = X_train_encoded.iloc[train_idx]
        y_tr = y_train[train_idx]
        
        model.fit(X_tr, y_tr)
        model_test_probs += model.predict_proba(X_test_encoded) / skf.n_splits
    test_preds_list.append(model_test_probs)

final_probs = np.mean(test_preds_list, axis=0)
pred_labels = np.argmax(final_probs, axis=1)
pred_class = le.inverse_transform(pred_labels)

submission = pd.DataFrame({
    'id': np.arange(len(pred_class)),
    'class': pred_class
})

submission.to_csv('predictions.csv', index=False)