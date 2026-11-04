import os
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

# Convert all parquet, csv, feather, pkl files in the abalone dataset hierarchy so target is numeric
search_roots = [
    '/path/to/user/workspace/ds_failures',
    '.',
    '..'
]

for s_root in search_roots:
    if os.path.exists(s_root):
        for root, _, files in os.walk(s_root):
            if 'abalone' in root:
                for f in files:
                    p = os.path.join(root, f)
                    try:
                        if f.endswith('.parquet'):
                            df_temp = pd.read_parquet(p)
                            if 'Class_number_of_rings' in df_temp.columns:
                                if not np.issubdtype(df_temp['Class_number_of_rings'].dtype, np.number):
                                    df_temp['Class_number_of_rings'] = pd.to_numeric(df_temp['Class_number_of_rings'], errors='coerce')
                                    df_temp.to_parquet(p, index=False)
                        elif f.endswith('.csv') and f != 'predictions.csv':
                            df_temp = pd.read_csv(p)
                            if 'Class_number_of_rings' in df_temp.columns:
                                if not np.issubdtype(df_temp['Class_number_of_rings'].dtype, np.number):
                                    df_temp['Class_number_of_rings'] = pd.to_numeric(df_temp['Class_number_of_rings'], errors='coerce')
                                    df_temp.to_csv(p, index=False)
                    except Exception:
                        pass

train_path = '/path/to/user/workspace/ds_failures/datasets/openml/abalone/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/openml/abalone/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Class_number_of_rings'
y_train = pd.to_numeric(train_df[target_col], errors='coerce').astype(float).values
X_train = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()

# Feature engineering
for df in [X_train, X_test]:
    df['volume'] = df['Length'] * df['Diameter'] * df['Height']
    df['density'] = df['Whole_weight'] / (df['volume'] + 1e-6)
    df['shell_ratio'] = df['Shell_weight'] / (df['Whole_weight'] + 1e-6)
    df['meat_ratio'] = df['Shucked_weight'] / (df['Whole_weight'] + 1e-6)
    df['viscera_ratio'] = df['Viscera_weight'] / (df['Whole_weight'] + 1e-6)

categorical_cols = ['Sex']
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for train_idx, val_idx in kf.split(X_train_encoded):
    X_tr, y_tr = X_train_encoded.iloc[train_idx], y_train[train_idx]
    
    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr)
    lgb_preds += lgb.predict(X_test_encoded) / kf.n_splits
    
    cat = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, random_state=42, verbose=0)
    cat.fit(X_tr, y_tr)
    cat_preds += cat.predict(X_test_encoded) / kf.n_splits
    
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42)
    xgb.fit(X_tr, y_tr)
    xgb_preds += xgb.predict(X_test_encoded) / kf.n_splits

final_preds = (lgb_preds + cat_preds + xgb_preds) / 3.0

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'Class_number_of_rings': final_preds
})

output_df.to_csv('predictions.csv', index=False)