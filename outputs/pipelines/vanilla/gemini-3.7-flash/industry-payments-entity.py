import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-entity/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/industry-payments-entity/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'entityPayment'
y_train = train_df[target_col].values

# Feature definitions
categorical_features = ['sourceID', 'reportingCompany', 'entityCountry', 'paymentType']
text_features = ['entityName', 'nrgiNotes', 'notes']
numeric_features = ['reportYear']

# Preprocessing
def prepare_features(df):
    df_feat = df.copy()
    for col in categorical_features:
        df_feat[col] = df_feat[col].fillna('missing').astype(str)
    for col in text_features:
        df_feat[col] = df_feat[col].fillna('').astype(str)
        df_feat[f'{col}_len'] = df_feat[col].apply(len)
    for col in numeric_features:
        df_feat[col] = df_feat[col].fillna(df_feat[col].median()).astype(float)
    return df_feat

train_processed = prepare_features(train_df)
test_processed = prepare_features(test_df)

feature_cols = categorical_features + text_features + numeric_features + [f'{col}_len' for col in text_features]
X_train = train_processed[feature_cols]
X_test = test_processed[feature_cols]

# Cross-validation training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42 + fold,
        cat_features=categorical_features,
        text_features=text_features,
        verbose=0
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )

    test_preds += model.predict(X_test) / kf.n_splits

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'entityPayment': test_preds
})
output_df.to_csv('predictions.csv', index=False)