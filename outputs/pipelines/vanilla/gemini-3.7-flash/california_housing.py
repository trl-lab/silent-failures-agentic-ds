import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/california_housing/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/california_housing/data/test.parquet')

# Feature Engineering
def engineer_features(df):
    df = df.copy()
    df['rooms_per_household'] = df['totalRooms'] / (df['households'] + 1e-5)
    df['bedrooms_per_room'] = df['totalBedrooms'] / (df['totalRooms'] + 1e-5)
    df['population_per_household'] = df['population'] / (df['households'] + 1e-5)
    df['bedrooms_per_household'] = df['totalBedrooms'] / (df['households'] + 1e-5)
    return df

X_train = engineer_features(train_df.drop('medianHouseValue', axis=1))
y_train = train_df['medianHouseValue']
X_test = engineer_features(test_df)

# Cross-validation and ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    test_preds_lgb += lgb_model.predict(X_test) / kf.n_splits
    
    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=False
    )
    cat_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=100,
        verbose=False
    )
    test_preds_cat += cat_model.predict(X_test) / kf.n_splits

# Blend predictions
final_predictions = 0.5 * test_preds_lgb + 0.5 * test_preds_cat

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'medianHouseValue': final_predictions
})
output_df.to_csv('predictions.csv', index=False)