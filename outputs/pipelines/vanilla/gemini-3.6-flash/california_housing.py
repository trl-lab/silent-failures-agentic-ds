import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/california_housing/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/california_housing/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "medianHouseValue"

y_train = train[target_col].values
X_train = train.drop(columns=[target_col])
X_test = test.copy()

def create_features(df):
    df = df.copy()
    
    if "totalRooms" in df and "households" in df:
        df["rooms_per_household"] = df["totalRooms"] / (df["households"] + 1e-5)
    if "totalBedrooms" in df and "totalRooms" in df:
        df["bedrooms_per_room"] = df["totalBedrooms"] / (df["totalRooms"] + 1e-5)
    if "population" in df and "households" in df:
        df["population_per_household"] = df["population"] / (df["households"] + 1e-5)
    if "totalBedrooms" in df and "households" in df:
        df["bedrooms_per_household"] = df["totalBedrooms"] / (df["households"] + 1e-5)
        
    if "medianIncome" in df:
        if "population" in df:
            df["income_per_pop"] = df["medianIncome"] / (df["population"] + 1e-5)
        if "households" in df:
            df["income_per_household"] = df["medianIncome"] / (df["households"] + 1e-5)
            
    if "latitude" in df and "longitude" in df:
        lat = df["latitude"]
        lon = df["longitude"]
        
        df["rot_45_1"] = lat * 0.7071 + lon * 0.7071
        df["rot_45_2"] = lat * 0.7071 - lon * 0.7071
        df["rot_30_1"] = lat * 0.8660 + lon * 0.5
        df["rot_30_2"] = lat * 0.5 - lon * 0.8660
        
        cities = {
            'SF': (37.7749, -122.4194),
            'LA': (34.0522, -118.2437),
            'SD': (32.7157, -117.1611),
            'SJ': (37.3382, -121.8863),
            'Sacramento': (38.5816, -121.4944),
            'Fresno': (36.7468, -119.7726)
        }
        
        for name, (c_lat, c_lon) in cities.items():
            df[f'dist_{name}'] = np.sqrt((lat - c_lat)**2 + (lon - c_lon)**2)
            
    return df

X_train_fe = create_features(X_train)
X_test_fe = create_features(X_test)

cat_cols = X_train_fe.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    X_train_fe[col] = X_train_fe[col].astype('category')
    X_test_fe[col] = X_test_fe[col].astype('category')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_test_preds = np.zeros(len(X_test_fe))
xgb_test_preds = np.zeros(len(X_test_fe))
cb_test_preds = np.zeros(len(X_test_fe))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_fe, y_train)):
    X_tr, y_tr = X_train_fe.iloc[train_idx], y_train[train_idx]
    X_val, y_val = X_train_fe.iloc[val_idx], y_train[val_idx]
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_test_preds += lgb_model.predict(X_test_fe) / 5.0
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True if len(cat_cols) > 0 else False,
        random_state=42 + fold,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_test_preds += xgb_model.predict(X_test_fe) / 5.0
    
    cb_model = cb.CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val)
    )
    cb_test_preds += cb_model.predict(X_test_fe) / 5.0

final_preds = 0.4 * lgb_test_preds + 0.3 * xgb_test_preds + 0.3 * cb_test_preds

y_min, y_max = y_train.min(), y_train.max()
final_preds = np.clip(final_preds, y_min, y_max)

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'medianHouseValue': final_preds
})
sub.to_csv('predictions.csv', index=False)