import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/strable/us-school-bus-fleet/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/strable/us-school-bus-fleet/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = 'Bus age'
y = train_df[target_col].values

# Feature engineering function
def extract_features(df):
    df_feat = df.copy()
    
    # Extract inspection year from date string
    def parse_year(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val)
        matches = re.findall(r'\b(19\d\d|20\d\d)\b', val_str)
        if matches:
            return float(matches[0])
        return np.nan

    df_feat['inspection_year'] = df_feat['Date updated or last inspected'].apply(parse_year)
    
    # Extract features from VIN
    def get_vin_char10(v):
        if pd.isna(v):
            return 'missing'
        s = str(v).strip()
        return s[9].upper() if len(s) >= 10 else 'missing'

    def get_vin_wmi(v):
        if pd.isna(v):
            return 'missing'
        s = str(v).strip()
        return s[:3].upper() if len(s) >= 3 else 'missing'

    # Mapping for standard 10th VIN year digit
    vin_year_map = {
        'A': 2010, 'B': 2011, 'C': 2012, 'D': 2013, 'E': 2014, 'F': 2015,
        'G': 2016, 'H': 2017, 'J': 2018, 'K': 2019, 'L': 2020, 'M': 2021,
        'N': 2022, 'P': 2023, 'R': 2024, 'S': 2025, 'T': 2026, 'V': 2027,
        '1': 2001, '2': 2002, '3': 2003, '4': 2004, '5': 2005, '6': 2006,
        '7': 2007, '8': 2008, '9': 2009, 'Y': 2000, 'X': 1999, 'W': 1998,
        'V': 1997, 'U': 1996, 'T': 1995, 'S': 1994, 'R': 1993, 'P': 1992
    }
    
    df_feat['vin_char10'] = df_feat['VIN'].apply(get_vin_char10)
    df_feat['vin_wmi'] = df_feat['VIN'].apply(get_vin_wmi)
    df_feat['vin_inferred_year'] = df_feat['vin_char10'].map(vin_year_map)

    # Impute or combine model year
    df_feat['effective_model_year'] = df_feat['Model year'].fillna(df_feat['vin_inferred_year'])
    df_feat['year_diff'] = df_feat['inspection_year'] - df_feat['effective_model_year']
    df_feat['log_year_diff'] = np.log1p(np.maximum(0, df_feat['year_diff'].fillna(0)))

    # Drop target and unneeded raw strings
    drop_cols = [target_col, 'WRI Bus ID', 'VIN', 'Date updated or last inspected']
    for c in drop_cols:
        if c in df_feat.columns:
            df_feat = df_feat.drop(columns=[c])

    # Convert object and categorical columns to string
    cat_cols = df_feat.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype(str).fillna('missing')
        df_feat[c] = df_feat[c].astype('category')
        
    return df_feat

X = extract_features(train_df)
X_test = extract_features(test_df)

# Align categories between train and test
cat_cols = X.select_dtypes(include=['category']).columns.tolist()
for c in cat_cols:
    train_cats = set(X[c].cat.categories)
    test_cats = set(X_test[c].cat.categories)
    all_cats = list(train_cats.union(test_cats))
    X[c] = pd.Categorical(X[c], categories=all_cats)
    X_test[c] = pd.Categorical(X_test[c], categories=all_cats)

# K-Fold Cross Validation & Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

# Prepare numeric/cat data for CatBoost
cb_cat_cols = cat_cols
X_cb = X.copy()
X_test_cb = X_test.copy()
for c in cb_cat_cols:
    X_cb[c] = X_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

for train_idx, val_idx in kf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    # CatBoost
    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cb_cat_cols,
        verbose=False,
        random_seed=42
    )
    cb_model.fit(
        X_cb.iloc[train_idx], y_tr,
        eval_set=(X_cb.iloc[val_idx], y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test_cb) / kf.n_splits

final_predictions = 0.6 * lgb_preds + 0.4 * cb_preds

# Save predictions
output_df = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'Bus age': final_predictions
})
output_df.to_csv('predictions.csv', index=False)