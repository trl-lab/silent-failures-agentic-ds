import os
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/sf-crime/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/sf-crime/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Category"
target_le = LabelEncoder()
y_train = target_le.fit_transform(train_df[target_col])

def clean_addr_str(s):
    return re.sub(r'^\d+\s+Block\s+of\s+', '', str(s))

combined_address = pd.concat([train_df['Address'], test_df['Address']], axis=0)
combined_addr_clean = combined_address.apply(clean_addr_str)
combined_district = pd.concat([train_df['PdDistrict'], test_df['PdDistrict']], axis=0)
combined_dow = pd.concat([train_df['DayOfWeek'], test_df['DayOfWeek']], axis=0)

freq_addr = combined_address.value_counts()
freq_addr_clean = combined_addr_clean.value_counts()
freq_district = combined_district.value_counts()
freq_dow = combined_dow.value_counts()

le_addr = LabelEncoder().fit(combined_address)
le_addr_clean = LabelEncoder().fit(combined_addr_clean)
le_district = LabelEncoder().fit(combined_district)
le_dow = LabelEncoder().fit(combined_dow)

def process_df(df):
    if not pd.api.types.is_datetime64_any_dtype(df['Dates']):
        dates = pd.to_datetime(df['Dates'])
    else:
        dates = df['Dates']
    
    res = pd.DataFrame(index=df.index)
    
    res['Year'] = dates.dt.year
    res['Month'] = dates.dt.month
    res['Day'] = dates.dt.day
    res['Hour'] = dates.dt.hour
    res['Minute'] = dates.dt.minute
    res['DayOfWeek_num'] = dates.dt.dayofweek
    res['DayOfYear'] = dates.dt.dayofyear
    res['WeekOfYear'] = dates.dt.isocalendar().week.astype(int)
    res['Is_Weekend'] = res['DayOfWeek_num'].isin([5, 6]).astype(int)
    res['Time_in_Minutes'] = res['Hour'] * 60 + res['Minute']
    
    addr_str = df['Address'].astype(str)
    addr_clean_str = addr_str.apply(clean_addr_str)
    
    res['Is_Intersection'] = addr_str.str.contains('/').astype(int)
    res['Is_Block'] = addr_str.str.contains('Block').astype(int)
    
    res['Address_freq'] = addr_str.map(freq_addr)
    res['Address_clean_freq'] = addr_clean_str.map(freq_addr_clean)
    res['Address_cat'] = le_addr.transform(addr_str)
    res['Address_clean_cat'] = le_addr_clean.transform(addr_clean_str)
    
    res['PdDistrict_freq'] = df['PdDistrict'].map(freq_district)
    res['PdDistrict_cat'] = le_district.transform(df['PdDistrict'])
    res['DayOfWeek_freq'] = df['DayOfWeek'].map(freq_dow)
    res['DayOfWeek_cat'] = le_dow.transform(df['DayOfWeek'])
    
    x = df['X'].astype(float)
    y = df['Y'].astype(float)
    
    outliers = (y > 40) | (y < 37) | (x > -122) | (x < -123)
    x_clean = x.copy()
    y_clean = y.copy()
    x_clean[outliers] = np.nan
    y_clean[outliers] = np.nan
    
    x_median = x_clean.median()
    y_median = y_clean.median()
    
    x_clean = x_clean.fillna(x_median)
    y_clean = y_clean.fillna(y_median)
    
    res['X_clean'] = x_clean
    res['Y_clean'] = y_clean
    
    rad30 = np.radians(30)
    rad45 = np.radians(45)
    
    res['X_rot30'] = x_clean * np.cos(rad30) - y_clean * np.sin(rad30)
    res['Y_rot30'] = x_clean * np.sin(rad30) + y_clean * np.cos(rad30)
    
    res['X_rot45'] = x_clean * np.cos(rad45) - y_clean * np.sin(rad45)
    res['Y_rot45'] = x_clean * np.sin(rad45) + y_clean * np.cos(rad45)
    
    res['Dist_Center'] = np.sqrt((x_clean - (-122.4074))**2 + (y_clean - 37.7879)**2)
    
    res = res.fillna(0)
    return res

X_train = process_df(train_df)
X_test = process_df(test_df)

cat_cols = ['PdDistrict_cat', 'DayOfWeek_cat', 'Address_cat', 'Address_clean_cat']

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train, y_train, categorical_feature=cat_cols)

preds_raw = model.predict(X_test)
preds_category = target_le.inverse_transform(preds_raw)

sub = pd.DataFrame({
    'id': np.arange(len(test_df)),
    'Category': preds_category
})

sub.to_csv('predictions.csv', index=False)