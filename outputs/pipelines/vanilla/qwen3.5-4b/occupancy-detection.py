import pandas as pd
import numpy as np
import lightgbm as lgb

TRAIN_FILE = '/path/to/user/workspace/ds_failures/datasets/uci/occupancy-detection/data/train.parquet'
TEST_FILE = '/path/to/user/workspace/ds_failures/datasets/uci/occupancy-detection/data/test.parquet'
OUTPUT_FILE = 'predictions.csv'

train_df = pd.read_parquet(TRAIN_FILE)
test_df = pd.read_parquet(TEST_FILE)

features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
target = 'Occupancy'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]

clf = lgb.LGBClassifier(objective='binary', verbose=-1, n_estimators=1000, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

output_df = pd.DataFrame({'id': range(len(test_df)), 'Occupancy': preds.astype(int)})
output_df.to_csv(OUTPUT_FILE, index=False)