import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor

TRAIN_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/media-ranking_wide/data/train.parquet'
TEST_PATH = '/path/to/user/workspace/ds_failures/datasets/strable/media-ranking_wide/data/test.parquet'
OUTPUT_PATH = 'predictions.csv'

train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

TARGET_COLUMN = 'Overall'
FEATURE_COLUMNS = [col for col in train_df.columns if col != TARGET_COLUMN]

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]
X_test = test_df[FEATURE_COLUMNS]

model = CatBoostRegressor(iterations=1000, depth=6, verbose=False)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

results = pd.DataFrame({
    'id': range(len(predictions)),
    'Overall': predictions
})

results.to_csv(OUTPUT_PATH, index=False)