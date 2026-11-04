import pandas as pd
import numpy as np
import catboost as cb

TRAIN_FILE = "/path/to/user/workspace/ds_failures/datasets/strable/cohort-default-rate/data/train.parquet"
TEST_FILE = "/path/to/user/workspace/ds_failures/datasets/strable/cohort-default-rate/data/test.parquet"

target_col = "Average or Greater than 30"

train_df = pd.read_parquet(TRAIN_FILE)
test_df = pd.read_parquet(TEST_FILE)

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col].astype(float)

model = cb.CatBoostClassifier()
model.fit(X_train, y_train, verbose=False)

test_prob = model.predict_proba(test_df)[::, 1]

results = pd.DataFrame({
    "id": np.arange(len(test_df)),
    "Average or Greater than 30": test_prob
})

results.to_csv("predictions.csv", index=False)