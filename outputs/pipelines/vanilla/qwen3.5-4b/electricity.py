import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/electricity/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/electricity/data/test.parquet"
OUTPUT_PATH = "predictions.csv"

df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

target = "class"
X_train = df_train.drop(columns=[target])
y_train = df_train[target]
X_test = df_test.drop(columns=[target])

model = CatBoostClassifier(depth=3, iterations=100, verbose=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pred_df = pd.DataFrame({"id": range(len(y_pred)), "class": y_pred})
pred_df.to_csv(OUTPUT_PATH, index=False)