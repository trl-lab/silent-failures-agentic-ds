import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/strable/wine-dataset/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/wine-dataset/data/test.parquet"
output_path = "predictions.csv"

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

target_column = "Price"
feature_columns = [col for col in df_train.columns if col != target_column]

X_train = df_train[feature_columns]
y_train = df_train[target_column]
X_test = df_test[feature_columns]

model = CatBoostRegressor(verbose=0, iterations=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions_df = pd.DataFrame({
    "id": range(len(y_pred)),
    "Price": y_pred
})

predictions_df.to_csv(output_path, index=False)