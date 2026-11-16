import pandas as pd
import numpy as np
import catboost as cb

train_file = "/path/to/user/workspace/ds_failures/datasets/openml/connect-4/data/train.parquet"
test_file = "/path/to/user/workspace/ds_failures/datasets/openml/connect-4/data/test.parquet"
output_file = "predictions.csv"

df_train = pd.read_parquet(train_file)
df_test = pd.read_parquet(test_file)

target_col = "class"
feature_cols = [col for col in df_train.columns if col != target_col]

X_train = df_train[feature_cols]
y_train = df_train[target_col]
X_test = df_test[feature_cols]

model = cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, verbose=0, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

results = pd.DataFrame({
    "id": list(range(len(predictions))),
    "class": predictions.tolist()
})

results.to_csv(output_file, index=False)