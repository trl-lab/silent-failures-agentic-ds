import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/transmission-towers/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/transmission-towers/data/test.parquet"
    target_col = "SupStruc"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    feature_cols = [col for col in train_df.columns if col != target_col]

    train_processed = train_df[feature_cols].copy()
    test_processed = test_df[feature_cols].copy()

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_processed[col]):
            median_val = train_processed[col].median()
            train_processed[col].fillna(median_val, inplace=True)
            test_processed[col].fillna(median_val, inplace=True)
        else:
            train_processed[col].fillna("UNK", inplace=True)
            test_processed[col].fillna("UNK", inplace=True)
            le = LabelEncoder()
            le.fit(train_processed[col].astype(str))
            train_processed[col] = le.transform(train_processed[col])
            test_processed[col] = le.transform(test_processed[col])

    y_train = train_df[target_col]
    model = lgb.LGBMRegressor(objective="regression", random_state=42, verbose=-1, num_leaves=31)
    model.fit(train_processed.values, y_train)

    predictions = model.predict(test_processed.values)

    output_df = pd.DataFrame({"id": range(len(test_df)), "SupStruc": predictions})
    output_df.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()