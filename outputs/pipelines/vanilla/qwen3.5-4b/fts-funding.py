import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


train_path = "/path/to/user/workspace/ds_failures/datasets/strable/fts-funding/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/strable/fts-funding/data/test.parquet"
output_path = "predictions.csv"


train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)


target_col = "amountUSD"

# Prepare training data
y = train_df[target_col]
X_train = train_df.drop(columns=[target_col])

# Prepare test data (ensure we only keep original feature columns)
feature_cols = X_train.columns.tolist()
test_X = test_df[feature_cols].copy()

# Handle missing values and categorical columns automatically
numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, feature_cols["dtypes"].apply(pd.api.types.is_numeric_dtype) == False),
        ("cat", categorical_transformer, feature_cols["dtypes"].apply(pd.api.types.is_numeric_dtype) == True)
    ]
)


def _get_numeric_mask(df):
    return df.select_dtypes(include=["number"]).columns


def _get_categorical_mask(df):
    return df.select_dtypes(include=["object", "category"]).columns

# Correctly identify features for preprocessor
num_cols = list(_get_numeric_mask(X_train).tolist())
cat_cols = list(_get_categorical_mask(X_train).tolist())

preprocessor._finalizers = lambda df: df.reindex(columns=num_cols + cat_cols)

clf = Pipeline(steps=[("preprocess", ColumnTransformer(transformers=[("num", numeric_transformer, num_cols), ("cat", categorical_transformer, cat_cols)])), ("model", lgb.LGBMRegressor(n_estimators=1000, verbose=-1, random_state=42))] )

clf.fit(X_train, y)

# Predictions
pred_values = clf.predict(test_X)

# Create output dataframe with id and amountUSD
result_df = pd.DataFrame({
    "id": np.arange(len(pred_values)),
    "amountUSD": pred_values
})

# Save to CSV
result_df.to_csv(output_path, index=False)