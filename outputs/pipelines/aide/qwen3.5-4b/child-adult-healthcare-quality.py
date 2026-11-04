import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Prepare features and target
feature_cols = [col for col in train_df.columns if col != "State Rate"]

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
        ),
    ]
)

# Create model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=100,
)

# Use 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(train_df):
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]

    X_train, y_train = train_data[feature_cols], train_data["State Rate"]
    X_val, y_val = val_data[feature_cols], val_data["State Rate"]

    # Fit model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Calculate RMSE
    rmse = np.sqrt(((model.predict(X_val) - y_val) ** 2).mean())
    cv_scores.append(rmse)

# Print evaluation metric
print(f"Cross-validation RMSE: {np.sqrt(np.mean(cv_scores)):.4f}")

# Predict on test data
X_test = test_df[feature_cols]
predictions = model.predict(X_test)

# Save submission
submission = pd.DataFrame({"id": range(len(predictions)), "State Rate": predictions})
submission.to_csv("./working/submission.csv", index=False)


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
target_col = "State Rate"
feature_cols = [col for col in train_df.columns if col != target_col]

# Identify numeric and categorical columns
numeric_features = feature_cols[
    feature_cols.apply(lambda x: pd.api.types.is_numeric_dtype(train_df[x]))
]
categorical_features = feature_cols[
    feature_cols.apply(lambda x: pd.api.types.is_categorical_dtype(train_df[x]))
]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
        ),
    ]
)

# Model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []

for train_idx, val_idx in kf.split(train_df):
    X_train, y_train = (
        train_df.iloc[train_idx][feature_cols].values,
        train_df.iloc[train_idx][target_col].values,
    )
    X_val, y_val = (
        train_df.iloc[val_idx][feature_cols].values,
        train_df.iloc[val_idx][target_col].values,
    )

    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    lgb_train = lgb.Dataset(X_train_processed, label=y_train)
    lgb_val = lgb.Dataset(X_val_processed, label=y_val, reference=lgb_train)

    model.fit(
        lgb_train,
        eval_set=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        n_jobs=-1,
    )

    y_pred = model.predict(X_val_processed)
    rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
    cv_rmse_scores.append(rmse)

print(f"Cross-validation RMSE: {np.sqrt(np.mean(cv_rmse_scores)):.4f}")

# Predict on test data
X_test = test_df[feature_cols].values
X_test_processed = preprocessor.transform(X_test)
test_predictions = model.predict(X_test_processed)

# Save submission
submission = pd.DataFrame(
    {"id": range(len(test_predictions)), "State Rate": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
