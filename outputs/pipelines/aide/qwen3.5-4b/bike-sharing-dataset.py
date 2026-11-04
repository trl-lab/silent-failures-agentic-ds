import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
feature_cols = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
categorical_cols = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
continuous_cols = ["temp", "atemp", "hum", "windspeed"]

# Create preprocessing
preprocessor = ColumnTransformer(
    [
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ),
        ("cont", "passthrough", continuous_cols),
    ]
)

# Create model pipeline
model = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "model",
            lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            ),
        ),
    ]
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    X_train, y_train = train_df[feature_cols].values, train_df["cnt"].values
    X_val, y_val = (
        train_df[feature_cols].iloc[val_idx].values,
        train_df["cnt"].iloc[val_idx].values,
    )

    X_train_processed, X_val_processed = preprocessor.fit_transform(
        X_train
    ), preprocessor.transform(X_val)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

    train_set = lgb.Dataset(
        X_train_processed,
        label=y_train,
        feature_name=preprocessor.get_feature_names_out().tolist(),
    )
    val_set = lgb.Dataset(X_val_processed, label=y_val, reference=train_set)

    model.fit(train_set, callbacks=callbacks, eval_set=[val_set], num_boost_round=1000)

    y_pred_val = model.predict(X_val_processed)
    rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
    scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.4f}")

mean_rmse = np.mean(scores)
print(f"\nMean Cross-Validation RMSE: {mean_rmse:.4f}")

# Predict on test data
X_test = test_df[feature_cols].values
X_test_processed = preprocessor.transform(X_test)

test_predictions = model.predict(X_test_processed)

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "cnt": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
