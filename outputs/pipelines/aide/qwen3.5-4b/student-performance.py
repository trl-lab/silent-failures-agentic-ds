import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target and features
target = "G3"
feature_cols = [col for col in train_df.columns if col not in [target, "id"]]

# Create Label Encoders for categorical features
label_encoders = {}
for col in train_df.select_dtypes(include=["object"]).columns:
    if col not in [
        "school",
        "sex",
        "address",
        "famsize",
        "Pstatus",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
    ]:
        # These are the categorical columns
        pass


# Encode categorical features
def encode_categorical(df, col, encoder):
    df[col] = df[col].map(encoder)
    return df


# Prepare training data
X_train = train_df[feature_cols].copy()
y_train = train_df[target].copy()

# Encode categorical variables
for col in X_train.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    label_encoders[col] = le

# Handle missing values
X_train = X_train.fillna(X_train.median())

# LightGBM model setup
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
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
cv_scores = []
for train_idx, val_idx in kf.split(X_train):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val, y_train.iloc[val_idx])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    y_pred = model.predict(X_val)
    cv_scores.append(np.sqrt(np.mean((y_pred - y_train.iloc[val_idx]) ** 2)))

print(f"Cross-validation RMSE: {np.mean(cv_scores):.2f}")

# Train final model on all data
model.fit(X_train, y_train)

# Predict on test data
test_features = test_df[feature_cols].copy()
for col in test_features.select_dtypes(include=["object"]).columns:
    test_features[col] = test_features[col].astype(str)
    test_features[col] = test_features[col].map(label_encoders[col])
    test_features[col] = test_features[col].fillna(test_features[col].median())

# Get predictions
y_pred = model.predict(test_features)

# Create submission
submission = pd.DataFrame({"id": test_df["id"], "G3": y_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully!")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define target and features
target = "G3"
feature_cols = [col for col in train_df.columns if col not in [target, "id"]]

# Separate categorical and numerical features
categorical_cols = train_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()
numerical_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Initialize label encoders for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on training data only
    le.fit(train_df[col])
    label_encoders[col] = le


# Encode categorical features
def encode_categorical(df, col, encoder):
    df[col] = df[col].map(encoder)
    return df


X_train = train_df.copy()
X_test = test_df.copy()

# Encode categorical variables in training data
for col in categorical_cols:
    X_train[col] = X_train[col].map(label_encoders[col])
    X_test[col] = X_test[col].map(label_encoders[col])

# Fill missing values with median for numerical columns
for col in numerical_cols:
    X_train[col] = X_train[col].fillna(X_train[col].median())
    X_test[col] = X_test[col].fillna(X_test[col].median())

# Prepare training data
y_train = train_df[target].values
X_train = X_train[feature_cols].values
X_test = X_test[feature_cols].values

# LightGBM model parameters
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    y_pred = model.predict(X_val_fold)
    fold_rmse = np.sqrt(np.mean((y_pred - y_val_fold) ** 2))
    cv_rmse_scores.append(fold_rmse)
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.4f}")

# Print final CV RMSE
print(f"Final CV RMSE: {np.mean(cv_rmse_scores):.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
)

# Predict on test data
y_test_pred = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"id": range(len(X_test)), "G3": y_test_pred})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
