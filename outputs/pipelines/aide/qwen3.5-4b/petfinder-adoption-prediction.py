import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target_col = "AdoptionSpeed"
feature_cols = [c for c in train_df.columns if c != target_col]

# Create feature matrix
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle categorical features
cat_cols = [
    "Breed1",
    "Breed2",
    "Color1",
    "Color2",
    "Color3",
    "Gender",
    "MaturitySize",
    "Type",
    "VideoAmt",
]
num_cols = [
    "Age",
    "Breed1",
    "Breed2",
    "Color1",
    "Color2",
    "Color3",
    "Dewormed",
    "Fee",
    "FurLength",
    "Health",
    "MaturitySize",
    "PetID",
    "PhotoAmt",
    "RescuerID",
    "State",
    "Sterilized",
    "Vaccinated",
    "VideoAmt",
]

# Encode categorical variables
cat_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined train and test to avoid leakage
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined)
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    cat_encoders[col] = le

# One-hot encode
X_train = pd.get_dummies(X_train, columns=cat_cols)
X_test = pd.get_dummies(X_test, columns=cat_cols)

# Train and evaluate LightGBM model
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create predictions
predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    predictions += model.predict(X_test) / 5

# Save predictions
submission = pd.DataFrame({"id": range(len(X_test)), "AdoptionSpeed": predictions})
submission.to_csv("./working/submission.csv", index=False)

# Calculate evaluation metric
from sklearn.metrics import mean_squared_error

# For categorical target, calculate weighted kappa
from sklearn.metrics import adjusted_rand_score
from scipy.stats import pearsonr

# Simple approach: calculate adoption rate for each fold
fold_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict(X_val_fold)
    y_pred = np.round(y_pred).astype(int)

    # Calculate weighted kappa
    from sklearn.metrics import make_scorer, classification_report
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_train_fold, y_pred)
    # Calculate weighted kappa
    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(y_train_fold, y_pred, weights="quadratic")
    fold_scores.append(kappa)

# Calculate average kappa
avg_kappa = np.mean(fold_scores)
print(f"Average Kappa: {avg_kappa:.4f}")


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score
import os

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Define features and target
target_col = "AdoptionSpeed"
feature_cols = [c for c in train_df.columns if c != target_col]

# Create feature matrix
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Identify categorical columns
cat_cols = [
    "Breed1",
    "Breed2",
    "Color1",
    "Color2",
    "Color3",
    "Gender",
    "MaturitySize",
    "Type",
    "VideoAmt",
    "Dewormed",
    "FurLength",
    "Health",
    "Sterilized",
    "Vaccinated",
    "State",
    "RescuerID",
    "PetID",
    "Name",
    "Description",
]
num_cols = [c for c in feature_cols if c not in cat_cols]

# Encode categorical features with LabelEncoder
cat_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    cat_encoders[col] = le

# Create final feature matrix
X_train_final = train_df[feature_cols].copy()
X_test_final = test_df[feature_cols].copy()

# LightGBM parameters
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = np.zeros(len(X_test_final))
fold_kappas = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_final)):
    X_train_fold = X_train_final.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    predictions += model.predict(X_test_final) / 5

    # Calculate kappa on validation fold
    val_pred = model.predict(X_train_final.iloc[val_idx])
    val_pred = np.round(val_pred).astype(int)
    val_true = y_train.iloc[val_idx]
    fold_kappas.append(cohen_kappa_score(val_true, val_pred, weights="quadratic"))

# Calculate average kappa score
avg_kappa = np.mean(fold_kappas)
print(f"Average Kappa (qwk): {avg_kappa:.4f}")

# Save submission
submission = pd.DataFrame(
    {"id": range(len(X_test_final)), "AdoptionSpeed": predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
