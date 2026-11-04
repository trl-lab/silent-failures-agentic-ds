import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")


# Feature engineering
def preprocess(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Dates"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["X"] = df["X"].fillna(df["X"].median())
    df["Y"] = df["Y"].fillna(df["Y"].median())
    df["Resolution"] = df["Resolution"].fillna("NONE")
    df["PdDistrict"] = df["PdDistrict"].fillna("UNKNOWN")
    df["Address"] = df["Address"].fillna("UNKNOWN")
    df["Descript"] = df["Descript"].fillna("UNKNOWN")
    return df


train = preprocess(train)
test = preprocess(test)

# Encode categorical features
cat_cols = ["Address", "Category", "Descript", "PdDistrict", "Resolution", "DayOfWeek"]
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Create feature matrix
feature_cols = [
    "Year",
    "Month",
    "DayOfWeek",
    "X",
    "Y",
    "PdDistrict",
    "Resolution",
    "Descript",
    "Address",
]
X_train = train[feature_cols].values
X_test = test[feature_cols].values
y_train = train["Category"].values

# Train LightGBM model with 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_log_loss = np.inf

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold)

    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        valid_sets=[val_data],
        valid_names=["val"],
    )

    y_pred = model.predict(X_val_fold)
    fold_log_loss = log_loss(y_val_fold, y_pred)
    print(f"Fold {fold+1} Log Loss: {fold_log_loss:.4f}")

    if fold_log_loss < best_log_loss:
        best_log_loss = fold_log_loss
        best_model = model

# Generate predictions for test set
y_pred_test = best_model.predict(X_test)
test_predictions = pd.DataFrame({"id": test.index, "Category": y_pred_test})

# Save predictions
test_predictions.to_csv("./working/submission.csv", index=False)
print(f"Test predictions saved to submission.csv")
print(f"Best validation Log Loss: {best_log_loss:.4f}")
