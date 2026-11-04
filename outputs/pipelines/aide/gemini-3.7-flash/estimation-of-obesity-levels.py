import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "NObeyesdad"


# 2. Feature Engineering
def engineer_features(df):
    data = df.copy()
    # BMI calculation: Weight (kg) / Height (m)^2
    data["BMI"] = data["Weight"] / (data["Height"] ** 2)
    # Physical activity vs sedentary screen time ratio
    data["FAF_TUE_ratio"] = data["FAF"] / (data["TUE"] + 0.1)
    # Water intake per unit of weight
    data["CH2O_Weight_ratio"] = data["CH2O"] / (data["Weight"] + 0.1)
    return data


train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# Separate features and target
X = train_df.drop(columns=[target_col]).copy()
y = train_df[target_col].copy()
X_test = test_df.copy()

# Target encoding
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
num_classes = len(target_encoder.classes_)

# Encode categorical features
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined train + test to handle any unseen categories gracefully
    combined_vals = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined_vals)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# 3. Model Training & Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X), num_classes))
test_preds = np.zeros((len(X_test), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_train, y_train = X.iloc[train_idx], y_encoded[train_idx]
    X_val, y_val = X.iloc[val_idx], y_encoded[val_idx]

    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_cols
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, categorical_feature=categorical_cols
    )

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 4. Evaluation
oof_class_preds = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y_encoded, oof_class_preds)
print(f"Validation Accuracy: {cv_accuracy:.5f}")

# 5. Generate Submission
test_class_preds = np.argmax(test_preds, axis=1)
test_labels = target_encoder.inverse_transform(test_class_preds)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_labels)), "NObeyesdad": test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
