import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "class"

# Map categorical labels to integer classes
unique_classes = sorted(train_df[target_col].unique().tolist())
label_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
idx_to_label = {idx: cls for idx, cls in enumerate(unique_classes)}

# Ensure y is a standard 1-D numpy array of integers
y = np.array(train_df[target_col].map(label_to_idx), dtype=np.int32)
X = train_df.drop(columns=[target_col]).copy()
X_test = test_df.copy()


# 2. Feature Engineering
def extract_features(df):
    data = df.copy()

    # Convert object/categorical columns to float if any
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Cyclical features for Day and Period
    if "Day" in data.columns:
        day_numeric = data["Day"].fillna(1)
        data["day_sin"] = np.sin(2 * np.pi * day_numeric / 7.0)
        data["day_cos"] = np.cos(2 * np.pi * day_numeric / 7.0)

    if "Period" in data.columns:
        period_numeric = data["Period"].fillna(0)
        data["period_sin"] = np.sin(2 * np.pi * period_numeric)
        data["period_cos"] = np.cos(2 * np.pi * period_numeric)

    # Domain interactions between NSW and VIC
    if "NSWprice" in data.columns and "VICprice" in data.columns:
        data["price_diff"] = data["NSWprice"] - data["VICprice"]
        data["price_ratio"] = data["NSWprice"] / (data["VICprice"] + 1e-6)

    if "NSWdemand" in data.columns and "VICdemand" in data.columns:
        data["demand_diff"] = data["NSWdemand"] - data["VICdemand"]
        data["demand_ratio"] = data["NSWdemand"] / (data["VICdemand"] + 1e-6)
        data["total_demand"] = data["NSWdemand"] + data["VICdemand"]

    if "transfer" in data.columns and "NSWdemand" in data.columns:
        data["transfer_to_nsw_demand"] = data["transfer"] / (data["NSWdemand"] + 1e-6)

    return data.astype(np.float32)


X_feats = extract_features(X)
X_test_feats = extract_features(X_test)
feature_cols = list(X_feats.columns)

# 3. Model Training & 5-Fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_feats))
test_preds = np.zeros(len(X_test_feats))

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_feats, y)):
    X_train_fold = X_feats.iloc[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_feats.iloc[val_idx]
    y_val_fold = y[val_idx]

    dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
    dval = lgb.Dataset(X_val_fold, label=y_val_fold, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    val_pred_probs = model.predict(X_val_fold, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred_probs

    test_preds += (
        model.predict(X_test_feats, num_iteration=model.best_iteration) / n_splits
    )

# 4. Evaluation
oof_class_preds = (oof_preds >= 0.5).astype(int)
cv_accuracy = accuracy_score(y, oof_class_preds)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# 5. Generate Submission
test_class_indices = (test_preds >= 0.5).astype(int)
test_class_labels = [idx_to_label[i] for i in test_class_indices]

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_class_labels)), "class": test_class_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
