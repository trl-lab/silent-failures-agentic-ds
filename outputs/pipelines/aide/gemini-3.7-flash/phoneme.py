import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and feature columns
target_col = "class"
feature_cols = [col for col in train_df.columns if col != target_col]


# Feature engineering: ratios, polynomial features, and statistics across harmonics
def engineer_features(df, features):
    data = df[features].copy()

    # Statistical aggregates across harmonic features
    data["mean_val"] = data[features].mean(axis=1)
    data["std_val"] = data[features].std(axis=1)
    data["min_val"] = data[features].min(axis=1)
    data["max_val"] = data[features].max(axis=1)
    data["sum_val"] = data[features].sum(axis=1)
    data["range_val"] = data["max_val"] - data["min_val"]

    # Pairwise interactions and ratios
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            data[f"{f1}_mult_{f2}"] = data[f1] * data[f2]
            data[f"{f1}_diff_{f2}"] = data[f1] - data[f2]
            data[f"{f1}_div_{f2}"] = data[f1] / (data[f2].abs() + 1e-6)

    return data


X = engineer_features(train_df, feature_cols)
y = train_df[target_col].values.astype(int)
X_test = engineer_features(test_df, feature_cols)

# Stratified 5-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "binary",
    "metric": "binary_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=trn_data)

    clf = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_preds_prob = clf.predict(X_val_fold, num_iteration=clf.best_iteration)
    oof_preds[val_idx] = val_preds_prob
    test_preds += clf.predict(X_test, num_iteration=clf.best_iteration) / n_splits

# Optimal thresholding & evaluation
best_acc = 0.0
best_th = 0.5
for th in np.linspace(0.3, 0.7, 41):
    acc = accuracy_score(y, (oof_preds >= th).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_th = th

cv_accuracy = accuracy_score(y, (oof_preds >= best_th).astype(int))
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f} (Best Threshold: {best_th:.2f})")

# Generate test predictions
test_binary_preds = (test_preds >= best_th).astype(int)

# Create submission dataframe
submission = pd.DataFrame({"id": np.arange(len(test_df)), "class": test_binary_preds})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
