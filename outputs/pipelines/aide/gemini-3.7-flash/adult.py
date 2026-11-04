import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Normalize column names
train_df.columns = [c.strip().replace("-", "_") for c in train_df.columns]
test_df.columns = [c.strip().replace("-", "_") for c in test_df.columns]

target_col = "income"

# Standardize target values (strip whitespace/periods)
train_df[target_col] = train_df[target_col].astype(str).str.strip().str.rstrip(".")
unique_targets = train_df[target_col].unique()
positive_label = ">50K" if ">50K" in unique_targets else unique_targets[1]
negative_label = "<=50K" if "<=50K" in unique_targets else unique_targets[0]

y = (train_df[target_col] == positive_label).astype(int).values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Enhanced Feature Engineering
# Handle string cleaning and missing values
for col in combined_df.select_dtypes(include=["object", "category"]).columns:
    combined_df[col] = combined_df[col].astype(str).str.strip()
    combined_df[col] = combined_df[col].replace(
        {"?": "Unknown", "nan": "Unknown", "": "Unknown"}
    )

# Capital features
combined_df["capital_diff"] = combined_df["capital_gain"] - combined_df["capital_loss"]
combined_df["capital_gain_log"] = np.log1p(combined_df["capital_gain"])
combined_df["capital_loss_log"] = np.log1p(combined_df["capital_loss"])
combined_df["has_capital_gain"] = (combined_df["capital_gain"] > 0).astype(int)
combined_df["has_capital_loss"] = (combined_df["capital_loss"] > 0).astype(int)
combined_df["capital_gain_max"] = (combined_df["capital_gain"] >= 99999).astype(int)

# Age & Work Intensity interactions
combined_df["hours_per_age"] = combined_df["hours_per_week"] / (
    combined_df["age"] + 1e-5
)
combined_df["age_x_edu"] = combined_df["age"] * combined_df["education_num"]
combined_df["edu_x_hours"] = (
    combined_df["education_num"] * combined_df["hours_per_week"]
)
combined_df["is_fulltime"] = (
    (combined_df["hours_per_week"] >= 35) & (combined_df["hours_per_week"] <= 45)
).astype(int)
combined_df["is_overtime"] = (combined_df["hours_per_week"] > 40).astype(int)

# Demographics indicators
combined_df["is_married"] = (
    combined_df["marital_status"].str.startswith("Married").astype(int)
)
combined_df["is_us"] = (combined_df["native_country"] == "United-States").astype(int)

# Categorical combinations
combined_df["edu_occ"] = (
    combined_df["education"].astype(str) + "_" + combined_df["occupation"].astype(str)
)
combined_df["work_occ"] = (
    combined_df["workclass"].astype(str) + "_" + combined_df["occupation"].astype(str)
)
combined_df["marital_rel"] = (
    combined_df["marital_status"].astype(str)
    + "_"
    + combined_df["relationship"].astype(str)
)

# Relative group aggregations (deviation from peers)
occ_edu_mean = combined_df.groupby("occupation")["education_num"].transform("mean")
combined_df["edu_diff_occ_mean"] = combined_df["education_num"] - occ_edu_mean

occ_hours_mean = combined_df.groupby("occupation")["hours_per_week"].transform("mean")
combined_df["hours_diff_occ_mean"] = combined_df["hours_per_week"] - occ_hours_mean

work_hours_mean = combined_df.groupby("workclass")["hours_per_week"].transform("mean")
combined_df["hours_diff_work_mean"] = combined_df["hours_per_week"] - work_hours_mean

# Identify categorical columns
categorical_cols = combined_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()
for col in categorical_cols:
    combined_df[col] = combined_df[col].fillna("Unknown").astype(str)

X = combined_df.iloc[:n_train].reset_index(drop=True)
X_test = combined_df.iloc[n_train:].reset_index(drop=True)

# 3. Model Training with 5-Fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

cat_features_idx = [X.columns.get_loc(c) for c in categorical_cols]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features_idx)
    val_pool = Pool(X_va, y_va, cat_features=cat_features_idx)
    test_pool = Pool(X_test, cat_features=cat_features_idx)

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=4.0,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=42 + fold,
        verbose=False,
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)

    oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
    test_preds += model.predict_proba(test_pool)[:, 1] / n_splits

# 4. Find Best Probability Threshold for Accuracy
best_acc = 0.0
best_threshold = 0.5
thresholds = np.linspace(0.3, 0.7, 81)

for thresh in thresholds:
    acc = accuracy_score(y, (oof_preds >= thresh).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_threshold = thresh

print(f"Validation Accuracy: {best_acc:.5f} (Optimal Threshold: {best_threshold:.3f})")

# 5. Generate Submission File
os.makedirs("./working", exist_ok=True)
binary_test_preds = (test_preds >= best_threshold).astype(int)
final_labels = [
    positive_label if pred == 1 else negative_label for pred in binary_test_preds
]

submission = pd.DataFrame({"id": np.arange(len(final_labels)), "income": final_labels})

submission.to_csv("./working/submission.csv", index=False)
