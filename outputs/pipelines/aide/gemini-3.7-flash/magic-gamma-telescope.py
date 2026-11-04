import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Prepare target variable (gamma 'g' = 1, hadron 'h' = 0 or standard binary)
target_col = "class"
if train_df[target_col].dtype == object or isinstance(
    train_df[target_col].iloc[0], str
):
    y = (train_df[target_col] == "g").astype(int)
else:
    y = train_df[target_col].astype(int)


# 2. Feature Engineering
def engineer_features(df):
    data = df.copy()
    feature_cols = [
        c
        for c in [
            "fLength",
            "fWidth",
            "fSize",
            "fConc",
            "fConc1",
            "fAsym",
            "fM3Long",
            "fM3Trans",
            "fAlpha",
            "fDist",
        ]
        if c in data.columns
    ]

    # Ratios and geometric interactions
    data["f_aspect_ratio"] = data["fLength"] / (data["fWidth"] + 1e-5)
    data["f_area"] = np.pi * data["fLength"] * data["fWidth"] / 4.0
    data["f_density"] = data["fSize"] / (data["f_area"] + 1e-5)
    data["f_rel_asym"] = data["fAsym"] / (data["fLength"] + 1e-5)
    data["f_conc_ratio"] = data["fConc1"] / (data["fConc"] + 1e-5)
    data["f_moment_mag"] = np.sqrt(data["fM3Long"] ** 2 + data["fM3Trans"] ** 2)

    # Angular / coordinate projections
    alpha_rad = np.radians(data["fAlpha"])
    data["f_dist_cos_alpha"] = data["fDist"] * np.cos(alpha_rad)
    data["f_dist_sin_alpha"] = data["fDist"] * np.sin(alpha_rad)

    # Non-linear terms
    data["f_length_sq"] = data["fLength"] ** 2
    data["f_width_sq"] = data["fWidth"] ** 2

    drop_cols = [target_col] if target_col in data.columns else []
    return data.drop(columns=drop_cols)


X = engineer_features(train_df)
X_test = engineer_features(test_df)

# 3. Model Training with 5-Fold Stratified K-Fold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred

    test_pred_fold = model.predict(X_test, num_iteration=model.best_iteration)
    test_preds += test_pred_fold / n_splits

# 4. Evaluation
overall_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC-AUC: {overall_auc:.5f}")

# 5. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "class": test_preds})
submission.to_csv("./working/submission.csv", index=False)
print("Saved predictions to ./working/submission.csv successfully.")
