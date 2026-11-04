import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Drop continuous stab column if present to avoid data leakage
for df in [train_df, test_df]:
    if "stab" in df.columns:
        df.drop(columns=["stab"], inplace=True)

# Encode binary target if string or boolean
if train_df["stabf"].dtype == object or train_df["stabf"].dtype == bool:
    target_mapping = {"unstable": 1, "stable": 0, True: 1, False: 0}
    y = train_df["stabf"].map(target_mapping).values
else:
    y = train_df["stabf"].values.astype(int)

train_df = train_df.drop(columns=["stabf"])


# Feature engineering function
def engineer_features(df):
    data = df.copy()

    # Power relationships
    data["p_sum"] = data["p1"] + data["p2"] + data["p3"] + data["p4"]
    data["p_consumer_sum"] = data["p2"] + data["p3"] + data["p4"]
    data["p_consumer_abs_sum"] = data["p2"].abs() + data["p3"].abs() + data["p4"].abs()
    data["p_ratio_producer_consumer"] = data["p1"] / (data["p_consumer_abs_sum"] + 1e-6)

    # Reaction time (tau) features
    tau_cols = ["tau1", "tau2", "tau3", "tau4"]
    data["tau_mean"] = data[tau_cols].mean(axis=1)
    data["tau_std"] = data[tau_cols].std(axis=1)
    data["tau_min"] = data[tau_cols].min(axis=1)
    data["tau_max"] = data[tau_cols].max(axis=1)
    data["tau_consumer_mean"] = data[["tau2", "tau3", "tau4"]].mean(axis=1)
    data["tau_ratio_1_consumers"] = data["tau1"] / (data["tau_consumer_mean"] + 1e-6)

    # Price elasticity (g) features
    g_cols = ["g1", "g2", "g3", "g4"]
    data["g_mean"] = data[g_cols].mean(axis=1)
    data["g_std"] = data[g_cols].std(axis=1)
    data["g_min"] = data[g_cols].min(axis=1)
    data["g_max"] = data[g_cols].max(axis=1)
    data["g_consumer_mean"] = data[["g2", "g3", "g4"]].mean(axis=1)
    data["g_ratio_1_consumers"] = data["g1"] / (data["g_consumer_mean"] + 1e-6)

    # Cross-parameter interactions for each node
    for i in range(1, 5):
        data[f"tau_p_{i}"] = data[f"tau{i}"] * data[f"p{i}"].abs()
        data[f"tau_g_{i}"] = data[f"tau{i}"] * data[f"g{i}"]
        data[f"p_g_{i}"] = data[f"p{i}"].abs() * data[f"g{i}"]
        data[f"tau_p_g_{i}"] = data[f"tau{i}"] * data[f"p{i}"].abs() * data[f"g{i}"]

    return data


X_train = engineer_features(train_df)
X_test = engineer_features(test_df)

# Stratified 5-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

xgb_params = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "gamma": 0.1,
    "random_state": 42,
    "eval_metric": "auc",
    "n_jobs": -1,
    "early_stopping_rounds": 50,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = XGBClassifier(**xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_pred = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_pred

    fold_auc = roc_auc_score(y_va, val_pred)
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold CV ROC AUC: {cv_auc:.6f}")

# Prepare submission
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "stabf": test_preds})
submission.to_csv("./working/submission.csv", index=False)
