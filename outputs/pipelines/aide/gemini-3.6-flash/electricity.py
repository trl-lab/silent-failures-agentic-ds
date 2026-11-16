import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Set directory paths
INPUT_DIR = "./input"
OUTPUT_DIR = "./working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

# Read dataset
train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

# Handle test IDs
if "id" in test.columns:
    test_ids = test["id"].values
else:
    test_ids = np.arange(len(test))

# Encode target
target_col = "class"
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train[target_col])

# Combine dataset for feature engineering
train_len = len(train)
df = pd.concat(
    [
        train.drop(columns=[target_col], errors="ignore"),
        test.drop(columns=["id"], errors="ignore"),
    ],
    axis=0,
    ignore_index=True,
)


def engineer_features(data):
    df_feat = data.copy()

    # Domain price and demand differentials
    if "NSWprice" in df_feat.columns and "VICprice" in df_feat.columns:
        df_feat["price_diff"] = df_feat["NSWprice"] - df_feat["VICprice"]
        df_feat["price_ratio"] = (df_feat["NSWprice"] + 1e-5) / (
            df_feat["VICprice"] + 1e-5
        )

    if "NSWdemand" in df_feat.columns and "VICdemand" in df_feat.columns:
        df_feat["demand_diff"] = df_feat["NSWdemand"] - df_feat["VICdemand"]
        df_feat["demand_ratio"] = (df_feat["NSWdemand"] + 1e-5) / (
            df_feat["VICdemand"] + 1e-5
        )
        df_feat["total_demand"] = df_feat["NSWdemand"] + df_feat["VICdemand"]

    # Transfer interaction signals
    if "transfer" in df_feat.columns:
        if "NSWdemand" in df_feat.columns:
            df_feat["nsw_transfer_ratio"] = df_feat["transfer"] / (
                df_feat["NSWdemand"] + 1e-5
            )
        if "VICdemand" in df_feat.columns:
            df_feat["vic_transfer_ratio"] = df_feat["transfer"] / (
                df_feat["VICdemand"] + 1e-5
            )
        if "price_diff" in df_feat.columns:
            df_feat["transfer_price_interaction"] = (
                df_feat["transfer"] * df_feat["price_diff"]
            )

    # Cyclical temporal transformations
    if "Period" in df_feat.columns:
        df_feat["period_sin"] = np.sin(2 * np.pi * df_feat["Period"])
        df_feat["period_cos"] = np.cos(2 * np.pi * df_feat["Period"])
    if "Day" in df_feat.columns:
        df_feat["day_sin"] = np.sin(2 * np.pi * df_feat["Day"] / 7.0)
        df_feat["day_cos"] = np.cos(2 * np.pi * df_feat["Day"] / 7.0)

    # Order-independent period and day baseline features
    base_cols = [
        c
        for c in ["NSWprice", "NSWdemand", "VICprice", "VICdemand", "transfer"]
        if c in df_feat.columns
    ]
    for col in base_cols:
        if "Period" in df_feat.columns:
            period_mean = df_feat.groupby("Period")[col].transform("mean")
            df_feat[f"{col}_period_diff"] = df_feat[col] - period_mean
            period_std = df_feat.groupby("Period")[col].transform("std").fillna(1e-5)
            df_feat[f"{col}_period_z"] = (df_feat[col] - period_mean) / (
                period_std + 1e-5
            )

        if "Day" in df_feat.columns and "Period" in df_feat.columns:
            day_period_mean = df_feat.groupby(["Day", "Period"])[col].transform("mean")
            df_feat[f"{col}_day_period_diff"] = df_feat[col] - day_period_mean

    return df_feat


df_engineered = engineer_features(df)

X_train = df_engineered.iloc[:train_len].reset_index(drop=True)
X_test = df_engineered.iloc[train_len:].reset_index(drop=True)

drop_cols = ["id"]
X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

# Cross-Validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "binary",
    "metric": "binary_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1200,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # Model 1: LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    val_probs_lgb = model_lgb.predict_proba(X_va)[:, 1]
    test_probs_lgb = model_lgb.predict_proba(X_test)[:, 1]

    # Model 2: HistGradientBoosting
    model_hgb = HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=63,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=30,
    )
    model_hgb.fit(X_tr, y_tr)
    val_probs_hgb = model_hgb.predict_proba(X_va)[:, 1]
    test_probs_hgb = model_hgb.predict_proba(X_test)[:, 1]

    # Model 3: ExtraTrees
    model_et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=22,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42,
    )
    model_et.fit(X_tr, y_tr)
    val_probs_et = model_et.predict_proba(X_va)[:, 1]
    test_probs_et = model_et.predict_proba(X_test)[:, 1]

    # Ensemble blending
    val_blend = 0.50 * val_probs_lgb + 0.30 * val_probs_hgb + 0.20 * val_probs_et
    test_blend = 0.50 * test_probs_lgb + 0.30 * test_probs_hgb + 0.20 * test_probs_et

    oof_preds[val_idx] = val_blend
    test_preds += test_blend / skf.n_splits

# Evaluation
oof_class_preds = (oof_preds >= 0.5).astype(int)
acc = accuracy_score(y, oof_class_preds)
print(f"Validation Accuracy: {acc:.5f}")

# Generate Submission File
test_class_preds_encoded = (test_preds >= 0.5).astype(int)
test_class_preds = label_encoder.inverse_transform(test_class_preds_encoded)

submission = pd.DataFrame({"id": test_ids, "class": test_class_preds})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
