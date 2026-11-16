import os
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Random seeds for multi-seed bagging
SEEDS = [42, 1337, 2024]

# Paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUTPUT_DIR = "./working"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "Pawpularity"
base_feature_cols = [
    col for col in train_df.columns if col not in ["Id", "id", target_col]
]


# Feature engineering: compute domain interactions and aggregate indicators
def engineer_features(df, cols):
    df_feat = df[cols].copy()

    # Aggregate count features
    df_feat["active_sum"] = df_feat[cols].sum(axis=1)

    # Domain composition interactions
    if "Subject Focus" in df_feat.columns and "Blur" in df_feat.columns:
        df_feat["focus_clarity"] = (
            df_feat["Subject Focus"] * (1 - df_feat["Blur"])
        ).astype(int)
    if "Face" in df_feat.columns and "Eyes" in df_feat.columns:
        df_feat["face_eyes"] = (df_feat["Face"] * df_feat["Eyes"]).astype(int)
    if "Human" in df_feat.columns and "Occlusion" in df_feat.columns:
        df_feat["human_occlusion"] = (df_feat["Human"] | df_feat["Occlusion"]).astype(
            int
        )
    if "Accessory" in df_feat.columns and "Collage" in df_feat.columns:
        df_feat["accessory_collage"] = (
            df_feat["Accessory"] * df_feat["Collage"]
        ).astype(int)
    if "Group" in df_feat.columns and "Human" in df_feat.columns:
        df_feat["group_human"] = (df_feat["Group"] * df_feat["Human"]).astype(int)

    # Key pairwise interactions among core binary attributes
    for col1, col2 in itertools.combinations(
        ["Subject Focus", "Eyes", "Face", "Near", "Blur"], 2
    ):
        if col1 in df_feat.columns and col2 in df_feat.columns:
            df_feat[f"{col1}_x_{col2}"] = (df_feat[col1] * df_feat[col2]).astype(int)

    return df_feat


X = engineer_features(train_df, base_feature_cols)
X_test = engineer_features(test_df, base_feature_cols)

# Target normalization to [0, 1] for binary cross-entropy loss
y_raw = train_df[target_col].values.astype(np.float64)
y_norm = y_raw / 100.0

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_norm = np.zeros(len(train_df), dtype=np.float64)
test_preds_norm = np.zeros(len(test_df), dtype=np.float64)

for seed in SEEDS:
    lgb_params = {
        "objective": "cross_entropy",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": -1,
    }

    seed_oof = np.zeros(len(train_df), dtype=np.float64)
    seed_test = np.zeros(len(test_df), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_norm)):
        X_train_f, y_train_f = X.iloc[train_idx], y_norm[train_idx]
        X_val_f, y_val_f = X.iloc[val_idx], y_norm[val_idx]

        trn_data = lgb.Dataset(X_train_f, label=y_train_f)
        val_data = lgb.Dataset(X_val_f, label=y_val_f, reference=trn_data)

        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1500,
            valid_sets=[trn_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=60, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        seed_oof[val_idx] = model.predict(X_val_f, num_iteration=model.best_iteration)
        seed_test += (
            model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        )

    oof_preds_norm += seed_oof / len(SEEDS)
    test_preds_norm += seed_test / len(SEEDS)

# Rescale predictions back to original Pawpularity scale [1, 100]
oof_preds = np.clip(oof_preds_norm * 100.0, 1.0, 100.0)
test_preds = np.clip(test_preds_norm * 100.0, 1.0, 100.0)

# Compute Evaluation Metric (RMSE)
cv_rmse = np.sqrt(mean_squared_error(y_raw, oof_preds))
print(f"5-Fold CV Out-of-Fold RMSE: {cv_rmse:.5f}")

# Prepare and save submission
submission_df = pd.DataFrame({"id": np.arange(len(test_df)), "Pawpularity": test_preds})

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(
    f"Submission saved successfully to {SUBMISSION_PATH} with shape {submission_df.shape}"
)
