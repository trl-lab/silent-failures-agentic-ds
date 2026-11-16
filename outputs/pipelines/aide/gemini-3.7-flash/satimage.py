import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]


# 2. Feature Engineering
def engineer_features(df):
    X = df[feature_cols].copy()
    raw_vals = X.values  # Shape: (N, 36) - 9 pixels x 4 bands

    # Reshape to (N, 9, 4) -> 9 pixels, 4 bands each
    pixels = raw_vals.reshape(-1, 9, 4)

    # 4 spectral bands across 9 pixels
    for b in range(4):
        band_vals = pixels[:, :, b]
        X[f"band_{b}_mean"] = np.mean(band_vals, axis=1)
        X[f"band_{b}_std"] = np.std(band_vals, axis=1)
        X[f"band_{b}_min"] = np.min(band_vals, axis=1)
        X[f"band_{b}_max"] = np.max(band_vals, axis=1)
        # Center pixel is pixel index 4 (0-indexed 5th pixel)
        center_val = pixels[:, 4, b]
        X[f"band_{b}_center"] = center_val
        X[f"band_{b}_center_diff_mean"] = center_val - X[f"band_{b}_mean"]

    # Spectral indices on central pixel (Band 0: Green, Band 1: Red, Band 2: NIR1, Band 3: NIR2)
    b0 = pixels[:, 4, 0].astype(float)
    b1 = pixels[:, 4, 1].astype(float)
    b2 = pixels[:, 4, 2].astype(float)
    b3 = pixels[:, 4, 3].astype(float)
    eps = 1e-6

    X["ndvi_like_21"] = (b2 - b1) / (b2 + b1 + eps)
    X["ndvi_like_31"] = (b3 - b1) / (b3 + b1 + eps)
    X["ratio_2_1"] = b2 / (b1 + eps)
    X["ratio_3_1"] = b3 / (b1 + eps)
    X["ratio_2_0"] = b2 / (b0 + eps)
    X["diff_3_2"] = b3 - b2

    return X


X_train = engineer_features(train_df)
X_test = engineer_features(test_df)

# Encode targets
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col])
n_classes = len(le.classes_)

# 3. Model Training & 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), n_classes))
test_preds = np.zeros((len(test_df), n_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
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

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )

# 4. Evaluation
oof_class_preds = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y_train, oof_class_preds)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# 5. Generate Submission
final_test_labels = le.inverse_transform(np.argmax(test_preds, axis=1))

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_labels)), "class": final_test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print(f"Submission successfully saved with shape: {submission.shape}")
