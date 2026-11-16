import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Set random seed
SEED = 42
np.random.seed(SEED)

# Paths
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "class"
ignore_cols = {target_col, "id"}
feature_cols = [c for c in train_df.columns if c not in ignore_cols]


def engineer_features(df: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    X = df[sensor_cols].astype(np.float32).copy()
    features = pd.DataFrame(index=df.index)

    # 1. Raw sensor features
    for col in sensor_cols:
        features[col] = X[col]

    sensor_matrix = X.values
    n_sensors = len(sensor_cols)

    # 2. Overall summary statistics across circular sensors
    features["sensor_min"] = sensor_matrix.min(axis=1)
    features["sensor_max"] = sensor_matrix.max(axis=1)
    features["sensor_mean"] = sensor_matrix.mean(axis=1)
    features["sensor_std"] = sensor_matrix.std(axis=1)
    features["sensor_argmin"] = sensor_matrix.argmin(axis=1)
    features["sensor_argmax"] = sensor_matrix.argmax(axis=1)

    if n_sensors == 24:
        # Sector definitions
        front_idx = [22, 23, 0, 1, 2]
        right_idx = [4, 5, 6, 7, 8]
        back_idx = [10, 11, 12, 13, 14]
        left_idx = [16, 17, 18, 19, 20]

        # Quadrant definitions
        front_right_idx = [0, 1, 2, 3, 4, 5]
        back_right_idx = [6, 7, 8, 9, 10, 11]
        back_left_idx = [12, 13, 14, 15, 16, 17]
        front_left_idx = [18, 19, 20, 21, 22, 23]

        features["front_min"] = sensor_matrix[:, front_idx].min(axis=1)
        features["front_mean"] = sensor_matrix[:, front_idx].mean(axis=1)
        features["right_min"] = sensor_matrix[:, right_idx].min(axis=1)
        features["right_mean"] = sensor_matrix[:, right_idx].mean(axis=1)
        features["back_min"] = sensor_matrix[:, back_idx].min(axis=1)
        features["back_mean"] = sensor_matrix[:, back_idx].mean(axis=1)
        features["left_min"] = sensor_matrix[:, left_idx].min(axis=1)
        features["left_mean"] = sensor_matrix[:, left_idx].mean(axis=1)

        features["front_right_min"] = sensor_matrix[:, front_right_idx].min(axis=1)
        features["front_left_min"] = sensor_matrix[:, front_left_idx].min(axis=1)
        features["back_right_min"] = sensor_matrix[:, back_right_idx].min(axis=1)
        features["back_left_min"] = sensor_matrix[:, back_left_idx].min(axis=1)

        # Directional comparisons
        features["diff_front_back"] = features["front_mean"] - features["back_mean"]
        features["diff_left_right"] = features["left_mean"] - features["right_mean"]
        features["min_diff_left_right"] = features["left_min"] - features["right_min"]
        features["min_diff_front_right_left"] = (
            features["front_right_min"] - features["front_left_min"]
        )
        features["ratio_left_right"] = (features["left_min"] + 1e-4) / (
            features["right_min"] + 1e-4
        )
        features["ratio_front_sides"] = (features["front_min"] + 1e-4) / (
            (features["left_min"] + features["right_min"]) / 2.0 + 1e-4
        )

        # Circular perimeter 1st and 2nd differences (gradients and curvature)
        grads = []
        for i in range(24):
            next_i = (i + 1) % 24
            g = sensor_matrix[:, next_i] - sensor_matrix[:, i]
            features[f"grad_{i}_{next_i}"] = g
            grads.append(g)

        for i in range(24):
            next_i = (i + 1) % 24
            features[f"curv_{i}"] = grads[next_i] - grads[i]

        # Bilateral symmetry differences (left vs right mirror sensors)
        for i in range(1, 12):
            left_mirror = (24 - i) % 24
            right_mirror = i
            features[f"sym_diff_{left_mirror}_{right_mirror}"] = (
                sensor_matrix[:, left_mirror] - sensor_matrix[:, right_mirror]
            )

        # 2D Obstacle Repulsion Force Vector (Cartesian coordinates)
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        inv_dist = 1.0 / (sensor_matrix + 0.05)
        features["repulsion_fx"] = np.sum(inv_dist * cos_angles, axis=1)
        features["repulsion_fy"] = np.sum(inv_dist * sin_angles, axis=1)
        features["repulsion_mag"] = np.sqrt(
            features["repulsion_fx"] ** 2 + features["repulsion_fy"] ** 2
        )

        # Rolling circular minimums with window size 3
        for i in range(24):
            win_idx = [(i - 1) % 24, i, (i + 1) % 24]
            features[f"roll_min_{i}"] = sensor_matrix[:, win_idx].min(axis=1)

    return features


# Feature extraction
X_train = engineer_features(train_df, feature_cols)
X_test = engineer_features(test_df, feature_cols)

# Encode targets
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col])
num_classes = len(le.classes_)

# Cross-Validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_preds_lgb = np.zeros((len(train_df), num_classes))
oof_preds_xgb = np.zeros((len(train_df), num_classes))
oof_preds_cat = np.zeros((len(train_df), num_classes))
oof_preds_et = np.zeros((len(train_df), num_classes))

test_preds_lgb = np.zeros((len(test_df), num_classes))
test_preds_xgb = np.zeros((len(test_df), num_classes))
test_preds_cat = np.zeros((len(test_df), num_classes))
test_preds_et = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

    # 1. LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_val)
    test_preds_lgb += lgb_model.predict_proba(X_test) / n_splits

    # 2. XGBoost Classifier
    xgb_model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=num_classes,
        random_state=SEED + fold,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        tree_method="hist",
    )
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_preds_xgb[val_idx] = xgb_model.predict_proba(X_val)
    test_preds_xgb += xgb_model.predict_proba(X_test) / n_splits

    # 3. CatBoost Classifier
    cat_model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.04,
        depth=6,
        random_seed=SEED + fold,
        verbose=False,
        thread_count=-1,
    )
    cat_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )
    oof_preds_cat[val_idx] = cat_model.predict_proba(X_val)
    test_preds_cat += cat_model.predict_proba(X_test) / n_splits

    # 4. Extra Trees Classifier
    et_model = ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_split=2,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    oof_preds_et[val_idx] = et_model.predict_proba(X_val)
    test_preds_et += et_model.predict_proba(X_test) / n_splits

# Multi-model Ensemble predictions (weighted average)
w_lgb, w_xgb, w_cat, w_et = 0.35, 0.35, 0.20, 0.10

oof_preds = (
    w_lgb * oof_preds_lgb
    + w_xgb * oof_preds_xgb
    + w_cat * oof_preds_cat
    + w_et * oof_preds_et
)
test_preds = (
    w_lgb * test_preds_lgb
    + w_xgb * test_preds_xgb
    + w_cat * test_preds_cat
    + w_et * test_preds_et
)

oof_class_preds = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y_train, oof_class_preds)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# Generate test predictions and save submission
test_class_indices = np.argmax(test_preds, axis=1)
test_class_labels = le.inverse_transform(test_class_indices)

submission = pd.DataFrame({"id": np.arange(len(test_df)), "class": test_class_labels})
submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
print("Submission successfully saved to ./working/submission.csv")
