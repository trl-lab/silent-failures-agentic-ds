import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Define paths
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
OUTPUT_DIR = "./working"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

X = train_df[feature_cols].copy()
y_raw = train_df[target_col].copy()

# Filter features present in test data
test_feature_cols = [c for c in feature_cols if c in test_df.columns]
X = X[test_feature_cols]
X_test = test_df[test_feature_cols].copy()


# Feature engineering: calculate row-wise summary statistics
def add_row_statistics(df):
    vals = df.values
    feats = pd.DataFrame(index=df.index)
    feats["row_mean"] = np.mean(vals, axis=1)
    feats["row_std"] = np.std(vals, axis=1)
    feats["row_min"] = np.min(vals, axis=1)
    feats["row_max"] = np.max(vals, axis=1)
    feats["row_median"] = np.median(vals, axis=1)
    feats["row_l1"] = np.linalg.norm(vals, ord=1, axis=1)
    feats["row_l2"] = np.linalg.norm(vals, ord=2, axis=1)
    feats["row_ptp"] = feats["row_max"] - feats["row_min"]
    return pd.concat([df, feats], axis=1)


X = add_row_statistics(X)
X_test = add_row_statistics(X_test)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), num_classes))
test_preds = np.zeros((len(test_df), num_classes))

# 5-Fold Stratified Cross-Validation with 5-Model Ensemble
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # CatBoost model
    clf_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.08,
        depth=6,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    clf_cat.fit(
        X_train_scaled,
        y_train,
        eval_set=(X_val_scaled, y_val),
        use_best_model=True,
    )

    # XGBoost model
    clf_xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    clf_xgb.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False,
    )

    # LightGBM model
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        objective="multiclass",
        num_class=num_classes,
        n_jobs=-1,
    )
    clf_lgb.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Logistic Regression model
    clf_lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42 + fold,
        n_jobs=-1,
    )
    clf_lr.fit(X_train_scaled, y_train)

    # Support Vector Classifier (RBF Kernel)
    clf_svc = SVC(
        C=10.0,
        kernel="rbf",
        probability=True,
        random_state=42 + fold,
    )
    clf_svc.fit(X_train_scaled, y_train)

    # Soft voting across all five model architectures
    val_probs = (
        clf_cat.predict_proba(X_val_scaled)
        + clf_xgb.predict_proba(X_val_scaled)
        + clf_lgb.predict_proba(X_val_scaled)
        + clf_lr.predict_proba(X_val_scaled)
        + clf_svc.predict_proba(X_val_scaled)
    ) / 5.0

    fold_test_probs = (
        clf_cat.predict_proba(X_test_scaled)
        + clf_xgb.predict_proba(X_test_scaled)
        + clf_lgb.predict_proba(X_test_scaled)
        + clf_lr.predict_proba(X_test_scaled)
        + clf_svc.predict_proba(X_test_scaled)
    ) / 5.0

    oof_preds[val_idx] = val_probs
    test_preds += fold_test_probs / skf.n_splits

# Overall validation accuracy metric calculation
oof_class_preds = np.argmax(oof_preds, axis=1)
val_accuracy = accuracy_score(y, oof_class_preds)
print(f"Validation Accuracy: {val_accuracy:.5f}")

# Generate test predictions and save submission
final_test_labels = label_encoder.inverse_transform(np.argmax(test_preds, axis=1))
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df)), "class": final_test_labels}
)

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission successfully saved to {SUBMISSION_PATH}")
