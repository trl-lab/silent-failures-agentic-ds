import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

# Create working directory if needed
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
feature_cols = [col for col in train_df.columns if col != target_col]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df[target_col].astype(str))
X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# Ensure numeric types and handle any missing values
X = X.apply(pd.to_numeric, errors="coerce").fillna(0).values
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0).values

num_classes = len(label_encoder.classes_)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), num_classes))
test_preds = np.zeros((len(test_df), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_f, y_train_f = X[train_idx], y[train_idx]
    X_val_f, y_val_f = X[val_idx], y[val_idx]

    # 1. LightGBM Model
    trn_data = lgb.Dataset(X_train_f, label=y_train_f)
    val_data = lgb.Dataset(X_val_f, label=y_val_f, reference=trn_data)

    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1000,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    val_pred_lgb = model_lgb.predict(X_val_f, num_iteration=model_lgb.best_iteration)
    test_pred_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)

    # 2. Extra Trees Classifier
    model_et = ExtraTreesClassifier(
        n_estimators=300, max_depth=None, random_state=42 + fold, n_jobs=-1
    )
    model_et.fit(X_train_f, y_train_f)
    val_pred_et = model_et.predict_proba(X_val_f)
    test_pred_et = model_et.predict_proba(X_test)

    # 3. Scaled Logistic Regression / Support Vector Model
    model_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42 + fold, n_jobs=-1),
    )
    model_lr.fit(X_train_f, y_train_f)
    val_pred_lr = model_lr.predict_proba(X_val_f)
    test_pred_lr = model_lr.predict_proba(X_test)

    # Weighted Ensemble of the 3 complementary models
    val_pred_fold = 0.40 * val_pred_lgb + 0.35 * val_pred_et + 0.25 * val_pred_lr
    test_pred_fold = 0.40 * test_pred_lgb + 0.35 * test_pred_et + 0.25 * test_pred_lr

    oof_preds[val_idx] = val_pred_fold
    test_preds += test_pred_fold / n_splits

# Compute overall CV accuracy
oof_class_preds = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y, oof_class_preds)
print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.5f}")

# Generate test predictions
final_test_classes = np.argmax(test_preds, axis=1)
original_test_labels = label_encoder.inverse_transform(final_test_classes)

# Format and save submission
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "class": original_test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv successfully.")
