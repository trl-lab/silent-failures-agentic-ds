import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target and features
target_col = "class"
ignore_cols = [target_col]

# Check if there is an explicit identifier column
id_cols = [c for c in ["id", "ID", "MouseID", "Mouse ID"] if c in train_df.columns]
ignore_cols.extend(id_cols)

feature_cols = [c for c in train_df.columns if c not in ignore_cols]

# Separate features and target
X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle categorical features if any exist in the predictor set
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    le_cat = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le_cat.fit(combined)
    X[col] = le_cat.transform(X[col].astype(str))
    X_test[col] = le_cat.transform(X_test[col].astype(str))

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# 2. Imputation with KNN and Scaling
imputer = KNNImputer(n_neighbors=5, weights="distance")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 3. Stratified 5-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds_probs = np.zeros((len(train_df), num_classes))
test_preds_probs = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_imputed, y_encoded)):
    X_train_imp, y_train = X_imputed.iloc[train_idx], y_encoded[train_idx]
    X_val_imp, y_val = X_imputed.iloc[val_idx], y_encoded[val_idx]

    X_train_scl = X_scaled[train_idx]
    X_val_scl = X_scaled[val_idx]

    # Model 1: LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    lgb_clf.fit(
        X_train_imp,
        y_train,
        eval_set=[(X_val_imp, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    val_lgb_probs = lgb_clf.predict_proba(X_val_imp)
    test_lgb_probs = lgb_clf.predict_proba(X_test_imputed)

    # Model 2: ExtraTrees
    et_clf = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=2,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et_clf.fit(X_train_imp, y_train)
    val_et_probs = et_clf.predict_proba(X_val_imp)
    test_et_probs = et_clf.predict_proba(X_test_imputed)

    # Model 3: Support Vector Classifier (RBF Kernel)
    svc_clf = SVC(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        random_state=42 + fold,
    )
    svc_clf.fit(X_train_scl, y_train)
    val_svc_probs = svc_clf.predict_proba(X_val_scl)
    test_svc_probs = svc_clf.predict_proba(X_test_scaled)

    # Model 4: Regularized Logistic Regression
    lr_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42 + fold)
    lr_clf.fit(X_train_scl, y_train)
    val_lr_probs = lr_clf.predict_proba(X_val_scl)
    test_lr_probs = lr_clf.predict_proba(X_test_scaled)

    # Blend probabilities
    val_blend = (
        0.30 * val_lgb_probs
        + 0.30 * val_et_probs
        + 0.25 * val_svc_probs
        + 0.15 * val_lr_probs
    )
    test_blend = (
        0.30 * test_lgb_probs
        + 0.30 * test_et_probs
        + 0.25 * test_svc_probs
        + 0.15 * test_lr_probs
    )

    oof_preds_probs[val_idx] = val_blend
    test_preds_probs += test_blend / n_splits

# 4. Evaluation
oof_predictions = np.argmax(oof_preds_probs, axis=1)
cv_accuracy = accuracy_score(y_encoded, oof_predictions)
print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.5f}")

# 5. Generate Test Predictions & Save Submission
test_predictions_encoded = np.argmax(test_preds_probs, axis=1)
test_predictions_labels = label_encoder.inverse_transform(test_predictions_encoded)

os.makedirs("./working", exist_ok=True)
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_df)), "class": test_predictions_labels}
)
submission_df.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
