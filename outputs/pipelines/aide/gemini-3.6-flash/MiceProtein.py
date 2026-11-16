import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load train and test data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Separate target and identification columns
target_col = "class"
y = train[target_col]

drop_cols = [target_col, "Mouse ID", "id"]
feature_cols = [c for c in train.columns if c not in drop_cols and c in test.columns]

categorical_cols = [
    c
    for c in feature_cols
    if train[c].dtype == "object" or train[c].dtype.name == "category"
]
numerical_cols = [c for c in feature_cols if c not in categorical_cols]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)


# Feature engineering function
def extract_features(df):
    df_feat = df[feature_cols].copy()

    # Process numeric block for row statistics
    num_df = df_feat[numerical_cols].copy()

    df_feat["num_missing"] = num_df.isnull().sum(axis=1)
    df_feat["row_mean"] = num_df.mean(axis=1)
    df_feat["row_std"] = num_df.std(axis=1)
    df_feat["row_min"] = num_df.min(axis=1)
    df_feat["row_max"] = num_df.max(axis=1)
    df_feat["row_median"] = num_df.median(axis=1)

    q25 = num_df.quantile(0.25, axis=1)
    q75 = num_df.quantile(0.75, axis=1)
    df_feat["row_iqr"] = q75 - q25

    # Factorize categorical features if present
    for col in categorical_cols:
        df_feat[col] = pd.factorize(df_feat[col].astype(str))[0]

    return df_feat


X_train_df = extract_features(train)
X_test_df = extract_features(test)

X_train = X_train_df.values
X_test = X_test_df.values

# Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y_encoded), dtype=int)
test_preds_proba = np.zeros((len(X_test), len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_encoded)):
    X_tr, y_tr = X_train[train_idx], y_encoded[train_idx]
    X_val, y_val = X_train[val_idx], y_encoded[val_idx]

    # Standardize features prior to KNN Imputation so all features share equal scale
    scaler = StandardScaler()
    X_tr_scaled_raw = scaler.fit_transform(X_tr)
    X_val_scaled_raw = scaler.transform(X_val)
    X_test_scaled_raw = scaler.transform(X_test)

    # Impute missing values on standardized features using KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    X_tr_scaled = imputer.fit_transform(X_tr_scaled_raw)
    X_val_scaled = imputer.transform(X_val_scaled_raw)
    X_test_scaled = imputer.transform(X_test_scaled_raw)

    # Diversity Ensemble components
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        random_state=42 + fold,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        random_state=42 + fold,
        n_jobs=-1,
    )
    svc = SVC(
        probability=True,
        class_weight="balanced",
        C=1.5,
        random_state=42 + fold,
    )
    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42 + fold
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42 + fold,
        early_stopping=True,
    )

    rf.fit(X_tr_scaled, y_tr)
    et.fit(X_tr_scaled, y_tr)
    svc.fit(X_tr_scaled, y_tr)
    lr.fit(X_tr_scaled, y_tr)
    mlp.fit(X_tr_scaled, y_tr)

    val_proba = (
        rf.predict_proba(X_val_scaled)
        + et.predict_proba(X_val_scaled)
        + svc.predict_proba(X_val_scaled)
        + lr.predict_proba(X_val_scaled)
        + mlp.predict_proba(X_val_scaled)
    ) / 5.0

    test_proba = (
        rf.predict_proba(X_test_scaled)
        + et.predict_proba(X_test_scaled)
        + svc.predict_proba(X_test_scaled)
        + lr.predict_proba(X_test_scaled)
        + mlp.predict_proba(X_test_scaled)
    ) / 5.0

    val_pred = np.argmax(val_proba, axis=1)
    oof_preds[val_idx] = val_pred

    test_preds_proba += test_proba / skf.n_splits

# Compute Out-Of-Fold accuracy score
oof_acc = accuracy_score(y_encoded, oof_preds)
print(f"Validation Accuracy: {oof_acc:.5f}")

# Convert test probability predictions to original class labels
test_preds = np.argmax(test_preds_proba, axis=1)
test_preds_labels = le.inverse_transform(test_preds)

# Save test predictions in required format
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds_labels)), "class": test_preds_labels}
)
submission.to_csv("./working/submission.csv", index=False)
