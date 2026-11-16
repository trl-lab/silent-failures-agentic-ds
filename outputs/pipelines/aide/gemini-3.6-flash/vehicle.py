import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Clean column names
train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns = [c.strip() for c in test_df.columns]

target_col = "class"
feature_cols = [c for c in train_df.columns if c not in [target_col, "id"]]

# Fit LabelEncoder on target
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))


# Domain feature engineering function
def engineer_features(df, cols):
    X = df[cols].copy()
    eps = 1e-6

    # Ratios and products of geometric shape properties
    if "PR.AXIS ASPECT RATIO" in X.columns and "MAX.LENGTH ASPECT RATIO" in X.columns:
        X["ASPECT_RATIO_PROD"] = (
            X["PR.AXIS ASPECT RATIO"] * X["MAX.LENGTH ASPECT RATIO"]
        )
        X["ASPECT_RATIO_RATIO"] = X["PR.AXIS ASPECT RATIO"] / (
            X["MAX.LENGTH ASPECT RATIO"] + eps
        )

    if (
        "SCALED VARIANCE ALONG MAJOR AXIS" in X.columns
        and "SCALED VARIANCE ALONG MINOR AXIS" in X.columns
    ):
        X["VAR_RATIO"] = X["SCALED VARIANCE ALONG MAJOR AXIS"] / (
            X["SCALED VARIANCE ALONG MINOR AXIS"] + eps
        )
        X["VAR_DIFF"] = (
            X["SCALED VARIANCE ALONG MAJOR AXIS"]
            - X["SCALED VARIANCE ALONG MINOR AXIS"]
        )
        X["VAR_SUM"] = (
            X["SCALED VARIANCE ALONG MAJOR AXIS"]
            + X["SCALED VARIANCE ALONG MINOR AXIS"]
        )

    if (
        "PR.AXIS RECTANGULARITY" in X.columns
        and "MAX.LENGTH RECTANGULARITY" in X.columns
    ):
        X["RECT_RATIO"] = X["PR.AXIS RECTANGULARITY"] / (
            X["MAX.LENGTH RECTANGULARITY"] + eps
        )
        X["RECT_DIFF"] = X["PR.AXIS RECTANGULARITY"] - X["MAX.LENGTH RECTANGULARITY"]

    if "COMPACTNESS" in X.columns and "CIRCULARITY" in X.columns:
        X["COMPACT_CIRC_RATIO"] = X["COMPACTNESS"] / (X["CIRCULARITY"] + eps)

    if (
        "SKEWNESS ABOUT MAJOR AXIS" in X.columns
        and "SKEWNESS ABOUT MINOR AXIS" in X.columns
    ):
        X["SKEW_RATIO"] = X["SKEWNESS ABOUT MAJOR AXIS"] / (
            X["SKEWNESS ABOUT MINOR AXIS"] + eps
        )

    if (
        "KURTOSIS ABOUT MAJOR AXIS" in X.columns
        and "KURTOSIS ABOUT MINOR AXIS" in X.columns
    ):
        X["KURT_RATIO"] = X["KURTOSIS ABOUT MAJOR AXIS"] / (
            X["KURTOSIS ABOUT MINOR AXIS"] + eps
        )

    # Statistical summary features across geometric features
    numeric_cols = [
        c
        for c in X.columns
        if X[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    X["ROW_MEAN"] = X[numeric_cols].mean(axis=1)
    X["ROW_STD"] = X[numeric_cols].std(axis=1)
    X["ROW_MAX"] = X[numeric_cols].max(axis=1)
    X["ROW_MIN"] = X[numeric_cols].min(axis=1)

    # Handle potential non-finite values from ratio divisions
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X


X_train = engineer_features(train_df, feature_cols)
X_test = engineer_features(test_df, feature_cols)

# Ensure columns alignment
X_test = X_test[X_train.columns]

# 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X_train))
oof_probs = np.zeros((len(X_train), len(le.classes_)))
test_probs = np.zeros((len(X_test), len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # Model 1: RBF Support Vector Classifier
    pipeline_svc = Pipeline(
        [
            ("power", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler()),
            ("svc", SVC(C=5.0, kernel="rbf", probability=True, random_state=SEED)),
        ]
    )

    # Model 2: Multi-Layer Perceptron with L-BFGS optimization
    pipeline_mlp = Pipeline(
        [
            ("power", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    solver="lbfgs",
                    alpha=0.01,
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    # Fit models inside fold to avoid data leakage
    pipeline_svc.fit(X_tr, y_tr)
    pipeline_mlp.fit(X_tr, y_tr)

    # Predict validation probabilities
    prob_svc_val = pipeline_svc.predict_proba(X_va)
    prob_mlp_val = pipeline_mlp.predict_proba(X_va)
    blended_val_prob = 0.5 * prob_svc_val + 0.5 * prob_mlp_val

    oof_probs[val_idx] = blended_val_prob
    oof_preds[val_idx] = np.argmax(blended_val_prob, axis=1)

    # Predict test probabilities
    prob_svc_test = pipeline_svc.predict_proba(X_test)
    prob_mlp_test = pipeline_mlp.predict_proba(X_test)
    blended_test_prob = 0.5 * prob_svc_test + 0.5 * prob_mlp_test

    test_probs += blended_test_prob / skf.n_splits

cv_accuracy = accuracy_score(y, oof_preds)
print(f"Validation Metric (5-Fold Stratified CV Accuracy): {cv_accuracy:.4f}")

# Prepare submission format
os.makedirs("./working", exist_ok=True)
test_class_indices = np.argmax(test_probs, axis=1)
test_class_labels = le.inverse_transform(test_class_indices)

sub = pd.DataFrame(
    {"id": np.arange(len(test_class_labels)), "class": test_class_labels}
)

sub.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
