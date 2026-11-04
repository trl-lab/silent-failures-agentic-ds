import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import os

os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create feature columns
categorical_cols = [
    "affiliate_channel",
    "affiliate_provider",
    "country_destination",
    "date_account_created",
    "date_first_booking",
    "first_affiliate_tracked",
    "first_browser",
    "first_device_type",
    "gender",
    "language",
    "signup_app",
    "signup_method",
]
numerical_cols = ["age", "signup_flow", "timestamp_first_active"]

# Handle categorical encoding
cat_encoder = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)]
)

# Handle numerical columns
num_transformer = Pipeline(steps=[("num", "passthrough")])

# Prepare feature matrix
cat_transformer = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)]
)

# Combine and fit
X_train = cat_transformer.fit_transform(
    train_df.drop("id", axis=1), train_df["country_destination"]
)
X_test = cat_transformer.transform(test_df.drop("id", axis=1))

# Split for validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


# NDCG computation for multi-class
def compute_ndcg(y_true, y_pred):
    """Compute NDCG for multi-class classification"""
    n_classes = len(np.unique(y_true))
    ndcg_scores = []

    for true_label in np.unique(y_true):
        # Get predicted probabilities for this class
        pred_probs = model.predict_proba(X_train)[:, true_label]

        # Create ranking based on predicted probabilities
        ranks = np.argsort(pred_probs)[::-1]

        # Compute DCG
        dcg = 0
        for i, rank in enumerate(ranks):
            if rank == true_label:
                dcg += 1 / (i + 1)

        # Compute ideal DCG
        ideal_dcg = 1 / (1 + 0)  # Perfect ranking

        ndcg_scores.append(dcg / ideal_dcg)

    return np.mean(ndcg_scores)


# Train and evaluate
best_score = 0
best_model = None

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_train, train_df["country_destination"])
):
    X_train_fold = X_train[train_idx]
    y_train_fold = train_df["country_destination"].iloc[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = train_df["country_destination"].iloc[val_idx]

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    clf.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    # Compute NDCG on validation set
    y_pred_proba = clf.predict_proba(X_val_fold)
    ndcg_val = compute_ndcg(y_val_fold, y_pred_proba)

    print(f"Fold {fold+1} NDCG: {ndcg_val:.4f}")

    if ndcg_val > best_score:
        best_score = ndcg_val
        best_model = clf

# Generate submission
y_test_proba = best_model.predict_proba(X_test)
y_test_pred = y_test_proba.argmax(axis=1)

# Create submission file
submission = pd.DataFrame(
    {"id": test_df["id"].values, "country_destination": y_test_pred}
)
submission.to_csv("./working/submission.csv", index=False)

print(f"\nBest validation NDCG: {best_score:.4f}")
print("Submission saved to ./working/submission.csv")
