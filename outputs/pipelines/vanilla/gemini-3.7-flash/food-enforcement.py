import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# 1. Load data
train_path = (
    "/path/to/user/workspace/ds_failures/datasets/strable/food-enforcement/data/train.parquet"
)
test_path = (
    "/path/to/user/workspace/ds_failures/datasets/strable/food-enforcement/data/test.parquet"
)

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "classification"

# 2. Text feature preparation
text_cols = [
    "reason_for_recall",
    "product_description",
    "distribution_pattern",
    "recalling_firm",
    "initial_firm_notification",
    "voluntary_mandated",
    "status",
    "code_info",
]


def extract_features(df):
    data = pd.DataFrame()
    for col in text_cols:
        if col in df.columns:
            data[col] = df[col].fillna("").astype(str)
        else:
            data[col] = ""

    # Combined text field for rich context
    data["combined_text"] = (
        data["reason_for_recall"]
        + " "
        + data["product_description"]
        + " "
        + data["distribution_pattern"]
        + " "
        + data["recalling_firm"]
    )
    return data


X_train = extract_features(train_df)
X_test = extract_features(test_df)

le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].astype(str))

# 3. Build Preprocessor and Model Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "tfidf_reason",
            TfidfVectorizer(
                ngram_range=(1, 3), max_features=30000, sublinear_tf=True
            ),
            "reason_for_recall",
        ),
        (
            "tfidf_desc",
            TfidfVectorizer(
                ngram_range=(1, 2), max_features=20000, sublinear_tf=True
            ),
            "product_description",
        ),
        (
            "tfidf_combined",
            TfidfVectorizer(
                ngram_range=(1, 2), max_features=30000, sublinear_tf=True
            ),
            "combined_text",
        ),
        (
            "tfidf_firm",
            TfidfVectorizer(
                ngram_range=(1, 1), max_features=5000, sublinear_tf=True
            ),
            "recalling_firm",
        ),
        (
            "tfidf_dist",
            TfidfVectorizer(
                ngram_range=(1, 2), max_features=5000, sublinear_tf=True
            ),
            "distribution_pattern",
        ),
    ]
)

clf1 = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
clf2 = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-5,
    max_iter=1000,
    random_state=42,
)
clf3 = LogisticRegression(C=1.0, max_iter=1000, random_state=43)

ensemble = VotingClassifier(
    estimators=[("lr1", clf1), ("sgd", clf2), ("lr2", clf3)], voting="soft"
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("ensemble", ensemble)])

# 4. Stratified K-Fold Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []

for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_va)
    val_scores.append(accuracy_score(y_va, preds))

# 5. Fit full dataset and generate predictions
pipeline.fit(X_train, y_train)
test_preds_encoded = pipeline.predict(X_test)
test_preds = le.inverse_transform(test_preds_encoded)

# 6. Save predictions
submission_df = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "classification": test_preds}
)
submission_df.to_csv("predictions.csv", index=False)