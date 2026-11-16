import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import os
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Create working directory if it doesn't exist
os.makedirs("./working", exist_ok=True)

# Feature engineering
# Extract date features
train_df["date_added"] = pd.to_datetime(train_df["date_added"])
train_df["date_updated"] = pd.to_datetime(train_df["date_updated"])
train_df["started_at"] = pd.to_datetime(train_df["started_at"])
train_df["read_at"] = pd.to_datetime(train_df["read_at"])

train_df["date_added_year"] = train_df["date_added"].dt.year
train_df["date_added_month"] = train_df["date_added"].dt.month
train_df["date_added_day"] = train_df["date_added"].dt.day

train_df["date_updated_year"] = train_df["date_updated"].dt.year
train_df["date_updated_month"] = train_df["date_updated"].dt.month
train_df["date_updated_day"] = train_df["date_updated"].dt.day

train_df["started_at_year"] = train_df["started_at"].dt.year
train_df["started_at_month"] = train_df["started_at"].dt.month
train_df["started_at_day"] = train_df["started_at"].dt.day

train_df["read_at_year"] = train_df["read_at"].dt.year
train_df["read_at_month"] = train_df["read_at"].dt.month
train_df["read_at_day"] = train_df["read_at"].dt.day

# Encode categorical features
le_user_id = LabelEncoder()
le_book_id = LabelEncoder()
le_review_id = LabelEncoder()

train_df["user_id_encoded"] = le_user_id.fit_transform(train_df["user_id"].astype(str))
train_df["book_id_encoded"] = le_book_id.fit_transform(train_df["book_id"].astype(str))
train_df["review_id_encoded"] = le_review_id.fit_transform(
    train_df["review_id"].astype(str)
)

# Fill missing values
train_df["n_comments"] = train_df["n_comments"].fillna(train_df["n_comments"].median())
train_df["n_votes"] = train_df["n_votes"].fillna(train_df["n_votes"].median())
train_df["review_text"] = train_df["review_text"].fillna("")

# Create feature matrix
feature_cols = [
    "n_comments",
    "n_votes",
    "date_added_year",
    "date_added_month",
    "date_added_day",
    "date_updated_year",
    "date_updated_month",
    "date_updated_day",
    "started_at_year",
    "started_at_month",
    "started_at_day",
    "read_at_year",
    "read_at_month",
    "read_at_day",
    "user_id_encoded",
    "book_id_encoded",
    "review_id_encoded",
]

X_train = train_df[feature_cols].copy()
y_train = train_df["rating"].copy()
X_train["review_text"] = X_train["review_text"]

# Handle review text with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_text = vectorizer.fit_transform(X_train["review_text"])
X_train = pd.DataFrame(
    X_train_text.toarray(), columns=[f"tfidf_{i}" for i in range(X_train_text.shape[1])]
)
X_train = X_train.join(X_train[feature_cols])

# Split data for validation
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Prepare LightGBM data
train_data = lgb.Dataset(X_train_val, label=y_train_val)
valid_data = lgb.Dataset(X_test_val, label=y_test_val, reference=train_data)

# Train LightGBM model
params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "max_depth": -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    valid_sets=[valid_data],
    keep_training=True,
)

# Get predictions on validation set
val_predictions = model.predict(X_test_val)
val_rmse = np.sqrt(mean_squared_error(y_test_val, val_predictions))
print(f"Validation RMSE: {val_rmse:.4f}")

# Save final predictions for test set
test_df["id"] = range(len(test_df))
X_test = test_df[feature_cols].copy()
X_test["review_text"] = test_df["review_text"]

# Handle review text with same vectorizer
X_test_text = vectorizer.transform(X_test["review_text"])
X_test = pd.DataFrame(
    X_test_text.toarray(), columns=[f"tfidf_{i}" for i in range(X_test_text.shape[1])]
)
X_test = X_test.join(X_test[feature_cols])

test_predictions = model.predict(X_test)
test_df["rating"] = test_predictions

# Save submission
test_df[["id", "rating"]].to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
print(f"Final RMSE on validation: {val_rmse:.4f}")
