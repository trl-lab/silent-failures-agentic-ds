import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate features and target
y = train_df["Company response to consumer"]
X_train = train_df.drop("Company response to consumer", axis=1)
X_test = test_df

# Identify feature types
cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()
num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_features,
        ),
        ("num", "passthrough", num_features),
    ]
)

# Create LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Create full pipeline
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y, cv=kf, scoring="accuracy")
print(
    f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
)

# Train final model on all training data
pipeline.fit(
    X_train,
    y,
    eval_set=[(X_train, y)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions for test set
y_pred = pipeline.predict(X_test)

# Create submission file
submission = pd.DataFrame(
    {"id": range(len(y_pred)), "Company response to consumer": y_pred}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved to ./working/submission.csv")
