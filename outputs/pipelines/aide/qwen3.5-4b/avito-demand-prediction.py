import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify categorical and numeric columns
categorical_cols = [
    "activation_date",
    "category_name",
    "city",
    "description",
    "image",
    "item_id",
    "param_1",
    "param_2",
    "param_3",
    "parent_category_name",
    "region",
    "title",
    "user_id",
]
numeric_cols = ["item_seq_number", "image_top_1", "price"]

# Handle missing values and encode categorical features
train_df = train_df.copy()
test_df = test_df.copy()

# Fill missing values in numeric columns
for col in numeric_cols:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

# Create feature matrix
train_features = train_df[categorical_cols + numeric_cols].copy()
test_features = test_df[categorical_cols + numeric_cols].copy()

# Drop item_id from features (it's an identifier, not a feature)
train_features = train_features.drop(columns=["item_id"])
test_features = test_features.drop(columns=["item_id"])

# Create target variable
y_train = train_df["deal_probability"].values

# Split data into train/validation for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)


# Function to compute RMSE on validation set
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Perform 5-fold cross-validation
rmse_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train_fold = train_features.iloc[train_idx]
    X_val_fold = train_features.iloc[val_idx]
    y_train_fold = y_train[train_idx]

    # Train model
    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]
    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_train_fold)],
        callbacks=callbacks,
        verbose_eval=False,
    )

    # Predict on validation set
    y_pred_fold = model.predict(X_val_fold)
    rmse = compute_rmse(y_train_fold, y_pred_fold)
    rmse_scores.append(rmse)
    print(f"Fold {fold_idx + 1} RMSE: {rmse:.4f}")

# Calculate mean RMSE across folds
mean_rmse = np.mean(rmse_scores)
print(f"\nMean RMSE on 5-fold CV: {mean_rmse:.4f}")

# Train final model on all training data
model.fit(
    train_features,
    y_train,
    eval_set=[(train_features, y_train)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
    verbose_eval=False,
)

# Predict on test set
test_predictions = model.predict(test_features)

# Create submission file
submission = pd.DataFrame(
    {"id": test_df["item_id"].astype(str), "deal_probability": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
print(f"\nSubmission saved to ./working/submission.csv")
print(f"Final test predictions shape: {submission.shape}")
