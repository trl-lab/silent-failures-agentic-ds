import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
feature_cols = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
target_col = "quality"
id_col = "id"

# Check if id column exists in test data
if id_col not in test_df.columns:
    test_df.insert(0, id_col, range(len(test_df)))

# Create feature matrix and target
X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation for evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    y_pred_fold = model.predict(X_val_fold)
    cv_scores.append(mean_squared_error(y_val_fold, y_pred_fold))

# Calculate RMSE from cross-validation
rmse_cv = np.sqrt(np.mean(cv_scores))
print(f"5-fold Cross-Validation RMSE: {rmse_cv:.4f}")

# Train final model on full training data
model.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_train_scaled, y_train)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
)

# Generate predictions on test set
y_test_pred = model.predict(X_test_scaled)

# Create submission file
submission = pd.DataFrame({id_col: range(len(test_df)), target_col: y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print(f"Saved predictions to ./working/submission.csv")
print(f"Final RMSE on validation set: {rmse_cv:.4f}")
