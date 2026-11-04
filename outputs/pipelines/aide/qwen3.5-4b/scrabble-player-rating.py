import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import os

# Read data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Feature engineering
train_features = train_df[["game_id", "nickname", "score"]]
test_features = test_df[["game_id", "nickname", "score"]]

# One-hot encode nickname
train_encoded = pd.get_dummies(train_features, columns=["nickname"], prefix="nickname")
test_encoded = pd.get_dummies(test_features, columns=["nickname"], prefix="nickname")

# Combine features
X_train = train_encoded.drop(columns=["nickname"])
y_train = train_df["rating"]
X_test = test_encoded

# Create model
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    # Train model with early stopping
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    callbacks = [
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(0, verbose=False),
    ]

    model.fit(
        train_data, valid_sets=[val_data], callbacks=callbacks, verbose_eval=False
    )

    # Predict on validation fold
    val_pred = model.predict(X_val_fold)
    mae = mean_absolute_error(y_val_fold, val_pred)
    mae_scores.append(mae)
    fold_scores.append(mae)

    print(f"Fold {fold+1}: MAE = {mae:.4f}")

# Final cross-validation MAE
cv_mae = np.mean(mae_scores)
print(f"\n5-Fold Cross-Validation MAE: {cv_mae:.4f}")

# Train final model on all training data
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(
    X_train.iloc[:10000], label=y_train.iloc[:10000], reference=train_data
)

callbacks = [
    lgb.early_stopping(100, verbose=False),
    lgb.log_evaluation(0, verbose=False),
]

model.fit(train_data, valid_sets=[val_data], callbacks=callbacks, verbose_eval=False)

# Train on full data with early stopping
model.fit(train_data, valid_sets=[val_data], callbacks=callbacks, verbose_eval=False)

# Predict on test set
test_predictions = model.predict(X_test)

# Save submission
submission = pd.DataFrame({"id": range(len(test_df)), "rating": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
print("\nSubmission saved to ./working/submission.csv")

# Final MAE on validation set
final_val_mae = mean_absolute_error(
    y_train.iloc[:10000], model.predict(X_train.iloc[:10000])
)
print(f"\nFinal Validation MAE: {final_val_mae:.4f}")
