import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Extract features and target
train_features = train_df.drop(columns=["outcomes"])
train_target = train_df["outcomes"]
test_features = test_df

# Define LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": len(train_target.unique()),
    "verbose": -1,
    "force_col_wise": True,
    "early_stopping_rounds": 50,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    X_train_fold = train_features.iloc[train_idx]
    y_train_fold = train_target.iloc[train_idx]
    X_val_fold = train_features.iloc[val_idx]
    y_val_fold = train_target.iloc[val_idx]

    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        num_boost_round=1000,
    )

    y_val_pred = model.predict(X_val_fold)
    y_val_pred = [int(y) for y in y_val_pred]
    cv_scores.append(accuracy_score(y_val_fold, y_val_pred))

print(
    f"5-fold CV Accuracy: {accuracy_score(train_target.iloc[val_idx], y_val_pred):.4f}"
)
print(f"Mean CV Accuracy: {sum(cv_scores)/len(cv_scores):.4f}")

# Train final model on all training data
train_data_final = lgb.Dataset(train_features, label=train_target)
final_model = lgb.train(params, train_data_final, num_boost_round=1000, verbose_eval=-1)

# Generate predictions for test set
test_features = test_df
test_predictions = final_model.predict(test_features)
test_predictions = [int(y) for y in test_predictions]

# Create submission file
submission = pd.DataFrame({"id": range(len(test_df)), "outcomes": test_predictions})
submission.to_csv("./working/submission.csv", index=False)

print("Submission saved to ./working/submission.csv")
