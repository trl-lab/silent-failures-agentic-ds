import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "outcomes"
features = [c for c in train_df.columns if c != target_col]

# 2. Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(train_df[target_col].astype(str))
n_classes = len(le_target.classes_)

# Combine to preprocess consistently
combined_df = pd.concat([train_df[features], test_df[features]], axis=0).reset_index(
    drop=True
)

# Feature engineering / type handling
cat_cols = []
for col in features:
    if combined_df[col].dtype == "object" or combined_df[col].dtype.name == "category":
        cat_cols.append(col)
        combined_df[col] = combined_df[col].astype(str).fillna("__MISSING__")
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
        combined_df[col] = combined_df[col].astype("category")
    elif pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        combined_df[f"{col}_year"] = combined_df[col].dt.year
        combined_df[f"{col}_month"] = combined_df[col].dt.month
        combined_df[f"{col}_day"] = combined_df[col].dt.day
        combined_df[f"{col}_dayofweek"] = combined_df[col].dt.dayofweek
        combined_df.drop(columns=[col], inplace=True)
    else:
        # Numeric columns: fill potential infs
        combined_df[col] = combined_df[col].replace([np.inf, -np.inf], np.nan)

final_features = [c for c in combined_df.columns]
cat_features = [c for c in final_features if str(combined_df[c].dtype) == "category"]

X = combined_df.iloc[: len(train_df)].copy()
X_test = combined_df.iloc[len(train_df) :].copy().reset_index(drop=True)

# 3. Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X), n_classes))
test_preds = np.zeros((len(X_test), n_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_data = lgb.Dataset(
        X_val, label=y_val, reference=train_data, categorical_feature=cat_features
    )

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )

oof_pred_labels = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y, oof_pred_labels)
print(f"OOF Accuracy Score: {cv_accuracy:.5f}")

# 4. Generate Submission
final_test_labels = np.argmax(test_preds, axis=1)
predicted_outcomes = le_target.inverse_transform(final_test_labels)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "outcomes": predicted_outcomes}
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully. Shape:", submission.shape)
