import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

os.makedirs("./working", exist_ok=True)

# 2. Feature Engineering & Preprocessing
target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

# Ordinal mappings reflecting the domain structure
ordinal_mappings = {
    "buying": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
    "maint": {"low": 1, "med": 2, "high": 3, "vhigh": 4},
    "doors": {"2": 2, "3": 3, "4": 4, "5more": 5},
    "persons": {"2": 2, "4": 4, "more": 5},
    "lug_boot": {"small": 1, "med": 2, "big": 3},
    "safety": {"low": 1, "med": 2, "high": 3},
}


def preprocess_features(df):
    df_feat = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in ordinal_mappings:
            df_feat[col + "_ord"] = (
                df[col].astype(str).map(ordinal_mappings[col]).fillna(-1).astype(int)
            )
        df_feat[col] = df[col].astype("category")
    return df_feat


X = preprocess_features(train_df)
X_test = preprocess_features(test_df)

# Encode target
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))
n_classes = len(le.classes_)

# 3. Model Training & Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), n_classes))
test_preds = np.zeros((len(test_df), n_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.9,
    "seed": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    val_prob = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_prob
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

oof_pred_labels = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y, oof_pred_labels)
print(f"5-Fold CV Accuracy: {cv_accuracy:.6f}")

# 4. Generate & Save Test Predictions
final_test_labels = le.inverse_transform(np.argmax(test_preds, axis=1))

submission = pd.DataFrame({"id": np.arange(len(test_df)), "class": final_test_labels})

submission.to_csv("./working/submission.csv", index=False)
print("Saved submission to ./working/submission.csv")
