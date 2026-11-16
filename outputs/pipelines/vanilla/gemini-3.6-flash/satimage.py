import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

train_path = (
    "/path/to/user/workspace/ds_failures/datasets/openml/satimage/data/train.parquet"
)
test_path = (
    "/path/to/user/workspace/ds_failures/datasets/openml/satimage/data/test.parquet"
)

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "class"
feature_cols = [c for c in train_df.columns if c not in [target_col, "id"]]

X_train = train_df[feature_cols].copy()
y_train_raw = train_df[target_col]
X_test = test_df[feature_cols].copy()

# Fill missing values if any
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

models = [
    LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    ),
    CatBoostClassifier(
        iterations=500, learning_rate=0.03, depth=6, verbose=0, random_seed=42
    ),
    XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss",
    ),
    ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
]

probs = np.zeros((len(X_test), len(le.classes_)))

for model in models:
  model.fit(X_train, y_train)
  probs += model.predict_proba(X_test) / len(models)

pred_indices = np.argmax(probs, axis=1)
preds = le.inverse_transform(pred_indices)

sub = pd.DataFrame({"id": np.arange(len(preds)), "class": preds})

sub.to_csv("predictions.csv", index=False)