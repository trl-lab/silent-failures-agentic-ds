import warnings
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/climate-model-simulation-crashes/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/climate-model-simulation-crashes/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "outcome"
feature_cols = [c for c in train_df.columns if c != target_col and c != "id"]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()
X_test = test_df[feature_cols].copy()

for c in X.columns:
    if X[c].isnull().any() or X_test[c].isnull().any():
        med = X[c].median()
        X[c] = X[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

le = LabelEncoder()
y = le.fit_transform(y)


def get_models(seed=42):
    return {
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=5, random_state=seed, max_features="sqrt"
        ),
        "et": ExtraTreesClassifier(
            n_estimators=300, max_depth=5, random_state=seed, max_features="sqrt"
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=3, random_state=seed
        ),
        "lgb": LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            random_state=seed,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            eval_metric="logloss",
            random_state=seed,
        ),
        "cat": CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=4, verbose=0, random_seed=seed
        ),
        "lr": LogisticRegression(C=0.5, max_iter=1000, random_state=seed),
        "svc": SVC(probability=True, C=1.0, kernel="rbf", random_state=seed),
    }


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

test_preds_list = []

for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    X_tr_sc = X_scaled.iloc[train_idx]
    X_val_sc = X_scaled.iloc[val_idx]

    models = get_models(seed=42 + fold)

    fold_test_preds = []

    for name, model in models.items():
        if name in ["lr", "svc"]:
            model.fit(X_tr_sc, y_tr)
            p_test = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_tr, y_tr)
            p_test = model.predict_proba(X_test)[:, 1]

        p_test_ranked = np.argsort(np.argsort(p_test)) / len(p_test)
        fold_test_preds.append(p_test_ranked)

    test_preds_list.append(np.mean(fold_test_preds, axis=0))

final_test_preds = np.mean(test_preds_list, axis=0)

submission = pd.DataFrame({"id": np.arange(len(test_df)), "outcome": final_test_preds})

submission.to_csv("predictions.csv", index=False)