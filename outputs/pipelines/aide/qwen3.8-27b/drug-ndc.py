import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score

TARGET = "dea_schedule"
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

train = train.dropna(subset=[TARGET]).reset_index(drop=True)

if len(train) == 0:
    submission = pd.DataFrame(
        {"id": np.arange(len(test)), "dea_schedule": ["unknown"] * len(test)}
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    print("CV Accuracy: 0.00000")
    raise SystemExit

y_raw = train[TARGET].copy()
train = train.drop(columns=[TARGET])

feature_cols = [c for c in train.columns if c in test.columns]

id_like = {
    "id",
    "index",
    "row_id",
    "rowid",
    "uid",
    "uuid",
    "sample_id",
    "drug_id",
    "ndc_id",
}
feature_cols = [c for c in feature_cols if c.lower() not in id_like]

numeric_cols = []
categorical_cols = []
for col in feature_cols:
    s = train[col]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        numeric_cols.append(col)
    else:
        converted = pd.to_numeric(s, errors="coerce")
        non_null = int(s.notna().sum())
        if non_null > 0 and int(converted.notna().sum()) >= 0.8 * non_null:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)


def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "missing").astype(str)


class Preprocessor:
    def __init__(self, feature_cols, numeric_cols, categorical_cols):
        self.feature_cols = feature_cols
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.cat_mappings = {}

    def fit(self, X: pd.DataFrame):
        for col in self.categorical_cols:
            vals = safe_str_series(X[col])
            cats = sorted(vals.unique())
            self.cat_mappings[col] = {v: i for i, v in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_cols].copy()

        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        for col in self.categorical_cols:
            mapping = self.cat_mappings[col]
            X[col] = safe_str_series(X[col]).map(mapping).fillna(-1).astype(np.int32)

        return X


try:
    classes = np.sort(y_raw.unique(), key=lambda x: str(x))
except Exception:
    classes = y_raw.unique()

class_to_int = {c: i for i, c in enumerate(classes)}
y_int = y_raw.map(class_to_int).astype(int).to_numpy()
n_classes = len(classes)

if len(feature_cols) == 0 or n_classes == 1:
    if n_classes > 1:
        majority_idx = int(np.argmax(np.bincount(y_int)))
        majority_class = classes[majority_idx]
        cv_acc = accuracy_score(y_int, np.full(len(y_int), majority_idx, dtype=int))
    else:
        majority_class = classes[0]
        cv_acc = 1.0

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "dea_schedule": np.array([majority_class] * len(test)),
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
    print(f"CV Accuracy: {cv_acc:.5f}")
    raise SystemExit

params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_error",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbose": -1,
}

X_train_raw = train[feature_cols]

min_class_count = int(np.min(np.bincount(y_int)))
n_splits = min(5, max(2, min_class_count))

if min_class_count >= n_splits:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
else:
    skf = KFold(n_splits=2, shuffle=True, random_state=42)

cv_accs = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw, y_int), start=1):
    X_tr_raw = X_train_raw.iloc[tr_idx]
    X_va_raw = X_train_raw.iloc[va_idx]

    pre = Preprocessor(feature_cols, numeric_cols, categorical_cols).fit(X_tr_raw)
    X_tr = pre.transform(X_tr_raw)
    X_va = pre.transform(X_va_raw)

    dtrain = lgb.Dataset(
        X_tr,
        label=y_int[tr_idx],
        categorical_feature=categorical_cols,
    )
    dvalid = lgb.Dataset(
        X_va,
        label=y_int[va_idx],
        categorical_feature=categorical_cols,
        reference=dtrain,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
    )

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None or best_iter <= 0:
        best_iter = 1
    else:
        best_iter = int(best_iter)

    best_iters.append(best_iter)

    pred_proba = model.predict(X_va, num_iteration=best_iter)
    pred = np.argmax(pred_proba, axis=1)
    acc = accuracy_score(y_int[va_idx], pred)
    cv_accs.append(acc)

    print(f"Fold {fold}: accuracy={acc:.5f}, best_iteration={best_iter}")

cv_acc = float(np.mean(cv_accs))
print(f"CV Accuracy: {cv_acc:.5f}")

final_pre = Preprocessor(feature_cols, numeric_cols, categorical_cols).fit(X_train_raw)
X_train_final = final_pre.transform(X_train_raw)
X_test_final = final_pre.transform(test[feature_cols])

final_rounds = int(max(1, np.mean(best_iters))) if best_iters else 100

dtrain_final = lgb.Dataset(
    X_train_final,
    label=y_int,
    categorical_feature=categorical_cols,
)

final_model = lgb.train(
    params,
    dtrain_final,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_proba = final_model.predict(X_test_final)
test_pred_idx = np.argmax(test_proba, axis=1)
test_pred = classes[test_pred_idx]

submission = pd.DataFrame(
    {
        "id": np.arange(len(test)),
        "dea_schedule": test_pred,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
