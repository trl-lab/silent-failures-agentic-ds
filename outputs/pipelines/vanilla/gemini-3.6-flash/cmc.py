import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/cmc/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/cmc/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Contraceptive_method_used"

X_train = train_df.drop(columns=[target_col]).copy()
y_raw = train_df[target_col].copy()
X_test = test_df.copy()

if "id" in X_train.columns:
    X_train = X_train.drop(columns=["id"])
if "id" in X_test.columns:
    X_test = X_test.drop(columns=["id"])

le = LabelEncoder()
y_train = le.fit_transform(y_raw)
num_classes = len(le.classes_)


def add_features(df):
    df = df.copy()

    col_age = None
    col_w_ed = None
    col_h_ed = None
    col_ch = None
    col_sol = None

    for col in df.columns:
        clow = str(col).lower()
        if "age" in clow:
            col_age = col
        elif "wife" in clow and "ed" in clow:
            col_w_ed = col
        elif "husband" in clow and "ed" in clow:
            col_h_ed = col
        elif "children" in clow or "born" in clow:
            col_ch = col
        elif "living" in clow or "sol" in clow or "standard" in clow:
            col_sol = col

    if col_age is not None and col_ch is not None:
        df["ch_per_age"] = df[col_ch] / (df[col_age] + 1e-5)
        df["age_minus_ch"] = df[col_age] - df[col_ch] * 2.0
        df["ch_per_fertile_years"] = df[col_ch] / np.maximum(1, df[col_age] - 15)

    if col_w_ed is not None and col_h_ed is not None:
        df["ed_diff"] = df[col_w_ed] - df[col_h_ed]
        df["ed_sum"] = df[col_w_ed] + df[col_h_ed]
        df["w_ed_higher"] = (df[col_w_ed] > df[col_h_ed]).astype(int)
        df["ed_equal"] = (df[col_w_ed] == df[col_h_ed]).astype(int)

    if col_w_ed is not None and col_sol is not None:
        df["w_ed_sol"] = df[col_w_ed] * 10 + df[col_sol]

    return df


combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)

for col in combined.columns:
    if combined[col].dtype == "object" or str(combined[col].dtype) == "category":
        combined[col] = combined[col].astype("category").cat.codes

combined_fe = add_features(combined)

X_train_fe = combined_fe.iloc[: len(X_train)].copy()
X_test_fe = combined_fe.iloc[len(X_train) :].copy()

X_train_fe = X_train_fe.fillna(-999)
X_test_fe = X_test_fe.fillna(-999)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_probs = np.zeros((len(X_test_fe), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_fe, y_train)):
    X_tr, y_tr = X_train_fe.iloc[train_idx], y_train[train_idx]

    m1 = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=5,
        verbose=0,
        random_seed=42 + fold,
    )
    m2 = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=15,
        random_state=42 + fold,
        verbose=-1,
    )
    m3 = XGBClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=4,
        random_state=42 + fold,
        eval_metric="mlogloss",
    )
    m4 = ExtraTreesClassifier(
        n_estimators=200, max_depth=10, min_samples_split=4, random_state=42 + fold
    )
    m5 = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=4, random_state=42 + fold
    )
    m6 = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.03,
        max_depth=5,
        max_leaf_nodes=15,
        random_state=42 + fold,
    )

    models = [m1, m2, m3, m4, m5, m6]

    fold_test_probs = np.zeros((len(X_test_fe), num_classes))

    for m in models:
        m.fit(X_tr, y_tr)
        fold_test_probs += m.predict_proba(X_test_fe) / len(models)

    test_preds_probs += fold_test_probs / n_splits

final_class_indices = np.argmax(test_preds_probs, axis=1)
final_predictions = le.inverse_transform(final_class_indices)

predictions_df = pd.DataFrame(
    {"id": np.arange(len(final_predictions)), target_col: final_predictions}
)

predictions_df.to_csv("predictions.csv", index=False)