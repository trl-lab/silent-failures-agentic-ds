import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/playground-series-s4e2/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/playground-series-s4e2/data/test.parquet"
TARGET = "NObeyesdad"
ID_COL = "id"
RANDOM_STATE = 42


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Height" in df.columns and "Weight" in df.columns:
        height = pd.to_numeric(df["Height"], errors="coerce")
        weight = pd.to_numeric(df["Weight"], errors="coerce")
        df["BMI"] = weight / (height**2)
        df["BMI"] = df["BMI"].replace([np.inf, -np.inf], np.nan)
    return df


def get_feature_columns(train_df: pd.DataFrame):
    feature_cols = [c for c in train_df.columns if c not in (ID_COL, TARGET)]
    cat_cols = [
        c
        for c in feature_cols
        if train_df[c].dtype == object or train_df[c].dtype.name == "category"
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, cat_cols, num_cols


def preprocess(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cat_cols, num_cols):
    X_train = train.drop(columns=[TARGET], errors="ignore").copy()
    X_val = val.drop(columns=[TARGET], errors="ignore").copy()
    X_test = test.copy()

    for X in (X_train, X_val, X_test):
        if ID_COL in X.columns:
            X = X.drop(columns=[ID_COL])

    for col in num_cols:
        if col not in X_train.columns:
            X_train[col] = np.nan
        if col not in X_val.columns:
            X_val[col] = np.nan
        if col not in X_test.columns:
            X_test[col] = np.nan

        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    num_fill = X_train[num_cols].median()
    for col in num_cols:
        fill_value = num_fill[col]
        if pd.isna(fill_value):
            fill_value = 0.0
        X_train[col] = X_train[col].fillna(fill_value)
        X_val[col] = X_val[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)

    for col in cat_cols:
        if col not in X_train.columns:
            X_train[col] = np.nan
        if col not in X_val.columns:
            X_val[col] = np.nan
        if col not in X_test.columns:
            X_test[col] = np.nan

        mode = X_train[col].mode()
        fill_value = mode.iloc[0] if not mode.empty else "missing"

        X_train[col] = X_train[col].fillna(fill_value).astype(str)
        X_val[col] = X_val[col].fillna(fill_value).astype(str)
        X_test[col] = X_test[col].fillna(fill_value).astype(str)

    train_dummies = pd.get_dummies(X_train[cat_cols], prefix=cat_cols, drop_first=False)
    val_dummies = pd.get_dummies(X_val[cat_cols], prefix=cat_cols, drop_first=False)
    test_dummies = pd.get_dummies(X_test[cat_cols], prefix=cat_cols, drop_first=False)

    val_dummies = val_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    X_train_out = pd.concat(
        [X_train[num_cols].reset_index(drop=True), train_dummies.reset_index(drop=True)],
        axis=1,
    )
    X_val_out = pd.concat(
        [X_val[num_cols].reset_index(drop=True), val_dummies.reset_index(drop=True)],
        axis=1,
    )
    X_test_out = pd.concat(
        [X_test[num_cols].reset_index(drop=True), test_dummies.reset_index(drop=True)],
        axis=1,
    )

    X_train_out = X_train_out.astype(float)
    X_val_out = X_val_out.astype(float)
    X_test_out = X_test_out.astype(float)

    return X_train_out, X_val_out, X_test_out


def make_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def align_proba(proba: np.ndarray, classes, n_classes: int) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
    for i, cls in enumerate(classes):
        aligned[:, int(cls)] = proba[:, i]
    return aligned


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])

    if len(test_df) == 0:
        pd.DataFrame({"id": np.arange(0, dtype=int), "NObeyesdad": []}).to_csv(
            "predictions.csv", index=False
        )
        return

    if train_df.empty:
        pd.DataFrame(
            {"id": np.arange(len(test_df), dtype=int), "NObeyesdad": []}
        ).to_csv("predictions.csv", index=False)
        return

    train_df = add_bmi(train_df)
    test_df = add_bmi(test_df)

    le = LabelEncoder().fit(train_df[TARGET])
    y_all = le.transform(train_df[TARGET])
    n_classes = len(le.classes_)

    if n_classes == 1:
        pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                "NObeyesdad": le.classes_[0],
            }
        ).to_csv("predictions.csv", index=False)
        return

    _, cat_cols, num_cols = get_feature_columns(train_df)

    stratify = y_all if pd.Series(y_all).value_counts().min() > 1 else None

    if len(train_df) < 10:
        X_train_full, _, X_test_full = preprocess(
            train_df, train_df, test_df, cat_cols, num_cols
        )
        model = make_models()["lgbm"]
        model.fit(X_train_full, y_all)
        proba = model.predict_proba(X_test_full)
        proba = align_proba(proba, model.classes_, n_classes)
        final_pred = le.classes_[np.argmax(proba, axis=1)]
        pd.DataFrame(
            {"id": np.arange(len(test_df), dtype=int), "NObeyesdad": final_pred}
        ).to_csv("predictions.csv", index=False)
        return

    train_raw, val_raw, y_train, y_val = train_test_split(
        train_df,
        y_all,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    X_train, X_val, X_test = preprocess(train_raw, val_raw, test_df, cat_cols, num_cols)

    models = make_models()
    val_probs = []
    val_scores = {}

    for name, model in models.items():
        if len(np.unique(y_train)) == 1:
            proba = np.zeros((len(X_val), n_classes), dtype=float)
            proba[:, int(y_train[0])] = 1.0
            val_probs.append(proba)
            val_scores[name] = accuracy_score(y_val, np.argmax(proba, axis=1))
            continue

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        proba = align_proba(proba, model.classes_, n_classes)
        val_probs.append(proba)
        val_scores[name] = accuracy_score(y_val, np.argmax(proba, axis=1))

    val_proba_avg = np.mean(val_probs, axis=0)
    val_scores["ensemble"] = accuracy_score(y_val, np.argmax(val_proba_avg, axis=1))

    best_name = max(val_scores, key=val_scores.get)

    X_train_full, _, X_test_full = preprocess(train_df, train_df, test_df, cat_cols, num_cols)
    y_full = le.transform(train_df[TARGET])

    if best_name == "ensemble":
        final_models = make_models()
        final_probs = []
        for model in final_models.values():
            model.fit(X_train_full, y_full)
            proba = model.predict_proba(X_test_full)
            final_probs.append(align_proba(proba, model.classes_, n_classes))
        final_proba = np.mean(final_probs, axis=0)
        final_pred = le.classes_[np.argmax(final_proba, axis=1)]
    else:
        final_model = make_models()[best_name]
        final_model.fit(X_train_full, y_full)
        proba = final_model.predict_proba(X_test_full)
        final_proba = align_proba(proba, final_model.classes_, n_classes)
        final_pred = le.classes_[np.argmax(final_proba, axis=1)]

    pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "NObeyesdad": final_pred,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()