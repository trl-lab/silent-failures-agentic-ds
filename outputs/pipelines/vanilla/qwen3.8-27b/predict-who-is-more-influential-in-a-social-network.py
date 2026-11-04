import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/predict-who-is-more-influential-in-a-social-network/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/predict-who-is-more-influential-in-a-social-network/data/test.parquet"
TARGET = "Choice"
RANDOM_STATE = 42


def add_pair_features(df):
    """Add safe A-vs-B features without using the target."""
    if df is None or df.empty:
        return df

    a_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("A_")]
    for a_col in a_cols:
        base = a_col[2:]
        b_col = f"B_{base}"
        if b_col not in df.columns:
            continue

        a = df[a_col]
        b = df[b_col]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            continue

        df[f"diff_{base}"] = a - b
        df[f"ratio_{base}"] = a / (b + 1e-6)

        a_nonneg = a.clip(lower=0)
        b_nonneg = b.clip(lower=0)
        df[f"log_ratio_{base}"] = np.log1p(a_nonneg) - np.log1p(b_nonneg)

    return df


def align_columns(train_df, test_df):
    """Make test columns match train columns exactly."""
    cols = train_df.columns.tolist()
    for col in cols:
        if col not in test_df.columns:
            test_df[col] = np.nan
    test_df = test_df[cols].copy()
    return train_df, test_df


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=50)


def build_preprocessor(X_train):
    """Build a leakage-safe preprocessor from the training split only."""
    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    transformers = []

    if numeric_cols:
        for col in numeric_cols:
            if X_train[col].isna().all():
                X_train[col] = 0.0
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))

    if categorical_cols:
        for col in categorical_cols:
            if X_train[col].isna().all():
                X_train[col] = "missing"

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", make_onehot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_cols))

    if not transformers:
        return Pipeline([("passthrough", "passthrough")])

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=600,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=600,
            learning_rate=0.04,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    y_raw = train_df[TARGET].copy()
    train_df = train_df.drop(columns=[TARGET])

    train_df = add_pair_features(train_df)
    test_df = add_pair_features(test_df)
    train_df, test_df = align_columns(train_df, test_df)

    # Encode target if needed, while remembering original labels for output.
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int)
        inverse_map = None
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        inverse_map = le.classes_

    if len(train_df) == 0:
        pred = np.zeros(len(test_df), dtype=int)
        if inverse_map is not None:
            pred = inverse_map[pred]
        pd.DataFrame({"id": np.arange(len(test_df)), TARGET: pred}).to_csv(
            "predictions.csv", index=False
        )
        return

    if y.nunique() == 1:
        pred = np.full(len(test_df), int(y.iloc[0]), dtype=int)
        if inverse_map is not None:
            pred = inverse_map[pred]
        pd.DataFrame({"id": np.arange(len(test_df)), TARGET: pred}).to_csv(
            "predictions.csv", index=False
        )
        return

    stratify = y if y.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        train_df,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    models = make_models()
    val_scores = {}
    val_probas = {}

    for name, model in models.items():
        try:
            model.fit(X_train_t, y_train)
            proba = model.predict_proba(X_val_t)

            if y_train.nunique() == 2:
                pos_idx = 1 if model.classes_[-1] == max(y_train.unique()) else 0
                val_probas[name] = proba[:, pos_idx]
            else:
                val_probas[name] = proba

            pred = model.predict(X_val_t)
            val_scores[name] = accuracy_score(y_val, pred)
        except Exception:
            val_scores[name] = 0.0
            val_probas[name] = None

    if not any(score > 0 for score in val_scores.values()):
        pred = np.full(len(test_df), int(y_train.mode().iloc[0]), dtype=int)
        if inverse_map is not None:
            pred = inverse_map[pred]
        pd.DataFrame({"id": np.arange(len(test_df)), TARGET: pred}).to_csv(
            "predictions.csv", index=False
        )
        return

    scores = np.array([val_scores[name] for name in models.keys()], dtype=float)
    weights = scores - scores.min() + 1e-6
    weights = weights ** 2
    weights /= weights.sum()

    # Refit on the full training set after validation-based weighting.
    full_preprocessor = build_preprocessor(train_df)
    X_full_t = full_preprocessor.fit_transform(train_df)
    X_test_t = full_preprocessor.transform(test_df)

    final_models = make_models()
    test_probas = []
    total_weight = 0.0

    for (name, _), weight in zip(models.items(), weights):
        if weight <= 0 or val_probas[name] is None:
            continue
        try:
            final_models[name].fit(X_full_t, y)
            proba = final_models[name].predict_proba(X_test_t)

            if y.nunique() == 2:
                pos_idx = 1 if final_models[name].classes_[-1] == max(y.unique()) else 0
                proba = proba[:, pos_idx]

            test_probas.append(weight * proba)
            total_weight += weight
        except Exception:
            continue

    if total_weight <= 0:
        pred = np.full(len(test_df), int(y.mode().iloc[0]), dtype=int)
        if inverse_map is not None:
            pred = inverse_map[pred]
    else:
        test_proba = sum(test_probas) / total_weight

        if y.nunique() == 2:
            # Choose a threshold on the validation split when possible.
            threshold = 0.5
            valid_val_probas = [
                val_probas[name]
                for name in models.keys()
                if val_probas[name] is not None and val_scores[name] > 0
            ]
            if valid_val_probas:
                val_proba = sum(
                    weight * val_probas[name]
                    for (name, _), weight in zip(models.items(), weights)
                    if val_probas[name] is not None and val_scores[name] > 0
                ) / sum(
                    weight
                    for (name, _), weight in zip(models.items(), weights)
                    if val_probas[name] is not None and val_scores[name] > 0
                )

                if len(y_val) > 0 and y_val.nunique() > 1:
                    candidate_thresholds = np.unique(
                        np.concatenate(
                            [
                                np.linspace(0.30, 0.70, 41),
                                np.quantile(val_proba, np.linspace(0.0, 1.0, 21)),
                            ]
                        )
                    )
                    threshold = max(
                        candidate_thresholds,
                        key=lambda t: accuracy_score(
                            y_val, (val_proba >= t).astype(int)
                        ),
                    )

            pred = (test_proba >= threshold).astype(int)
        else:
            pred = np.argmax(test_proba, axis=1)

        if inverse_map is not None:
            pred = inverse_map[pred]

    output = pd.DataFrame({"id": np.arange(len(test_df)), TARGET: pred})
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()