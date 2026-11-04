import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/default-of-credit-card-clients/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/default-of-credit-card-clients/data/test.parquet"
TARGET = "default payment next month"


class FeaturePreprocessor:
    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_fill = {}
        self.category_maps = {}

    def _is_categorical(self, series):
        dtype = series.dtype
        return dtype == object or dtype.name in {"category", "string"}

    def fit(self, X):
        X = X.copy()
        self.categorical_cols = [c for c in X.columns if self._is_categorical(X[c])]
        self.numeric_cols = [c for c in X.columns if c not in self.categorical_cols]

        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            if X[col].isna().any():
                fill = X[col].median()
                if pd.isna(fill):
                    fill = 0.0
                self.numeric_fill[col] = float(fill)

        for col in self.categorical_cols:
            X[col] = X[col].astype(str)
            X[col] = X[col].replace({"nan": "missing", "None": "missing", "": "missing"})
            categories = sorted(X[col].unique())
            self.category_maps[col] = {cat: i for i, cat in enumerate(categories)}

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.numeric_cols:
            if col not in X.columns:
                X[col] = np.nan
            X[col] = pd.to_numeric(X[col], errors="coerce")
            if col in self.numeric_fill:
                X[col] = X[col].fillna(self.numeric_fill[col])
            elif X[col].isna().any():
                X[col] = X[col].fillna(0.0)

        for col in self.categorical_cols:
            if col not in X.columns:
                X[col] = "missing"
            X[col] = X[col].astype(str)
            X[col] = X[col].replace({"nan": "missing", "None": "missing", "": "missing"})
            mapping = self.category_maps.get(col, {})
            X[col] = X[col].map(mapping).fillna(-1).astype(int)

        expected_cols = self.numeric_cols + self.categorical_cols
        X = X[expected_cols]

        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(0.0)

        return X


def make_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
        ),
        "catboost": CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    id_like_cols = [c for c in train_df.columns if str(c).lower() in {"id", "passengerid", "customerid"}]
    train_df = train_df.drop(columns=id_like_cols, errors="ignore")
    test_df = test_df.drop(columns=id_like_cols, errors="ignore")

    train_df = train_df.dropna(subset=[TARGET])
    y = train_df[TARGET].astype(int)
    X = train_df.drop(columns=[TARGET])

    X_test = test_df.drop(columns=[TARGET], errors="ignore")
    X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)

    if len(np.unique(y)) > 1:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    preprocessor = FeaturePreprocessor().fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    models = make_models()
    val_scores = {}
    val_preds = {}

    for name, model in models.items():
        try:
            model.fit(X_train_p, y_train)
            proba = model.predict_proba(X_val_p)[:, 1]
            val_preds[name] = proba
            val_scores[name] = safe_auc(y_val, proba)
        except Exception:
            continue

    if val_preds:
        ensemble_proba = np.mean(list(val_preds.values()), axis=0)
        val_scores["ensemble"] = safe_auc(y_val, ensemble_proba)
        best_name = max(val_scores, key=val_scores.get)
    else:
        best_name = "lgbm"

    final_preprocessor = FeaturePreprocessor().fit(X)
    X_full_p = final_preprocessor.transform(X)
    X_test_final_p = final_preprocessor.transform(X_test)

    if best_name == "ensemble":
        final_models = make_models()
        final_preds = []
        for model in final_models.values():
            try:
                model.fit(X_full_p, y)
                final_preds.append(model.predict_proba(X_test_final_p)[:, 1])
            except Exception:
                continue

        if final_preds:
            final_proba = np.mean(final_preds, axis=0)
        else:
            fallback = LGBMClassifier(n_estimators=500, random_state=42, verbose=-1)
            fallback.fit(X_full_p, y)
            final_proba = fallback.predict_proba(X_test_final_p)[:, 1]
    else:
        model = make_models()[best_name]
        model.fit(X_full_p, y)
        final_proba = model.predict_proba(X_test_final_p)[:, 1]

    final_proba = np.clip(final_proba, 0.0, 1.0)

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            TARGET: final_proba,
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()