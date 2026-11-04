import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/estimation-of-obesity-levels/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/estimation-of-obesity-levels/data/test.parquet"
TARGET = "NObeyesdad"
RANDOM_STATE = 42


class Preprocessor:
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)

    def _numeric_series(self, X, col):
        if col in X.columns:
            series = pd.to_numeric(X[col], errors="coerce")
        else:
            series = pd.Series(np.nan, index=X.index)
        return series.replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y=None):
        X = X.copy()

        self.numeric_medians_ = {}
        for col in self.numeric_cols:
            series = self._numeric_series(X, col)
            if series.isna().all():
                self.numeric_medians_[col] = 0.0
            else:
                median = series.median()
                self.numeric_medians_[col] = 0.0 if pd.isna(median) else float(median)

        self.categorical_categories_ = {}
        for col in self.categorical_cols:
            if col not in X.columns:
                categories = ["missing"]
            else:
                series = X[col].astype("string").fillna("missing").astype(str)
                categories = sorted(series.unique().tolist())
                if "missing" not in categories:
                    categories.append("missing")
            self.categorical_categories_[col] = categories

        return self

    def transform(self, X):
        X = X.copy()
        transformed = {}

        for col in self.numeric_cols:
            series = self._numeric_series(X, col)
            transformed[col] = series.fillna(self.numeric_medians_.get(col, 0.0)).astype(float)

        for col in self.categorical_cols:
            if col in X.columns:
                series = X[col].astype("string").fillna("missing").astype(str)
            else:
                series = pd.Series("missing", index=X.index)

            categories = self.categorical_categories_.get(col, ["missing"])
            for category in categories:
                transformed[f"{col}__{category}"] = (series == category).astype(int)

        return pd.DataFrame(transformed, index=X.index)


def add_features(df):
    df = df.copy()
    if "Height" in df.columns and "Weight" in df.columns:
        height = pd.to_numeric(df["Height"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        weight = pd.to_numeric(df["Weight"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        df["BMI"] = np.where(height > 0, weight / (height ** 2), np.nan)
    return df


def make_model(name, n_classes):
    if name == "lgbm":
        objective = "binary" if n_classes == 2 else "multiclass"
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            objective=objective,
        )

    if name == "xgb":
        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
            extra = {}
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"
            extra = {"num_class": n_classes}

        return XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            objective=objective,
            eval_metric=eval_metric,
            **extra,
        )

    if name == "catboost":
        loss_function = "Logloss" if n_classes == 2 else "MultiClass"
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            thread_count=-1,
            verbose=0,
            loss_function=loss_function,
        )

    raise ValueError(f"Unknown model: {name}")


def aligned_proba(model, X, n_classes):
    proba = model.predict_proba(X)
    model_classes = getattr(model, "classes_", np.arange(n_classes))
    out = np.zeros((X.shape[0], n_classes), dtype=float)

    for i, cls in enumerate(model_classes):
        idx = int(cls)
        if 0 <= idx < n_classes:
            out[:, idx] = proba[:, i]

    return out


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])
    train_df[TARGET] = train_df[TARGET].astype(str).str.strip()

    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET])

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    X_all = train_df.drop(columns=[TARGET])
    y_all = train_df[TARGET]

    drop_cols = [c for c in ["id", "ID", "Id"] if c in X_all.columns]
    if drop_cols:
        X_all = X_all.drop(columns=drop_cols)
        test_df = test_df.drop(columns=drop_cols, errors="ignore")

    test_df = test_df.reindex(columns=X_all.columns)

    numeric_cols = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]
    categorical_cols = [c for c in X_all.columns if c not in numeric_cols]

    class_names = np.sort(y_all.unique())
    class_to_int = {name: i for i, name in enumerate(class_names)}
    y_all_int = y_all.map(class_to_int).astype(int)

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all_int,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_all_int,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all_int,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    prep_val = Preprocessor(numeric_cols, categorical_cols).fit(X_train)
    X_train_t = prep_val.transform(X_train)
    X_val_t = prep_val.transform(X_val)

    n_classes = len(class_names)
    model_names = ["lgbm", "xgb", "catboost"]

    val_scores = {}
    val_probas = []

    for name in model_names:
        model = make_model(name, n_classes)
        model.fit(X_train_t, y_train)
        proba = aligned_proba(model, X_val_t, n_classes)
        pred = np.argmax(proba, axis=1)
        val_scores[name] = accuracy_score(y_val, pred)
        val_probas.append(proba)

    ensemble_proba = np.mean(val_probas, axis=0)
    val_scores["ensemble"] = accuracy_score(y_val, np.argmax(ensemble_proba, axis=1))

    best_name = max(val_scores, key=val_scores.get)

    prep_final = Preprocessor(numeric_cols, categorical_cols).fit(X_all)
    X_all_t = prep_final.transform(X_all)
    X_test_t = prep_final.transform(test_df)

    if best_name == "ensemble":
        test_probas = []
        for name in model_names:
            model = make_model(name, n_classes)
            model.fit(X_all_t, y_all_int)
            test_probas.append(aligned_proba(model, X_test_t, n_classes))
        test_proba = np.mean(test_probas, axis=0)
    else:
        model = make_model(best_name, n_classes)
        model.fit(X_all_t, y_all_int)
        test_proba = aligned_proba(model, X_test_t, n_classes)

    predictions = class_names[np.argmax(test_proba, axis=1)]

    pd.DataFrame(
        {
            "id": np.arange(len(predictions)),
            "NObeyesdad": predictions,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()