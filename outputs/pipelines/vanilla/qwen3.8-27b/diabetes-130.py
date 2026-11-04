import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/diabetes-130/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/diabetes-130/data/test.parquet"
TARGET = "readmitted"
ID_COLS = {"encounter_id", "patient_nbr"}
MISSING = "__MISSING__"
MISSING_STRINGS = ["", "none", "None", "NULL", "null", "NaN", "nan", "NA", "N/A"]
RANDOM_STATE = 42


def normalize_label(value):
    s = str(value).strip()
    if s.lower() == "no":
        return "NO"
    if s in {"<30", ">30"}:
        return s
    return s


def basic_clean(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train data.")

    train_df[TARGET] = train_df[TARGET].map(normalize_label)
    train_df = train_df.dropna(subset=[TARGET])

    drop_cols = [c for c in ID_COLS if c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=[TARGET], errors="ignore")

    feature_cols = [c for c in train_df.columns if c != TARGET]
    for col in feature_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan
    test_df = test_df[feature_cols]

    return train_df, test_df


class Preprocessor:
    def __init__(self):
        self.feature_cols = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_medians = {}
        self.categorical_categories = {}

    @staticmethod
    def _clean_categorical(series):
        s = series.astype("object")
        s = s.where(s.notna(), MISSING)
        s = s.astype(str).str.strip()
        s = s.replace(MISSING_STRINGS, MISSING)
        return s

    @staticmethod
    def _clean_numeric(series):
        s = pd.to_numeric(series, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        return s

    def fit(self, train_df):
        self.feature_cols = [c for c in train_df.columns if c != TARGET]

        self.numeric_cols = []
        self.categorical_cols = []
        for col in self.feature_cols:
            if pd.api.types.is_numeric_dtype(train_df[col]) and not col.lower().endswith("_id"):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

        self.numeric_medians = {}
        for col in self.numeric_cols:
            s = self._clean_numeric(train_df[col])
            median = s.median()
            if pd.isna(median):
                median = 0.0
            self.numeric_medians[col] = float(median)

        self.categorical_categories = {}
        for col in self.categorical_cols:
            s = self._clean_categorical(train_df[col])
            cats = list(s.unique())
            if MISSING not in cats:
                cats.append(MISSING)
            self.categorical_categories[col] = cats

        constant_cols = []
        for col in self.feature_cols:
            if col in self.numeric_cols:
                s = self._clean_numeric(train_df[col]).fillna(self.numeric_medians[col])
            else:
                s = self._clean_categorical(train_df[col])
            if s.nunique() <= 1:
                constant_cols.append(col)

        self.feature_cols = [c for c in self.feature_cols if c not in constant_cols]
        self.numeric_cols = [c for c in self.numeric_cols if c not in constant_cols]
        self.categorical_cols = [c for c in self.categorical_cols if c not in constant_cols]

        return self

    def transform(self, df):
        out = df[self.feature_cols].copy()

        for col in self.numeric_cols:
            out[col] = self._clean_numeric(out[col]).fillna(self.numeric_medians[col])

        for col in self.categorical_cols:
            s = self._clean_categorical(out[col])
            cats = self.categorical_categories[col]
            s = s.where(s.isin(cats), MISSING)
            out[col] = s.astype("category")

        return out

    def fit_transform(self, train_df):
        return self.fit(train_df).transform(train_df)


def make_models(categorical_cols):
    models = {}

    models["lgbm"] = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        categorical_feature=categorical_cols,
    )

    try:
        models["catboost"] = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            cat_features=categorical_cols,
            allow_writing_files=False,
        )
    except Exception:
        pass

    return models


def safe_predict(model, X, labels):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        model_labels = list(model.classes_)
        proba_df = pd.DataFrame(proba, columns=model_labels, index=X.index)
        proba_df = proba_df.reindex(columns=list(labels), fill_value=0.0)
        idx = proba_df.idxmax(axis=1)
        return idx.to_numpy()
    return model.predict(X)


def main():
    train_raw, test_raw = basic_clean(
        pd.read_parquet(TRAIN_PATH),
        pd.read_parquet(TEST_PATH),
    )

    X_raw = train_raw.drop(columns=[TARGET])
    y_raw = train_raw[TARGET]

    try:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_raw,
            y_raw,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_raw,
        )
    except ValueError:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_raw,
            y_raw,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    val_preprocessor = Preprocessor().fit(X_train_raw)
    X_train = val_preprocessor.transform(X_train_raw)
    X_val = val_preprocessor.transform(X_val_raw)

    labels = list(pd.Series(y_raw).unique())
    val_scores = {}

    if X_train.shape[1] == 0:
        best_name = "majority"
    else:
        val_models = make_models(val_preprocessor.categorical_cols)
        for name, model in val_models.items():
            try:
                model.fit(X_train, y_train)
                pred_val = safe_predict(model, X_val, labels)
                val_scores[name] = accuracy_score(y_val, pred_val)
            except Exception:
                continue

        if val_scores:
            best_name = max(val_scores, key=val_scores.get)
        else:
            best_name = "majority"

    final_preprocessor = Preprocessor().fit(X_raw)
    X_full = final_preprocessor.transform(X_raw)
    X_test = final_preprocessor.transform(test_raw)

    if best_name == "majority" or X_full.shape[1] == 0:
        most_common = y_raw.mode().iloc[0] if not y_raw.mode().empty else labels[0]
        test_pred = np.full(len(X_test), most_common, dtype=object)
    else:
        final_models = make_models(final_preprocessor.categorical_cols)
        final_model = final_models[best_name]
        final_model.fit(X_full, y_raw)
        test_pred = safe_predict(final_model, X_test, labels)

    pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "readmitted": test_pred,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()