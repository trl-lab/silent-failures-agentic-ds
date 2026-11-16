import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TARGET = "H index"
DROP_COLS = ["Sourceid", "Title", "Issn"]
CAT_COLS = ["Type", "Country", "Region", "Publisher"]
NUM_COLS = ["Total Refs.", "Overton", "SDG"]


def _clean_str(x):
    if x is None:
        return ""
    s = str(x)
    if s.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return s


def _coverage_features(x):
    s = _clean_str(x)
    if not s:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", s)]
    ranges = re.findall(r"(\d{4})\s*-\s*(\d{4})", s)

    if not years:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    start = min(years)
    end = max(years)
    n_ranges = len(ranges) if ranges else 1

    if ranges:
        total_years = sum(max(0, int(b) - int(a) + 1) for a, b in ranges)
    else:
        total_years = max(0, end - start + 1)

    span = max(0, end - start + 1)
    return (start, end, total_years, n_ranges, span)


def _category_features(x):
    s = _clean_str(x)
    if not s:
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)

    parts = [p for p in s.split(";") if p.strip()]
    n_categories = len(parts)

    q1 = len(re.findall(r"\(Q1\)", s))
    q2 = len(re.findall(r"\(Q2\)", s))
    q3 = len(re.findall(r"\(Q3\)", s))
    q4 = len(re.findall(r"\(Q4\)", s))

    return (
        n_categories,
        q1,
        q2,
        q3,
        q4,
        int(q1 > 0),
        int(q2 > 0),
        int(q3 > 0),
        int(q4 > 0),
    )


def _area_features(x):
    s = _clean_str(x)
    if not s:
        return (0, 0)

    parts = [p for p in s.split(";") if p.strip()]
    return (len(parts), len(s))


def _issn_count(x):
    s = _clean_str(x)
    if not s:
        return 0
    return len([p for p in s.split(",") if p.strip()])


class Preprocessor:
    def __init__(self):
        self.num_feature_cols = []
        self.medians = {}
        self.encoders = {}
        self.freqs = {}
        self.feature_names = []

    def _derive(self, df):
        df = df.copy()

        for col in NUM_COLS:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        total_refs = df["Total Refs."].fillna(0.0).clip(lower=0.0)
        df["log_total_refs"] = np.log1p(total_refs)
        df["sqrt_total_refs"] = np.sqrt(total_refs)

        if "Issn" in df.columns:
            df["issn_count"] = df["Issn"].apply(_issn_count)
        else:
            df["issn_count"] = 0

        if "Title" in df.columns:
            df["title_word_count"] = (
                df["Title"].fillna("").astype(str).str.split().str.len().fillna(0)
            )
        else:
            df["title_word_count"] = 0

        if "Coverage" in df.columns:
            cov = df["Coverage"].apply(_coverage_features)
            df["coverage_start"] = cov.apply(lambda x: x[0])
            df["coverage_end"] = cov.apply(lambda x: x[1])
            df["coverage_total_years"] = cov.apply(lambda x: x[2])
            df["coverage_n_ranges"] = cov.apply(lambda x: x[3])
            df["coverage_span"] = cov.apply(lambda x: x[4])
        else:
            df["coverage_start"] = np.nan
            df["coverage_end"] = np.nan
            df["coverage_total_years"] = np.nan
            df["coverage_n_ranges"] = np.nan
            df["coverage_span"] = np.nan

        if "Categories" in df.columns:
            cat = df["Categories"].apply(_category_features)
            df["n_categories"] = cat.apply(lambda x: x[0])
            df["q1_count"] = cat.apply(lambda x: x[1])
            df["q2_count"] = cat.apply(lambda x: x[2])
            df["q3_count"] = cat.apply(lambda x: x[3])
            df["q4_count"] = cat.apply(lambda x: x[4])
            df["has_q1"] = cat.apply(lambda x: x[5])
            df["has_q2"] = cat.apply(lambda x: x[6])
            df["has_q3"] = cat.apply(lambda x: x[7])
            df["has_q4"] = cat.apply(lambda x: x[8])
        else:
            df["n_categories"] = 0
            df["q1_count"] = 0
            df["q2_count"] = 0
            df["q3_count"] = 0
            df["q4_count"] = 0
            df["has_q1"] = 0
            df["has_q2"] = 0
            df["has_q3"] = 0
            df["has_q4"] = 0

        if "Areas" in df.columns:
            areas = df["Areas"].apply(_area_features)
            df["n_areas"] = areas.apply(lambda x: x[0])
            df["areas_length"] = areas.apply(lambda x: x[1])
        else:
            df["n_areas"] = 0
            df["areas_length"] = 0

        if "Publisher" in df.columns:
            df["publisher_length"] = df["Publisher"].fillna("").astype(str).str.len()
        else:
            df["publisher_length"] = 0

        for col in NUM_COLS:
            df[f"{col}_missing"] = df[col].isna().astype(int)

        for col in CAT_COLS:
            if col in df.columns:
                df[f"{col}_missing"] = df[col].isna().astype(int)
            else:
                df[f"{col}_missing"] = 1

        return df

    def fit(self, df):
        df = self._derive(df)

        exclude = set(DROP_COLS + CAT_COLS + [TARGET])
        self.num_feature_cols = [c for c in df.columns if c not in exclude]

        for col in self.num_feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median = df[col].median()
            self.medians[col] = 0.0 if pd.isna(median) else float(median)

        for col in CAT_COLS:
            if col not in df.columns:
                df[col] = "missing"
            df[col] = df[col].fillna("missing").astype(str)

            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            encoder.fit(df[[col]])
            self.encoders[col] = encoder

            freq = df[col].value_counts(normalize=True).to_dict()
            self.freqs[col] = freq

        self.feature_names = (
            self.num_feature_cols
            + [f"{col}_code" for col in CAT_COLS]
            + [f"{col}_freq" for col in CAT_COLS]
        )
        return self

    def transform(self, df):
        df = self._derive(df)

        for col in self.num_feature_cols:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(self.medians.get(col, 0.0))

        for col in CAT_COLS:
            if col not in df.columns:
                df[col] = "missing"
            df[col] = df[col].fillna("missing").astype(str)

            if col in self.encoders:
                df[f"{col}_code"] = self.encoders[col].transform(df[[col]]).ravel()
            else:
                df[f"{col}_code"] = -1

            df[f"{col}_freq"] = df[col].map(self.freqs.get(col, {})).fillna(0.0)

        drop_cols = [c for c in DROP_COLS + CAT_COLS + [TARGET] if c in df.columns]
        df = df.drop(columns=drop_cols)

        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0

        return df[self.feature_names]


def make_models(categorical_feature_names):
    lgbm = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        categorical_features=categorical_feature_names,
    )

    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    catboost = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
        cat_features=categorical_feature_names,
    )

    return {"lgbm": lgbm, "xgb": xgb, "catboost": catboost}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/strable/journal-ranking_wide/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/strable/journal-ranking_wide/data/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_df = train_df.dropna(subset=[TARGET]).copy()
    train_df[TARGET] = pd.to_numeric(train_df[TARGET], errors="coerce")
    train_df = train_df.dropna(subset=[TARGET]).copy()

    if len(train_df) < 10:
        X_train = pd.DataFrame(index=train_df.index)
        y_train = train_df[TARGET].astype(float)
        X_test = pd.DataFrame(index=test_df.index)
        preds = np.zeros(len(test_df), dtype=float)
    else:
        train_part, val_part = train_test_split(
            train_df, test_size=0.2, random_state=42
        )

        val_preprocessor = Preprocessor()
        val_preprocessor.fit(train_part)

        X_train_part = val_preprocessor.transform(train_part)
        y_train_part = train_part[TARGET].astype(float)

        X_val = val_preprocessor.transform(val_part)
        y_val = val_part[TARGET].astype(float)

        categorical_feature_names = [f"{col}_code" for col in CAT_COLS]
        models = make_models(categorical_feature_names)

        val_preds = {}
        val_rmses = {}

        for name, model in models.items():
            try:
                model.fit(X_train_part, y_train_part)
                pred = model.predict(X_val)
                pred = np.asarray(pred, dtype=float)
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                pred = np.maximum(pred, 0.0)
                val_preds[name] = pred
                val_rmses[name] = rmse(y_val, pred)
            except Exception:
                val_rmses[name] = np.inf

        if all(np.isfinite(v) for v in val_rmses.values()):
            weights = {name: 1.0 / max(rmse_val, 1e-9) for name, rmse_val in val_rmses.items()}
            weight_sum = sum(weights.values())
            weights = {name: w / weight_sum for name, w in weights.items()}
        else:
            finite = {name: v for name, v in val_rmses.items() if np.isfinite(v)}
            if finite:
                best_name = min(finite, key=finite.get)
                weights = {name: (1.0 if name == best_name else 0.0) for name in val_rmses}
            else:
                weights = {name: 1.0 / len(val_rmses) for name in val_rmses}

        final_preprocessor = Preprocessor()
        final_preprocessor.fit(train_df)

        X_train_full = final_preprocessor.transform(train_df)
        y_train_full = train_df[TARGET].astype(float)
        X_test = final_preprocessor.transform(test_df)

        final_models = make_models(categorical_feature_names)
        test_preds = {}

        for name, model in final_models.items():
            if weights.get(name, 0.0) <= 0:
                continue
            try:
                model.fit(X_train_full, y_train_full)
                pred = model.predict(X_test)
                pred = np.asarray(pred, dtype=float)
                pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                pred = np.maximum(pred, 0.0)
                test_preds[name] = pred
            except Exception:
                test_preds[name] = np.zeros(len(test_df), dtype=float)

        if test_preds:
            preds = np.zeros(len(test_df), dtype=float)
            for name, pred in test_preds.items():
                preds += weights.get(name, 0.0) * pred
        else:
            preds = np.zeros(len(test_df), dtype=float)

    output = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "H index": preds.astype(float),
        }
    )
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()