import ast
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/tmdb-box-office-prediction/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/tmdb-box-office-prediction/data/test.parquet"
TARGET = "revenue"
RANDOM_STATE = 42


def safe_parse_list(value):
    """Parse a value that may be a list, dict, JSON-like string, or missing."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"nan", "none", "null", "[]", "{}"}:
            return []
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return list(obj)
            if isinstance(obj, dict):
                return [obj]
            return [obj]
        except Exception:
            pass
        try:
            obj = json.loads(s)
            if isinstance(obj, (list, tuple)):
                return list(obj)
            if isinstance(obj, dict):
                return [obj]
            return [obj]
        except Exception:
            return [s]
    return [value]


def get_names(parsed):
    """Extract 'name' fields from a parsed list of dicts, or string values."""
    names = []
    for item in parsed:
        if isinstance(item, dict):
            name = item.get("name")
            if name is not None:
                names.append(str(name))
        elif item is not None:
            names.append(str(item))
    return names


def get_directors(parsed):
    """Extract director names from crew records."""
    directors = []
    for item in parsed:
        if isinstance(item, dict) and str(item.get("job", "")).lower() == "director":
            name = item.get("name")
            if name is not None:
                directors.append(str(name))
    return directors


def get_departments(parsed):
    """Extract department names from crew records."""
    departments = []
    for item in parsed:
        if isinstance(item, dict):
            department = item.get("department")
            if department is not None:
                departments.append(str(department))
    return departments


class TMDBTransformer:
    """
    Fit on a training frame and transform raw TMDB frames into numeric features.
    All category selection and imputation statistics are learned only from the
    frame passed to fit(), avoiding leakage from validation/test data.
    """

    def __init__(self, top_n=50):
        self.top_n = top_n
        self.list_cols = [
            "belongs_to_collection",
            "genres",
            "production_companies",
            "production_countries",
            "spoken_languages",
            "Keywords",
            "cast",
            "crew",
        ]
        self.cat_cols = ["original_language", "status"]
        self.top = {}
        self.cats = {}
        self.runtime_median = 0.0
        self.year_median = 0.0
        self.month_median = 6.0
        self.day_median = 15.0
        self.department_cats = []

    def _parse_dates(self, df):
        if "release_date" not in df:
            return pd.Series(pd.NaT, index=df.index)

        s = df["release_date"]
        dates = pd.to_datetime(s, errors="coerce")

        if dates.isna().any():
            mask = dates.isna() & s.notna()
            if mask.any():
                try:
                    dates.loc[mask] = pd.to_datetime(
                        s.loc[mask], errors="coerce", format="%m/%d/%y"
                    )
                except Exception:
                    pass

        return dates

    def fit(self, df):
        # Runtime imputation statistic.
        if "runtime" in df:
            med = df["runtime"].median()
            self.runtime_median = float(med) if pd.notna(med) else 0.0

        # Release date imputation statistics.
        dates = self._parse_dates(df)
        if dates.notna().any():
            year_med = dates.dt.year.median()
            month_med = dates.dt.month.median()
            day_med = dates.dt.day.median()
            self.year_median = float(year_med) if pd.notna(year_med) else 0.0
            self.month_median = float(month_med) if pd.notna(month_med) else 6.0
            self.day_median = float(day_med) if pd.notna(day_med) else 15.0

        # Top entity names for list-valued columns.
        for col in self.list_cols:
            all_names = []
            if col in df:
                for value in df[col]:
                    all_names.extend(get_names(safe_parse_list(value)))
            if all_names:
                freq = pd.Series(all_names).value_counts()
                self.top[col] = list(freq.head(self.top_n).index)
            else:
                self.top[col] = []

        # Top directors from crew.
        all_directors = []
        if "crew" in df:
            for value in df["crew"]:
                all_directors.extend(get_directors(safe_parse_list(value)))
        if all_directors:
            freq = pd.Series(all_directors).value_counts()
            self.top["directors"] = list(freq.head(self.top_n).index)
        else:
            self.top["directors"] = []

        # Top crew departments.
        all_departments = []
        if "crew" in df:
            for value in df["crew"]:
                all_departments.extend(get_departments(safe_parse_list(value)))
        if all_departments:
            freq = pd.Series(all_departments).value_counts()
            self.department_cats = list(freq.head(20).index)
        else:
            self.department_cats = []

        # Low-cardinality categorical columns.
        for col in self.cat_cols:
            if col in df:
                vals = df[col].dropna().astype(str)
                self.cats[col] = list(vals.value_counts().head(50).index)
            else:
                self.cats[col] = []

        return self

    def _text_features(self, out, df, col, prefix):
        n = len(df)
        if col in df:
            s = df[col]
        else:
            s = pd.Series([None] * n, index=df.index)

        out[f"{prefix}_missing"] = s.isna().astype(int)
        text = s.fillna("").astype(str)
        out[f"{prefix}_len"] = text.str.len()
        out[f"{prefix}_words"] = text.apply(lambda x: len(x.split()))

    def transform(self, df):
        n = len(df)
        out = pd.DataFrame(index=df.index)

        def raw_col(name, default=None):
            if name in df:
                return df[name]
            return pd.Series([default] * n, index=df.index)

        # Budget features.
        budget_raw = raw_col("budget", 0)
        out["budget_missing"] = budget_raw.isna().astype(int)
        budget = budget_raw.fillna(0).clip(lower=0)
        out["budget"] = budget
        out["budget_log"] = np.log1p(budget)
        out["budget_is_zero"] = (budget <= 0).astype(int)

        # Popularity features.
        popularity_raw = raw_col("popularity", 0)
        out["popularity_missing"] = popularity_raw.isna().astype(int)
        popularity = popularity_raw.fillna(0).clip(lower=0)
        out["popularity"] = popularity
        out["popularity_log"] = np.log1p(popularity)

        # Runtime features.
        runtime_raw = raw_col("runtime", np.nan)
        out["runtime_missing"] = runtime_raw.isna().astype(int)
        out["runtime"] = runtime_raw.fillna(self.runtime_median)

        # Release date features.
        dates = self._parse_dates(df)
        out["release_date_missing"] = dates.isna().astype(int)
        out["release_year"] = dates.dt.year.fillna(self.year_median)
        out["release_month"] = dates.dt.month.fillna(self.month_median)
        out["release_day"] = dates.dt.day.fillna(self.day_median)

        # Text features.
        self._text_features(out, df, "overview", "overview")
        self._text_features(out, df, "title", "title")
        self._text_features(out, df, "tagline", "tagline")

        # Simple presence features.
        out["has_homepage"] = raw_col("homepage", None).notna().astype(int)
        out["has_poster"] = raw_col("poster_path", None).notna().astype(int)
        out["has_imdb_id"] = raw_col("imdb_id", None).notna().astype(int)

        # List-valued features.
        for col in self.list_cols:
            s = raw_col(col, None)
            out[f"{col}_missing"] = s.isna().astype(int)

            parsed = s.apply(safe_parse_list)
            out[f"{col}_count"] = parsed.apply(len)

            names = parsed.apply(get_names)
            out[f"{col}_name_count"] = names.apply(len)

            name_sets = [set(lst) for lst in names]
            for i, cat in enumerate(self.top.get(col, [])):
                out[f"{col}_top_{i}"] = np.array(
                    [cat in name_set for name_set in name_sets], dtype=int
                )

        # Crew-specific features.
        if "crew" in df:
            parsed_crew = df["crew"].apply(safe_parse_list)
            directors = parsed_crew.apply(get_directors)
            out["crew_director_count"] = directors.apply(len)

            director_sets = [set(lst) for lst in directors]
            for i, director in enumerate(self.top.get("directors", [])):
                out[f"director_top_{i}"] = np.array(
                    [director in director_set for director_set in director_sets],
                    dtype=int,
                )

            departments = parsed_crew.apply(get_departments)
            dept_sets = [set(lst) for lst in departments]
            for i, dept in enumerate(self.department_cats):
                out[f"crew_department_{i}"] = np.array(
                    [dept in dept_set for dept_set in dept_sets], dtype=int
                )
        else:
            out["crew_director_count"] = 0
            for i in range(self.top_n):
                out[f"director_top_{i}"] = 0
            for i in range(20):
                out[f"crew_department_{i}"] = 0

        # Low-cardinality categorical one-hot features.
        for col in self.cat_cols:
            s = raw_col(col, None)
            out[f"{col}_missing"] = s.isna().astype(int)
            vals = s.fillna("__missing__").astype(str)
            for i, cat in enumerate(self.cats.get(col, [])):
                out[f"{col}_{i}"] = (vals == cat).astype(int)

        # Ensure a fully numeric, finite matrix.
        out = out.apply(pd.to_numeric, errors="coerce")
        out = out.fillna(0.0)
        out = out.replace([np.inf, -np.inf], 0.0)

        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def make_models():
    return [
        LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        XGBRegressor(
            n_estimators=1200,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        CatBoostRegressor(
            iterations=1200,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    ]


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])
    train_df[TARGET] = train_df[TARGET].clip(lower=0)

    train_features_raw = train_df.drop(columns=[TARGET], errors="ignore")
    y_train = train_df[TARGET]

    # Validation split to estimate generalisation performance.
    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
        train_features_raw,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    val_transformer = TMDBTransformer(top_n=50)
    X_fit = val_transformer.fit_transform(X_fit_raw)
    X_val = val_transformer.transform(X_val_raw)

    y_fit_log = np.log1p(y_fit)
    y_val_log = np.log1p(y_val)

    val_preds = []
    for model in make_models():
        model.fit(X_fit, y_fit_log)
        val_preds.append(model.predict(X_val))

    val_pred_log = np.mean(val_preds, axis=0)
    val_rmsle = float(np.sqrt(mean_squared_error(y_val_log, val_pred_log)))
    print(f"Validation RMSLE: {val_rmsle:.6f}")

    # Final models are trained on the full training set.
    final_transformer = TMDBTransformer(top_n=50)
    X_train_full = final_transformer.fit_transform(train_features_raw)
    X_test = final_transformer.transform(test_df.drop(columns=[TARGET], errors="ignore"))

    y_train_log = np.log1p(y_train)

    test_preds = []
    for model in make_models():
        model.fit(X_train_full, y_train_log)
        test_preds.append(model.predict(X_test))

    pred_log = np.mean(test_preds, axis=0)
    pred_revenue = np.expm1(np.clip(pred_log, 0.0, None))
    pred_revenue = np.maximum(pred_revenue, 0.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "revenue": pred_revenue,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()