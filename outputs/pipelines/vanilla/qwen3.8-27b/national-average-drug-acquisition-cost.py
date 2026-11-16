import pandas as pd
import numpy as np
import re
import warnings

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/national-average-drug-acquisition-cost/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/national-average-drug-acquisition-cost/data/test.parquet"
TARGET = "NADAC Per Unit"

DATE_COLS = [
    "Effective Date",
    "Corresponding Generic Drug Effective Date",
    "As of Date",
]

DESC_COL = "NDC Description"
NDC_COL = "NDC"
CODE_COL = "Explanation Code"


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df[TARGET] = df[TARGET].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[TARGET])
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = {}

    for col in DATE_COLS:
        if col not in df.columns:
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        parsed[col] = dt
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_dayofweek"] = dt.dt.dayofweek

    if "Effective Date" in parsed and "As of Date" in parsed:
        df["effective_asof_days"] = (parsed["As of Date"] - parsed["Effective Date"]).dt.days
    if "Corresponding Generic Drug Effective Date" in parsed and "As of Date" in parsed:
        df["generic_asof_days"] = (
            parsed["As of Date"] - parsed["Corresponding Generic Drug Effective Date"]
        ).dt.days

    for col in DATE_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


def add_description_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if DESC_COL not in df.columns:
        return df

    desc = df[DESC_COL].fillna("").astype(str)
    lower = desc.str.lower()

    df["desc_len"] = desc.str.len()
    df["desc_word_count"] = desc.str.split().str.len().fillna(0).astype(int)
    df["desc_digit_count"] = desc.str.count(r"\d")
    df["desc_number_count"] = desc.str.count(r"\d+(?:\.\d+)?")

    first_number = desc.str.extract(r"(\d+(?:\.\d+)?)")[0]
    df["desc_first_number"] = pd.to_numeric(first_number, errors="coerce")

    last_number = desc.str.extract(r".*(\d+(?:\.\d+)?)(?:\D|$)")[0]
    df["desc_last_number"] = pd.to_numeric(last_number, errors="coerce")

    tokens = [
        "mg",
        "ml",
        "mcg",
        "tab",
        "tablet",
        "capsule",
        "cap",
        "inj",
        "injection",
        "solution",
        "syrup",
        "suspension",
        "inhal",
        "patch",
        "kit",
        "pack",
        "pak",
        "vial",
        "amp",
        "unit",
        "dose",
        "film",
        "coated",
        "extended",
        "er",
        "xr",
        "sr",
        "ir",
        "sublingual",
        "nasal",
        "ophthalmic",
        "topical",
        "oral",
        "rectal",
        "dental",
        "powder",
        "emulsion",
        "foam",
        "gel",
        "cream",
        "ointment",
        "spray",
        "metered",
        "puff",
        "pen",
        "autoinjector",
        "prefilled",
        "syringe",
        "system",
    ]

    for token in tokens:
        df[f"desc_has_{token}"] = lower.str.contains(token, regex=False).astype(int)

    return df


def add_ndc_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if NDC_COL not in df.columns:
        return df

    ndc = pd.to_numeric(df[NDC_COL], errors="coerce")
    ndc_filled = ndc.fillna(0)

    df["ndc_log"] = np.log1p(ndc_filled)
    df["ndc_mod_1000"] = ndc_filled % 1000
    df["ndc_mod_10000"] = ndc_filled % 10000
    df["ndc_div_10000"] = ndc_filled // 10000
    df["ndc_len"] = df[NDC_COL].astype(str).str.len()

    return df


def add_code_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if CODE_COL not in df.columns:
        return df

    codes = df[CODE_COL].fillna("").astype(str)
    df["code_count"] = codes.str.split(",").str.len().where(codes != "", 0)

    for code in range(1, 21):
        df[f"code_has_{code}"] = codes.str.contains(rf"\b{code}\b", regex=True).astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_date_features(df)
    df = add_description_features(df)
    df = add_ndc_features(df)
    df = add_code_features(df)

    drop_cols = [DESC_COL, CODE_COL]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


class Preprocessor:
    def __init__(self, high_card_threshold: int = 50):
        self.high_card_threshold = high_card_threshold
        self.n_train = 0
        self.cat_cols = []
        self.high_card_cols = []
        self.dummy_cols = {}
        self.freq_maps = {}
        self.medians = pd.Series(dtype=float)

    def _categorical_columns(self, X: pd.DataFrame):
        cols = []
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                cols.append(col)
        return cols

    def _sanitize(self, value) -> str:
        s = str(value)
        s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
        return s or "missing"

    def fit(self, X: pd.DataFrame):
        X = X.copy()
        self.n_train = max(1, len(X))

        self.cat_cols = []
        self.high_card_cols = []
        self.dummy_cols = {}
        self.freq_maps = {}

        for col in self._categorical_columns(X):
            nunique = X[col].nunique(dropna=True)
            if nunique <= self.high_card_threshold:
                self.cat_cols.append(col)
                self.dummy_cols[col] = [self._sanitize(x) for x in X[col].dropna().unique()]
            else:
                self.high_card_cols.append(col)
                freq = X[col].value_counts(dropna=True)
                self.freq_maps[col] = {self._sanitize(k): float(v) for k, v in freq.items()}

        X_num = self._transform(X)
        self.medians = X_num.median()
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.cat_cols + self.high_card_cols:
            if col in X.columns:
                X[col] = X[col].astype(object).fillna("missing").astype(str)

        for col in self.cat_cols:
            if col not in X.columns:
                continue
            X[col] = X[col].map(self._sanitize)
            for cat in self.dummy_cols[col]:
                X[f"{col}__{cat}"] = (X[col] == cat).astype(int)
            X.drop(columns=[col], inplace=True)

        for col in self.high_card_cols:
            if col not in X.columns:
                continue
            freq = self.freq_maps[col]
            X[col] = (
                X[col]
                .map(lambda v: freq.get(self._sanitize(v), 0.0))
                .astype(float)
                / float(self.n_train)
            )

        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors="coerce")

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = self._transform(X)
        X_num = X_num.fillna(self.medians)
        X_num = X_num.replace([np.inf, -np.inf], np.nan)
        X_num = X_num.fillna(0.0)
        return X_num


def make_validation_mask(train_df: pd.DataFrame) -> np.ndarray:
    n = len(train_df)
    mask = pd.Series(False, index=train_df.index)

    if n <= 1:
        return mask.to_numpy()

    for col in ["Effective Date", "As of Date"]:
        if col not in train_df.columns:
            continue

        dt = pd.to_datetime(train_df[col], errors="coerce").dropna()
        if len(dt) < max(10, int(0.5 * n)) or dt.nunique() <= 1:
            continue

        order = dt.sort_values(kind="mergesort").index
        n_val = max(1, min(int(round(0.2 * len(dt))), len(dt) - 1))
        val_idx = order[-n_val:]
        mask.loc[val_idx] = True
        return mask.to_numpy()

    _, val_idx = train_test_split(train_df.index, test_size=0.2, random_state=42)
    mask.loc[val_idx] = True
    return mask.to_numpy()


def safe_rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return np.nan

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return np.nan

    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def build_models():
    return {
        "lgbm": lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if len(test_df) == 0:
        pd.DataFrame(
            {
                "id": np.array([], dtype=int),
                TARGET: np.array([], dtype=float),
            }
        ).to_csv("predictions.csv", index=False)
        return

    train_df = clean_target(train_df).reset_index(drop=True)

    if len(train_df) == 0:
        pd.DataFrame(
            {
                "id": np.arange(len(test_df), dtype=int),
                TARGET: np.zeros(len(test_df), dtype=float),
            }
        ).to_csv("predictions.csv", index=False)
        return

    y_full = train_df[TARGET].astype(float)
    X_full_raw = train_df.drop(columns=[TARGET]).copy()
    X_test_raw = test_df.drop(columns=[TARGET], errors="ignore").copy()

    X_full_raw = engineer_features(X_full_raw)
    X_test_raw = engineer_features(X_test_raw)

    models = build_models()
    weights = {name: 1.0 / len(models) for name in models}

    val_mask = make_validation_mask(train_df)
    train_idx = train_df.index[~val_mask]
    val_idx = train_df.index[val_mask]

    if len(train_idx) == 0:
        train_idx = train_df.index
        val_idx = train_df.index[:0]

    if len(val_idx) > 0 and X_full_raw.shape[1] > 0:
        X_train_raw = X_full_raw.loc[train_idx]
        X_val_raw = X_full_raw.loc[val_idx]
        y_train = y_full.loc[train_idx]
        y_val = y_full.loc[val_idx]

        preprocessor = Preprocessor(high_card_threshold=50).fit(X_train_raw)
        X_train = preprocessor.transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)

        common_cols = list(X_train.columns)
        X_val = X_val.reindex(columns=common_cols, fill_value=0.0)

        if X_train.shape[1] > 0:
            val_scores = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                val_scores[name] = safe_rmse(y_val, pred)

            finite_scores = {k: v for k, v in val_scores.items() if np.isfinite(v) and v > 0}
            if finite_scores:
                inv = {k: 1.0 / v for k, v in finite_scores.items()}
                total = sum(inv.values())
                weights = {k: v / total for k, v in inv.items()}

    preprocessor_full = Preprocessor(high_card_threshold=50).fit(X_full_raw)
    X_full = preprocessor_full.transform(X_full_raw)
    X_test_full = preprocessor_full.transform(X_test_raw)

    common_cols_full = list(X_full.columns)
    X_test_full = X_test_full.reindex(columns=common_cols_full, fill_value=0.0)

    final_preds = np.zeros(len(X_test_full), dtype=float)

    if X_full.shape[1] > 0:
        for name, model in models.items():
            model.fit(X_full, y_full)
            pred = model.predict(X_test_full)
            final_preds += weights.get(name, 0.0) * np.asarray(pred, dtype=float)

    final_preds = np.nan_to_num(final_preds, nan=0.0, posinf=0.0, neginf=0.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(final_preds), dtype=int),
            TARGET: final_preds.astype(float),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()