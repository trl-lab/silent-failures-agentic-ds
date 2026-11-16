import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
INPUT_DIR = Path("./input")
WORKING_DIR = Path("./working")
SUBMISSION_DIR = Path("./submission")

for directory in [WORKING_DIR, SUBMISSION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TARGET = "Female (%)"
ID_COL = "id"
SEED = 42
VALIDATION_SIZE = 0.15

np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "col"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def is_count_like(col: str) -> bool:
    s = str(col).lower()
    if "%" in s or "percent" in s or "pct" in s:
        return False
    return any(
        token in s
        for token in [
            "count",
            "number",
            "num",
            "workforce",
            "population",
            "headcount",
            "staff",
            "employee",
            "female",
            "male",
        ]
    )


def is_total_like(col: str) -> bool:
    s = str(col).lower()
    return any(token in s for token in ["total", "sum", "all"])


def is_percent_like(col: str) -> bool:
    s = str(col).lower()
    return "%" in s or "percent" in s or "pct" in s


def is_year_like(col: str) -> bool:
    return "year" in str(col).lower()


# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
train_path = INPUT_DIR / "train.parquet"
test_path = INPUT_DIR / "test.parquet"

if not train_path.exists():
    raise FileNotFoundError(f"Expected training data at {train_path}")
if not test_path.exists():
    raise FileNotFoundError(f"Expected test data at {test_path}")

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

# Test must not contain target; if it does, remove it before processing.
test = test.drop(columns=[TARGET], errors="ignore")

# Keep only rows with usable target in training data.
if TARGET not in train.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train data.")

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

if train.empty:
    raise ValueError("No usable training rows with target values.")

# ---------------------------------------------------------------------------
# Split FIRST, then fit preprocessing only on the training split.
# ---------------------------------------------------------------------------
if len(train) >= 20:
    try:
        strat = pd.qcut(
            train[TARGET], q=min(5, max(2, len(train) // 2)), duplicates="drop"
        )
        train_raw, val_raw, y_train, y_val = train_test_split(
            train,
            train[TARGET],
            test_size=VALIDATION_SIZE,
            random_state=SEED,
            stratify=strat,
        )
    except ValueError:
        train_raw, val_raw, y_train, y_val = train_test_split(
            train,
            train[TARGET],
            test_size=VALIDATION_SIZE,
            random_state=SEED,
        )
elif len(train) >= 2:
    train_raw, val_raw, y_train, y_val = train_test_split(
        train,
        train[TARGET],
        test_size=VALIDATION_SIZE,
        random_state=SEED,
    )
else:
    train_raw = train.copy()
    val_raw = train.iloc[0:0].copy()
    y_train = train[TARGET].copy()
    y_val = train[TARGET].iloc[0:0].copy()

train_raw = train_raw.reset_index(drop=True)
val_raw = val_raw.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature builder: fit on training split only, transform val/test identically.
# ---------------------------------------------------------------------------
class WorkforceFeatureBuilder:
    def __init__(self, target: str):
        self.target = target
        self.cat_cols = []
        self.num_cols = []
        self.generated_num_cols = []
        self.missing_cols = []
        self.onehot_specs = {}
        self.freq_maps = {}
        self.ordinal_encoder = None
        self.ordinal_cols = []
        self.medians = pd.Series(dtype=float)
        self.scaler_mean = pd.Series(dtype=float)
        self.scaler_scale = pd.Series(dtype=float)
        self.log_cols = []
        self.bin_specs = {}
        self.interaction_specs = []
        self.ratio_specs = []
        self.complement_specs = []
        self.year_specs = []
        self.male_col = None
        self.female_col = None
        self.direct_pct_name = "direct_pct"
        self.has_direct_name = "has_direct"
        self.feature_names = []

    def _base_columns(self, df: pd.DataFrame) -> list:
        cols = [c for c in df.columns if c not in [self.target, ID_COL]]
        return cols

    def _coerce_numeric(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        out = df.copy()
        for col in cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def _classify_columns(self, train_fit: pd.DataFrame):
        base_cols = self._base_columns(train_fit)
        cat_cols = []
        num_cols = []

        for col in base_cols:
            series = train_fit[col]
            if (
                series.dtype == bool
                or series.dtype == object
                or str(series.dtype) == "category"
            ):
                cat_cols.append(col)
                continue

            if pd.api.types.is_numeric_dtype(series):
                num_cols.append(col)
                continue

            converted = pd.to_numeric(series, errors="coerce")
            non_missing = series.notna().sum()
            if non_missing > 0 and converted.notna().sum() >= 0.5 * non_missing:
                num_cols.append(col)
            else:
                cat_cols.append(col)

        # Year-like columns should be numeric if possible.
        for col in cat_cols.copy():
            if is_year_like(col):
                converted = pd.to_numeric(train_fit[col], errors="coerce")
                if converted.notna().sum() >= 0.5 * train_fit[col].notna().sum():
                    cat_cols.remove(col)
                    num_cols.append(col)

        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def _fit_generated_numeric(self, train_fit: pd.DataFrame):
        train_fit = self._coerce_numeric(train_fit, self.num_cols)

        generated = []
        seen_generated = set()

        def add_generated(name: str) -> None:
            if name not in self.num_cols and name not in seen_generated:
                seen_generated.add(name)
                generated.append(name)

        total_cols = [c for c in self.num_cols if is_total_like(c)]
        count_cols = [
            c for c in self.num_cols if is_count_like(c) and c not in total_cols
        ]

        # count / total ratios
        seen_ratio_names = set()
        for total in total_cols:
            for count in count_cols:
                if count == total:
                    continue
                name = f"{sanitize_name(count)}__over__{sanitize_name(total)}"
                if name in seen_ratio_names:
                    continue
                seen_ratio_names.add(name)
                self.ratio_specs.append((count, total, name))
                add_generated(name)

        # male/female arithmetic if present
        female_col = next(
            (
                c
                for c in self.num_cols
                if "female" in c.lower() and not is_percent_like(c)
            ),
            None,
        )
        male_col = next(
            (
                c
                for c in self.num_cols
                if "male" in c.lower() and not is_percent_like(c)
            ),
            None,
        )
        self.male_col = male_col
        self.female_col = female_col

        for name in [self.direct_pct_name, self.has_direct_name]:
            if name not in seen_ratio_names:
                seen_ratio_names.add(name)
                add_generated(name)

        if female_col is not None and male_col is not None:
            for a, b, name in [
                (female_col, male_col, "female__over__male"),
                (male_col, female_col, "male__over__female"),
            ]:
                if name not in seen_ratio_names:
                    seen_ratio_names.add(name)
                    self.ratio_specs.append((a, b, name))
                    add_generated(name)

            # female share from counts, aligned with target semantics
            name = "female_count__share"
            if name not in seen_ratio_names:
                seen_ratio_names.add(name)
                self.ratio_specs.append((female_col, None, name))
                add_generated(name)

        # percent complements for non-target percentage columns
        seen_complement_names = set()
        for col in self.num_cols:
            if is_percent_like(col):
                name = f"{sanitize_name(col)}__complement"
                if name not in seen_complement_names:
                    seen_complement_names.add(name)
                    self.complement_specs.append((col, name))
                    add_generated(name)

        # year polynomial features
        seen_year_names = set()
        year_cols = [c for c in self.num_cols if is_year_like(c)]
        for col in year_cols:
            names = [f"{sanitize_name(col)}__sq", f"{sanitize_name(col)}__cube"]
            if col not in seen_year_names:
                seen_year_names.add(col)
                self.year_specs.append((col, names))
                for name in names:
                    add_generated(name)

        # interaction categorical specs: state/county/occupation/year-like pairs
        important_cats = [
            c
            for c in self.cat_cols
            if any(
                token in c.lower()
                for token in [
                    "state",
                    "county",
                    "occupation",
                    "job",
                    "role",
                    "type",
                    "category",
                    "group",
                ]
            )
        ][:3]

        seen_interaction_names = set()
        for i in range(len(important_cats)):
            for j in range(i + 1, len(important_cats)):
                a, b = important_cats[i], important_cats[j]
                combined = (
                    train_fit[a].astype("string").fillna("missing")
                    + "||"
                    + train_fit[b].astype("string").fillna("missing")
                )
                if combined.nunique() <= 1500:
                    name = f"{sanitize_name(a)}__x__{sanitize_name(b)}"
                    if name not in seen_interaction_names:
                        seen_interaction_names.add(name)
                        self.interaction_specs.append((a, b, name))
                        add_generated(name)

        self.generated_num_cols = generated

    def _generate_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._coerce_numeric(df, self.num_cols)

        for count, total, name in self.ratio_specs:
            if total is None:
                if count in out.columns:
                    if self.male_col is not None and self.male_col in out.columns:
                        denominator = out[count] + out[self.male_col]
                    else:
                        denominator = out[count]
                    out[name] = safe_divide(out[count], denominator)
                else:
                    out[name] = np.nan
            elif count in out.columns and total in out.columns:
                out[name] = safe_divide(out[count], out[total])
            else:
                out[name] = np.nan

        for col, name in self.complement_specs:
            out[name] = 100.0 - out[col]

        for col, names in self.year_specs:
            if col in out.columns:
                out[names[0]] = out[col] ** 2
                out[names[1]] = out[col] ** 3
            else:
                out[names[0]] = np.nan
                out[names[1]] = np.nan

        if self.female_col is not None and self.male_col is not None:
            female = out.get(self.female_col)
            male = out.get(self.male_col)
            if female is not None and male is not None:
                denom = female + male
                has_direct = (female.notna() & male.notna() & (denom > 0)).astype(float)
                direct = safe_divide(female, denom) * 100.0
                direct = direct.where(has_direct > 0)
                out[self.direct_pct_name] = direct
                out[self.has_direct_name] = has_direct
            else:
                out[self.direct_pct_name] = np.nan
                out[self.has_direct_name] = 0.0
        else:
            out[self.direct_pct_name] = np.nan
            out[self.has_direct_name] = 0.0

        all_num_cols = self.num_cols + self.generated_num_cols
        for col in all_num_cols:
            if col not in out.columns:
                out[col] = np.nan

        return out

    def compute_direct(self, df: pd.DataFrame) -> pd.Series:
        if self.female_col is None or self.male_col is None:
            return pd.Series(np.nan, index=df.index)
        if self.female_col not in df.columns or self.male_col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        out = self._coerce_numeric(df, [self.female_col, self.male_col])
        female = out[self.female_col]
        male = out[self.male_col]
        denom = female + male
        has_direct = female.notna() & male.notna() & (denom > 0)
        direct = safe_divide(female, denom) * 100.0
        direct = direct.where(has_direct)
        return direct

    def _generate_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for a, b, name in self.interaction_specs:
            if a in out.columns and b in out.columns:
                out[name] = (
                    out[a].astype("string").fillna("missing")
                    + "||"
                    + out[b].astype("string").fillna("missing")
                )
            else:
                out[name] = "missing"

        all_cat_cols = self.cat_cols + [spec[2] for spec in self.interaction_specs]
        for col in all_cat_cols:
            if col not in out.columns:
                out[col] = "missing"
            out[col] = out[col].astype("string").fillna("missing")

        return out

    def _fit_encoders(self, train_fit: pd.DataFrame):
        train_fit = self._generate_categorical_features(train_fit)
        all_cat_cols = self.cat_cols + [spec[2] for spec in self.interaction_specs]

        # Frequency maps and one-hot specs
        for col in all_cat_cols:
            counts = train_fit[col].value_counts(dropna=False)
            self.freq_maps[col] = counts.to_dict()

            if len(counts) <= 50:
                self.onehot_specs[col] = list(counts.index.astype(str))

        # Ordinal encoder for all categoricals, including generated interactions.
        if all_cat_cols:
            self.ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            self.ordinal_encoder.fit(train_fit[all_cat_cols].astype(str))
            self.ordinal_cols = all_cat_cols
        else:
            self.ordinal_encoder = None
            self.ordinal_cols = []

    def _fit_numeric_stats(self, train_fit: pd.DataFrame):
        train_fit = self._generate_numeric_features(train_fit)
        all_num_cols = self.num_cols + self.generated_num_cols

        self.medians = train_fit[all_num_cols].median().fillna(0.0)
        self.scaler_mean = train_fit[all_num_cols].mean().fillna(0.0)
        self.scaler_scale = (
            train_fit[all_num_cols].std(ddof=0).replace(0, 1.0).fillna(1.0)
        )

        # Log features for non-negative skewed columns.
        self.log_cols = []
        for col in all_num_cols:
            s = train_fit[col].dropna()
            if len(s) == 0:
                continue
            if s.min() >= 0:
                skew = s.skew() if len(s) > 2 else 0.0
                if skew > 1.0:
                    self.log_cols.append(col)

        # Binning specs from train quantiles.
        self.bin_specs = {}
        for col in all_num_cols:
            s = train_fit[col].dropna()
            if len(s) < 10:
                continue
            try:
                quantiles = np.unique(np.quantile(s, np.linspace(0, 1, 6)))
                if len(quantiles) >= 3:
                    self.bin_specs[col] = quantiles
            except Exception:
                pass

    def fit(self, train_fit: pd.DataFrame):
        train_fit = train_fit.copy()
        train_fit = train_fit.drop(columns=[self.target, ID_COL], errors="ignore")

        self._classify_columns(train_fit)
        self._fit_generated_numeric(train_fit)
        self._fit_encoders(train_fit)
        self._fit_numeric_stats(train_fit)

        # Missing indicators are based on original base columns only.
        self.missing_cols = self.cat_cols + self.num_cols
        self.feature_names = self._build_feature_names()
        return self

    def _build_feature_names(self):
        names = []
        for col in self.missing_cols:
            names.append(f"{sanitize_name(col)}__missing")

        all_num_cols = self.num_cols + self.generated_num_cols
        for col in all_num_cols:
            names.append(sanitize_name(col))
            names.append(f"{sanitize_name(col)}__z")

        for col in self.log_cols:
            names.append(f"{sanitize_name(col)}__log")

        for col in self.bin_specs:
            names.append(f"{sanitize_name(col)}__bin")

        all_cat_cols = self.cat_cols + [spec[2] for spec in self.interaction_specs]
        for col in all_cat_cols:
            names.append(f"{sanitize_name(col)}__freq")
            names.append(f"{sanitize_name(col)}__ord")
            if col in self.onehot_specs:
                for category in self.onehot_specs[col]:
                    names.append(f"{sanitize_name(col)}__{sanitize_name(category)}")

        # Ensure unique names while preserving order.
        seen = set()
        unique_names = []
        for name in names:
            base = name
            i = 2
            while name in seen:
                name = f"{base}_{i}"
                i += 1
            seen.add(name)
            unique_names.append(name)
        return unique_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(columns=[self.target, ID_COL], errors="ignore")

        out = pd.DataFrame(index=df.index)

        # Missing indicators from original columns.
        for col in self.missing_cols:
            if col in df.columns:
                out[f"{sanitize_name(col)}__missing"] = df[col].isna().astype(int)
            else:
                out[f"{sanitize_name(col)}__missing"] = 1

        # Numeric features and transformations.
        num_df = self._generate_numeric_features(df)
        all_num_cols = self.num_cols + self.generated_num_cols
        for col in all_num_cols:
            out[sanitize_name(col)] = num_df[col].fillna(self.medians.get(col, 0.0))

        for col in all_num_cols:
            out[f"{sanitize_name(col)}__z"] = (
                out[sanitize_name(col)] - self.scaler_mean.get(col, 0.0)
            ) / self.scaler_scale.get(col, 1.0)

        for col in self.log_cols:
            out[f"{sanitize_name(col)}__log"] = np.log1p(
                out[sanitize_name(col)].clip(lower=0)
            )

        for col, edges in self.bin_specs.items():
            binned = pd.cut(
                out[sanitize_name(col)], bins=edges, labels=False, include_lowest=True
            )
            out[f"{sanitize_name(col)}__bin"] = binned.astype(float).fillna(-1.0)

        # Categorical features and encodings.
        cat_df = self._generate_categorical_features(df)
        all_cat_cols = self.cat_cols + [spec[2] for spec in self.interaction_specs]

        for col in all_cat_cols:
            freq_map = self.freq_maps.get(col, {})
            out[f"{sanitize_name(col)}__freq"] = (
                cat_df[col].astype(str).map(freq_map).fillna(0.0).astype(float)
            )

        if self.ordinal_encoder is not None and len(self.ordinal_cols) > 0:
            ordinal_matrix = self.ordinal_encoder.transform(
                cat_df[self.ordinal_cols].astype(str)
            )
            for idx, col in enumerate(self.ordinal_cols):
                out[f"{sanitize_name(col)}__ord"] = ordinal_matrix[:, idx].astype(int)

        for col in all_cat_cols:
            if col in self.onehot_specs:
                for category in self.onehot_specs[col]:
                    out[f"{sanitize_name(col)}__{sanitize_name(category)}"] = (
                        cat_df[col].astype(str) == str(category)
                    ).astype(int)

        # Reindex to the exact fitted feature schema.
        out = out.reindex(columns=self.feature_names)
        out = out.fillna(0.0)

        # Replace any remaining non-finite values.
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return out.astype(float)


# ---------------------------------------------------------------------------
# Fit and transform
# ---------------------------------------------------------------------------
builder = WorkforceFeatureBuilder(target=TARGET).fit(train_raw)

X_train = builder.transform(train_raw)
X_val = builder.transform(val_raw)
X_test = builder.transform(test)

# Drop constant features in the training split to keep downstream models clean.
train_var = X_train.var(axis=0)
keep_cols = train_var[train_var > 0].index.tolist()
if len(keep_cols) == 0:
    keep_cols = X_train.columns.tolist()

for preserve_col in [builder.direct_pct_name, builder.has_direct_name]:
    if preserve_col in X_train.columns and preserve_col not in keep_cols:
        keep_cols.append(preserve_col)

X_train = X_train[keep_cols]
X_val = X_val[keep_cols]
X_test = X_test[keep_cols]

# ---------------------------------------------------------------------------
# Save processed artifacts for auditability
# ---------------------------------------------------------------------------
X_train.to_parquet(WORKING_DIR / "train_features.parquet")
X_val.to_parquet(WORKING_DIR / "val_features.parquet")
X_test.to_parquet(WORKING_DIR / "test_features.parquet")

pd.DataFrame({TARGET: y_train.values}).to_parquet(WORKING_DIR / "train_target.parquet")
pd.DataFrame({TARGET: y_val.values}).to_parquet(WORKING_DIR / "val_target.parquet")

metadata = {
    "target": TARGET,
    "seed": SEED,
    "validation_size": VALIDATION_SIZE,
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "n_features": int(X_train.shape[1]),
    "feature_names": X_train.columns.tolist(),
    "validation_split": "train_test_split on raw train before preprocessing; preprocessing fit only on training split",
    "metric": "rmse",
}
with open(WORKING_DIR / "feature_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

# ---------------------------------------------------------------------------
# Normalize feature matrices and targets without touching validation/test labels.
# ---------------------------------------------------------------------------
X_train = pd.DataFrame(X_train).astype(float)
X_val = pd.DataFrame(X_val).astype(float)
X_test = pd.DataFrame(X_test).astype(float)

y_train = np.asarray(y_train, dtype=float).reshape(-1)
y_val = np.asarray(y_val, dtype=float).reshape(-1)

X_train = X_train.drop(columns=[TARGET, ID_COL], errors="ignore")
X_val = X_val.drop(columns=[TARGET, ID_COL], errors="ignore")
X_test = X_test.drop(columns=[TARGET, ID_COL], errors="ignore")

n_train = min(len(X_train), len(y_train))
X_train = X_train.iloc[:n_train].reset_index(drop=True)
y_train = y_train[:n_train]

n_val = min(len(X_val), len(y_val))
X_val = X_val.iloc[:n_val].reset_index(drop=True)
y_val = y_val[:n_val]

mask_train = np.isfinite(y_train)
X_train = X_train[mask_train].reset_index(drop=True)
y_train = y_train[mask_train]

mask_val = np.isfinite(y_val)
X_val = X_val[mask_val].reset_index(drop=True)
y_val = y_val[mask_val]

# If the previous step did not leave a usable validation split, create one from train only.
if len(X_val) == 0 and len(X_train) >= 20:
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=SEED,
    )
    X_val = X_val.reset_index(drop=True)
    y_val = np.asarray(y_val, dtype=float).reshape(-1)

# Align columns and sanitize numeric matrices.
X_val = X_val.reindex(columns=X_train.columns).fillna(0.0).astype(float)
X_test = X_test.reindex(columns=X_train.columns).fillna(0.0).astype(float)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

if len(X_train) == 0:
    raise ValueError("No usable training rows available for model training.")

# If no features survived, provide a constant feature so LightGBM can still infer a mean-like prediction.
if X_train.shape[1] == 0:
    X_train = pd.DataFrame({"constant_feature": 1.0}, index=X_train.index)
    X_val = pd.DataFrame({"constant_feature": 1.0}, index=X_val.index)
    X_test = pd.DataFrame({"constant_feature": 1.0}, index=X_test.index)

def _direct_arrays_from_features(X: pd.DataFrame):
    if builder.direct_pct_name in X.columns and builder.has_direct_name in X.columns:
        direct = X[builder.direct_pct_name].astype(float).to_numpy()
        has_direct = X[builder.has_direct_name].astype(float).to_numpy() > 0
        direct = np.where(has_direct, direct, np.nan)
    else:
        direct = np.full(len(X), np.nan, dtype=float)
        has_direct = np.zeros(len(X), dtype=bool)
    return direct, has_direct

direct_train_arr, has_direct_train_arr = _direct_arrays_from_features(X_train)
direct_val_arr, has_direct_val_arr = _direct_arrays_from_features(X_val)
direct_test_arr, has_direct_test_arr = _direct_arrays_from_features(X_test)

y_train_full = np.asarray(y_train, dtype=float).copy()
y_val_full = np.asarray(y_val, dtype=float).copy()

y_train = np.where(
    has_direct_train_arr,
    y_train_full - direct_train_arr,
    0.0,
).astype(float)

# ---------------------------------------------------------------------------
# Percentage-aware post-processing bounds, fitted from training target only.
# The same clipping is applied to validation and test predictions.
# ---------------------------------------------------------------------------
y_min = float(np.nanmin(y_train_full))
y_max = float(np.nanmax(y_train_full))

if y_min >= 0.0 and y_max <= 100.0:
    clip_lower, clip_upper = 0.0, 100.0
else:
    clip_lower, clip_upper = y_min, y_max

train_mean_full = float(np.nanmean(y_train_full))
train_mean = train_mean_full

# ---------------------------------------------------------------------------
# Adaptive LightGBM configuration for small-to-medium tabular demographic data.
# ---------------------------------------------------------------------------
n_train_rows = len(X_train)
n_features = X_train.shape[1]

if n_train_rows < 300:
    base_leaves = 15
    base_min_child = 5
    base_lr = 0.03
    base_n_estimators = 2000
    patience = 100
elif n_train_rows < 1500:
    base_leaves = 31
    base_min_child = 10
    base_lr = 0.02
    base_n_estimators = 3000
    patience = 150
else:
    base_leaves = 63
    base_min_child = 20
    base_lr = 0.015
    base_n_estimators = 4000
    patience = 200


def make_lgbm_params(seed: int) -> dict:
    leaves = max(7, min(int(base_leaves), max(7, n_train_rows // 10)))
    min_child = max(1, min(int(base_min_child), max(1, n_train_rows // 2)))
    colsample = 1.0 if n_features <= 10 else 0.80

    return dict(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=base_n_estimators,
        learning_rate=float(base_lr),
        num_leaves=leaves,
        max_depth=-1,
        min_child_samples=min_child,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=colsample,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_split_gain=0.01,
        max_bin=255,
        random_state=int(seed),
        n_jobs=-1,
        verbose=-1,
    )


def make_xgb_params(seed: int) -> dict:
    colsample = 1.0 if n_features <= 10 else 0.80
    return dict(
        objective="reg:squarederror",
        max_depth=6,
        learning_rate=0.03,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=5,
        n_estimators=base_n_estimators,
        subsample=0.9,
        colsample_bytree=colsample,
        random_state=int(seed),
        n_jobs=-1,
        verbosity=0,
    )


def train_xgb_fold(X_fit, y_fit, X_es, y_es, seed: int, use_early_stopping: bool):
    params = make_xgb_params(seed)
    if not use_early_stopping:
        params["n_estimators"] = min(base_n_estimators, 300)
        model = xgb.XGBRegressor(**params)
        model.fit(X_fit, y_fit, verbose=False)
        return model

    params["early_stopping_rounds"] = patience
    try:
        model = xgb.XGBRegressor(**params)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    except TypeError:
        params.pop("early_stopping_rounds", None)
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_es, y_es)],
            verbose=False,
            early_stopping_rounds=patience,
        )
    return model


# ---------------------------------------------------------------------------
# K-fold training on the training split only.
# Validation split X_val is never used to fit model weights.
# ---------------------------------------------------------------------------
if n_train_rows >= 50:
    n_folds = 5
elif n_train_rows >= 20:
    n_folds = 3
elif n_train_rows >= 5:
    n_folds = 2
else:
    n_folds = 1

kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

val_preds = []
test_preds = []
oof_pred_full = np.full(len(y_train), np.nan, dtype=float)
fold_infos = []

for fold_idx, (train_idx, fold_val_idx) in enumerate(kf.split(X_train)):
    X_fit = X_train.iloc[train_idx].reset_index(drop=True)
    y_fit = y_train[train_idx]

    X_es = X_train.iloc[fold_val_idx].reset_index(drop=True)
    y_es = y_train[fold_val_idx]

    params = make_lgbm_params(seed=SEED + fold_idx)

    # If the internal fold validation set is too small, avoid unstable early stopping.
    if len(fold_val_idx) < 5:
        params["n_estimators"] = min(base_n_estimators, 300)
        model = LGBMRegressor(**params)
        model.fit(X_fit, y_fit)
        best_iter = int(model.n_estimators)
    else:
        model = LGBMRegressor(**params)
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_es, y_es)],
            eval_metric="rmse",
            callbacks=[
                early_stopping(stopping_rounds=patience, verbose=False, min_delta=1e-4),
                log_evaluation(period=0),
            ],
        )
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is None:
            best_iter = getattr(model.booster_, "best_iteration", 0)
        if best_iter is None or best_iter <= 0:
            best_iter = int(model.n_estimators)
        best_iter = int(max(1, min(best_iter, model.n_estimators)))

    model_path = WORKING_DIR / f"lgbm_fold_{fold_idx}.txt"
    model.booster_.save_model(str(model_path))

    # Load the saved model and use it for all downstream inference.
    booster = lgb.Booster(model_file=str(model_path))

    lgb_val_pred = None
    lgb_oof_pred = None
    if len(X_val) > 0:
        lgb_val_pred = np.asarray(booster.predict(X_val, num_iteration=best_iter), dtype=float)
    else:
        lgb_oof_pred = np.asarray(booster.predict(X_es, num_iteration=best_iter), dtype=float)

    lgb_test_pred = None
    if len(X_test) > 0:
        lgb_test_pred = np.asarray(booster.predict(X_test, num_iteration=best_iter), dtype=float)

    xgb_val_pred = None
    xgb_oof_pred = None
    xgb_test_pred = None
    if XGB_AVAILABLE:
        try:
            xgb_model = train_xgb_fold(
                X_fit,
                y_fit,
                X_es,
                y_es,
                seed=SEED + fold_idx,
                use_early_stopping=len(fold_val_idx) >= 5,
            )
            if len(X_val) > 0:
                xgb_val_pred = np.asarray(xgb_model.predict(X_val), dtype=float)
            else:
                xgb_oof_pred = np.asarray(xgb_model.predict(X_es), dtype=float)
            if len(X_test) > 0:
                xgb_test_pred = np.asarray(xgb_model.predict(X_test), dtype=float)
        except Exception:
            xgb_val_pred = None
            xgb_oof_pred = None
            xgb_test_pred = None

    if len(X_val) > 0:
        if xgb_val_pred is not None:
            val_pred = (lgb_val_pred + xgb_val_pred) / 2.0
        else:
            val_pred = lgb_val_pred
        val_preds.append(np.asarray(val_pred, dtype=float))
    else:
        if xgb_oof_pred is not None:
            oof_pred = (lgb_oof_pred + xgb_oof_pred) / 2.0
        else:
            oof_pred = lgb_oof_pred
        oof_pred_full[fold_val_idx] = np.asarray(oof_pred, dtype=float)

    if len(X_test) > 0:
        if xgb_test_pred is not None:
            test_pred = (lgb_test_pred + xgb_test_pred) / 2.0
        else:
            test_pred = lgb_test_pred
        test_preds.append(np.asarray(test_pred, dtype=float))

    fold_infos.append(
        {
            "fold": int(fold_idx),
            "best_iteration": int(best_iter),
            "train_rows": int(len(train_idx)),
            "fold_val_rows": int(len(fold_val_idx)),
            "model_file": str(model_path),
        }
    )

# ---------------------------------------------------------------------------
# Fallback full-target model for rows without a usable direct formula.
# ---------------------------------------------------------------------------
fallback_train_pred = np.full(len(X_train), train_mean_full, dtype=float)
fallback_val_pred = np.full(len(X_val), train_mean_full, dtype=float)
fallback_test_pred = np.full(len(X_test), train_mean_full, dtype=float)

need_fallback = (
    bool(np.any(~has_direct_train_arr))
    or bool(np.any(~has_direct_val_arr))
    or bool(np.any(~has_direct_test_arr))
)

if need_fallback and len(X_train) > 0:
    fb_val_preds = []
    fb_test_preds = []
    fb_oof_pred = np.full(len(y_train_full), np.nan, dtype=float)

    for fb_fold_idx, (fb_train_idx, fb_fold_val_idx) in enumerate(kf.split(X_train)):
        X_fb_fit = X_train.iloc[fb_train_idx].reset_index(drop=True)
        y_fb_fit = y_train_full[fb_train_idx]
        X_fb_es = X_train.iloc[fb_fold_val_idx].reset_index(drop=True)
        y_fb_es = y_train_full[fb_fold_val_idx]

        fb_params = make_lgbm_params(seed=SEED + 1000 + fb_fold_idx)
        if len(fb_fold_val_idx) < 5:
            fb_params["n_estimators"] = min(base_n_estimators, 300)
            fb_model = LGBMRegressor(**fb_params)
            fb_model.fit(X_fb_fit, y_fb_fit)
            fb_best_iter = int(fb_model.n_estimators)
        else:
            fb_model = LGBMRegressor(**fb_params)
            fb_model.fit(
                X_fb_fit,
                y_fb_fit,
                eval_set=[(X_fb_es, y_fb_es)],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(
                        stopping_rounds=patience,
                        verbose=False,
                        min_delta=1e-4,
                    ),
                    log_evaluation(period=0),
                ],
            )
            fb_best_iter = getattr(fb_model, "best_iteration_", None)
            if fb_best_iter is None:
                fb_best_iter = getattr(fb_model.booster_, "best_iteration", 0)
            if fb_best_iter is None or fb_best_iter <= 0:
                fb_best_iter = int(fb_model.n_estimators)
            fb_best_iter = int(max(1, min(fb_best_iter, fb_model.n_estimators)))

        fb_oof = fb_model.predict(X_fb_es, num_iteration=fb_best_iter)
        fb_oof_pred[fb_fold_val_idx] = np.asarray(fb_oof, dtype=float)

        if len(X_val) > 0:
            fb_val_preds.append(
                np.asarray(fb_model.predict(X_val, num_iteration=fb_best_iter), dtype=float)
            )
        if len(X_test) > 0:
            fb_test_preds.append(
                np.asarray(fb_model.predict(X_test, num_iteration=fb_best_iter), dtype=float)
            )

    if np.all(np.isfinite(fb_oof_pred)):
        fallback_train_pred = np.nan_to_num(
            fb_oof_pred,
            nan=train_mean_full,
            posinf=clip_upper,
            neginf=clip_lower,
        )
        fallback_train_pred = np.clip(fallback_train_pred, clip_lower, clip_upper)

    if len(fb_val_preds) > 0:
        fallback_val_pred = np.nanmean(np.vstack(fb_val_preds), axis=0)
        fallback_val_pred = np.nan_to_num(
            fallback_val_pred,
            nan=train_mean_full,
            posinf=clip_upper,
            neginf=clip_lower,
        )
        fallback_val_pred = np.clip(fallback_val_pred, clip_lower, clip_upper)

    if len(fb_test_preds) > 0:
        fallback_test_pred = np.nanmean(np.vstack(fb_test_preds), axis=0)
        fallback_test_pred = np.nan_to_num(
            fallback_test_pred,
            nan=train_mean_full,
            posinf=clip_upper,
            neginf=clip_lower,
        )
        fallback_test_pred = np.clip(fallback_test_pred, clip_lower, clip_upper)

# ---------------------------------------------------------------------------
# Validation metric: official RMSE on the held-out validation split.
# If no validation split exists, fall back to out-of-fold training predictions.
# ---------------------------------------------------------------------------
if len(X_val) > 0 and len(val_preds) > 0:
    residual_val_avg = np.nanmean(np.vstack(val_preds), axis=0)
    residual_val_avg = np.nan_to_num(
        residual_val_avg,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    val_pred_avg = np.where(
        has_direct_val_arr,
        direct_val_arr + residual_val_avg,
        fallback_val_pred,
    )
    val_pred_avg = np.nan_to_num(
        val_pred_avg,
        nan=train_mean_full,
        posinf=clip_upper,
        neginf=clip_lower,
    )
    val_pred_avg = np.clip(val_pred_avg, clip_lower, clip_upper)
    score = float(np.sqrt(np.mean((y_val_full - val_pred_avg) ** 2)))
else:
    if np.all(np.isfinite(oof_pred_full)):
        residual_oof = np.nan_to_num(
            oof_pred_full,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        oof_pred_full = np.where(
            has_direct_train_arr,
            direct_train_arr + residual_oof,
            fallback_train_pred,
        )
        oof_pred_full = np.nan_to_num(
            oof_pred_full,
            nan=train_mean_full,
            posinf=clip_upper,
            neginf=clip_lower,
        )
        oof_pred_full = np.clip(oof_pred_full, clip_lower, clip_upper)
        score = float(np.sqrt(np.mean((y_train_full - oof_pred_full) ** 2)))
    else:
        score = 0.0 if len(y_train_full) == 0 else float("nan")

# ---------------------------------------------------------------------------
# Test inference using the same saved fold models and identical post-processing.
# ---------------------------------------------------------------------------
if len(X_test) > 0 and len(test_preds) > 0:
    residual_test_avg = np.nanmean(np.vstack(test_preds), axis=0)
    residual_test_avg = np.nan_to_num(
        residual_test_avg,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    test_pred_avg = np.where(
        has_direct_test_arr,
        direct_test_arr + residual_test_avg,
        fallback_test_pred,
    )
    test_pred_avg = np.nan_to_num(
        test_pred_avg,
        nan=train_mean_full,
        posinf=clip_upper,
        neginf=clip_lower,
    )
    test_pred_avg = np.clip(test_pred_avg, clip_lower, clip_upper)
else:
    test_pred_avg = np.array([], dtype=float)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred_avg), dtype=int),
        TARGET: test_pred_avg.astype(float),
    }
)
submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

# ---------------------------------------------------------------------------
# Save training metadata for auditability.
# ---------------------------------------------------------------------------
metadata = {
    "target": TARGET,
    "metric": "rmse",
    "score": float(score) if np.isfinite(score) else None,
    "clip_bounds": [float(clip_lower), float(clip_upper)],
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "n_features": int(X_train.shape[1]),
    "n_folds": int(n_folds),
    "validation_protocol": (
        "K-fold LightGBM/XGBoost ensemble trained only on X_train residual target; "
        "final predictions add direct_pct where available and use a full-target fallback otherwise; "
        "X_val used only for final RMSE evaluation; "
        "same saved fold models used for validation and test inference."
    ),
    "folds": fold_infos,
}

with open(WORKING_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"Final Validation Score: {score}")
