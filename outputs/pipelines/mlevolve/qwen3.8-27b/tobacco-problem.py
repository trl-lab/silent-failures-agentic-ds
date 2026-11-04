import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

SEED = 42
TARGET = "number_health_problems"
WORK_DIR = "./working"
SUB_DIR = "./submission"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------
train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")

if TARGET not in train_raw.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train data.")

y_all = train_raw[TARGET].copy()
X_all = train_raw.drop(columns=[TARGET])

# Remove explicit id columns from features.
id_cols = [
    c for c in X_all.columns if str(c).lower() in {"id", "index", "row_id", "sample_id"}
]
X_all = X_all.drop(columns=id_cols, errors="ignore")

X_test = test_raw.copy()
X_test = X_test.drop(
    columns=[
        c
        for c in X_test.columns
        if str(c).lower() in {"id", "index", "row_id", "sample_id"}
    ],
    errors="ignore",
)

# Keep only rows with a valid target.
valid_target = y_all.notna()
y_all = y_all[valid_target].astype(int)
X_all = X_all.loc[valid_target]

# Keep only columns present in both train and test to guarantee identical inference schema.
common_cols = [c for c in X_all.columns if c in X_test.columns]
X_all = X_all[common_cols]
X_test = X_test[common_cols]

# ------------------------------------------------------------------
# 2. Split FIRST: train -> train / validation. Test remains untouched.
# ------------------------------------------------------------------
if len(X_all) <= 5 or y_all.nunique() <= 1:
    # Degenerate tiny-data fallback: use all data for training and an empty validation set.
    X_train, X_val = X_all.copy(), X_all.iloc[0:0].copy()
    y_train, y_val = y_all.copy(), y_all.iloc[0:0].copy()
else:
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            stratify=y_all,
            random_state=SEED,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=SEED,
        )


# ------------------------------------------------------------------
# 3. Feature engineering: fit only on train, transform val/test.
# ------------------------------------------------------------------
class TobaccoFeatureBuilder:
    """
    Leakage-safe tabular feature builder.

    Fit statistics are learned only from the training split:
      - numeric medians/stds/quantiles
      - categorical categories/frequencies
      - missingness indicators
      - interaction terms selected from train-only correlation
    """

    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.missing_cols = []
        self.num_median = None
        self.num_std = None
        self.num_q1 = None
        self.num_q3 = None
        self.num_log_cols = []
        self.cat_categories = {}
        self.cat_freq = {}
        self.onehot_cols = []
        self.interaction_pairs = []
        self.ratio_pairs = []
        self.keep_cols = None

    @staticmethod
    def _is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "TobaccoFeatureBuilder":
        self.numeric_cols = [c for c in X.columns if self._is_numeric_series(X[c])]
        self.categorical_cols = [
            c for c in X.columns if not self._is_numeric_series(X[c])
        ]

        # Remove constant features based on train only.
        self.numeric_cols = [
            c for c in self.numeric_cols if X[c].nunique(dropna=True) > 1
        ]
        self.categorical_cols = [
            c for c in self.categorical_cols if X[c].nunique(dropna=True) > 1
        ]

        self.missing_cols = [c for c in X.columns if X[c].isna().any()]

        # Numeric statistics.
        if self.numeric_cols:
            num = X[self.numeric_cols]
            self.num_median = num.median()
            self.num_std = num.std().replace(0, 1.0)
            self.num_q1 = num.quantile(0.25)
            self.num_q3 = num.quantile(0.75)

            self.num_log_cols = []
            for c in self.numeric_cols:
                s = X[c].dropna()
                if len(s) == 0:
                    continue
                nonnegative = (s >= 0).all()
                positive_fraction = float((s > 0).mean())
                has_variance = float(s.std()) > 0
                if nonnegative and positive_fraction > 0.5 and has_variance:
                    self.num_log_cols.append(c)
        else:
            self.num_median = pd.Series(dtype=float)
            self.num_std = pd.Series(dtype=float)
            self.num_q1 = pd.Series(dtype=float)
            self.num_q3 = pd.Series(dtype=float)

        # Categorical statistics.
        self.cat_categories = {}
        self.cat_freq = {}
        self.onehot_cols = []
        for c in self.categorical_cols:
            freq = X[c].value_counts(dropna=True)
            self.cat_freq[c] = freq
            self.cat_categories[c] = list(freq.index)
            if len(freq) <= 20:
                self.onehot_cols.append(c)

        # Train-only interaction selection using target correlation when available.
        self.interaction_pairs = []
        self.ratio_pairs = []
        if self.numeric_cols:
            if y is not None and len(y) > 0:
                corr = X[self.numeric_cols].corrwith(y)
                top_numeric = (
                    corr.abs()
                    .sort_values(ascending=False)
                    .head(min(6, len(self.numeric_cols)))
                    .index.tolist()
                )
            else:
                top_numeric = self.numeric_cols[:6]

            for i in range(len(top_numeric)):
                for j in range(i + 1, len(top_numeric)):
                    self.interaction_pairs.append((top_numeric[i], top_numeric[j]))

            top_ratio = top_numeric[:4]
            for i in range(len(top_ratio)):
                for j in range(i + 1, len(top_ratio)):
                    self.ratio_pairs.append((top_ratio[i], top_ratio[j]))

        # Fit a temporary transform to determine stable non-constant output columns.
        temp = self.transform(X)
        self.keep_cols = [c for c in temp.columns if temp[c].nunique(dropna=True) > 1]
        if len(self.keep_cols) == 0:
            self.keep_cols = list(temp.columns)

        return self

    def _numeric_block(self, X: pd.DataFrame) -> dict:
        out = {}
        if not self.numeric_cols:
            return out

        num = X[self.numeric_cols].copy()

        for c in self.numeric_cols:
            if c in self.missing_cols:
                out[f"{c}__missing"] = num[c].isna().astype(int).to_numpy()

        filled = num.fillna(self.num_median)
        scaled = (filled - self.num_median) / self.num_std

        for c in self.numeric_cols:
            out[c] = scaled[c].to_numpy()

        for c in self.num_log_cols:
            out[f"{c}__log1p"] = np.log1p(filled[c].clip(lower=0.0)).to_numpy()

        for c in self.numeric_cols:
            q1 = (
                float(self.num_q1[c])
                if pd.notna(self.num_q1[c])
                else float(filled[c].min())
            )
            med = (
                float(self.num_median[c])
                if pd.notna(self.num_median[c])
                else float(filled[c].median())
            )
            q3 = (
                float(self.num_q3[c])
                if pd.notna(self.num_q3[c])
                else float(filled[c].max())
            )
            bins = sorted({q1, med, q3})

            values = filled[c].to_numpy(dtype=float)
            if len(bins) == 0:
                out[f"{c}__bin"] = np.zeros(len(values), dtype=int)
            elif len(bins) == 1:
                out[f"{c}__bin"] = (values > bins[0]).astype(int)
            else:
                out[f"{c}__bin"] = np.digitize(values, bins).astype(int)

        return out

    def _categorical_block(self, X: pd.DataFrame) -> dict:
        out = {}
        for c in self.categorical_cols:
            s = X[c]

            if c in self.missing_cols:
                out[f"{c}__missing"] = s.isna().astype(int).to_numpy()

            cats = self.cat_categories[c]
            mapping = {cat: i for i, cat in enumerate(cats)}
            code = s.map(mapping)
            out[f"{c}__code"] = code.fillna(-1).astype(int).to_numpy()

            freq = self.cat_freq[c]
            out[f"{c}__freq"] = s.map(freq).fillna(0.0).astype(float).to_numpy()

            if c in self.onehot_cols:
                for cat in cats:
                    out[f"{c}__{repr(cat)}"] = (s == cat).astype(int).to_numpy()

        return out

    def _interaction_block(self, X: pd.DataFrame) -> dict:
        out = {}
        if not self.numeric_cols:
            return out

        filled = X[self.numeric_cols].fillna(self.num_median)

        for a, b in self.interaction_pairs:
            out[f"{a}__x__{b}"] = (filled[a] * filled[b]).to_numpy(dtype=float)

        for a, b in self.ratio_pairs:
            denom = filled[b].replace(0, np.nan)
            out[f"{a}__div__{b}"] = (
                (filled[a] / denom).fillna(0.0).to_numpy(dtype=float)
            )

        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Align columns defensively; unseen columns are dropped, missing train columns become NaN.
        expected = self.numeric_cols + self.categorical_cols
        for c in expected:
            if c not in X.columns:
                X[c] = np.nan

        out = {}
        out.update(self._numeric_block(X))
        out.update(self._categorical_block(X))
        out.update(self._interaction_block(X))

        if not out:
            out["const"] = np.zeros(len(X), dtype=float)

        out_df = pd.DataFrame(out, index=X.index)

        # Keep only columns that were non-constant during fit.
        if self.keep_cols is not None:
            out_df = out_df[self.keep_cols]

        # Final safety: replace any remaining non-finite values with 0.
        out_df = out_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return out_df


feature_builder = TobaccoFeatureBuilder()
feature_builder.fit(X_train, y_train)

X_train_fe = feature_builder.transform(X_train)
X_val_fe = feature_builder.transform(X_val)
X_test_fe = feature_builder.transform(X_test)

# Persist processed artifacts for later pipeline steps.
X_train_fe.to_parquet(os.path.join(WORK_DIR, "X_train_fe.parquet"))
X_val_fe.to_parquet(os.path.join(WORK_DIR, "X_val_fe.parquet"))
X_test_fe.to_parquet(os.path.join(WORK_DIR, "X_test_fe.parquet"))
y_train.to_frame(TARGET).to_parquet(os.path.join(WORK_DIR, "y_train.parquet"))
y_val.to_frame(TARGET).to_parquet(os.path.join(WORK_DIR, "y_val.parquet"))
pd.Series(feature_builder.keep_cols, name="feature").to_csv(
    os.path.join(WORK_DIR, "feature_names.csv"),
    index=False,
)

# ------------------------------------------------------------------
# 4. Minimal sanity model for required submission and official metric.
#    This is not the final modeling step; it validates that the engineered
#    features support end-to-end inference and official accuracy evaluation.
# ------------------------------------------------------------------
if len(y_train) == 0:
    # No training rows: fallback to the only available class or 0.
    if y_all.nunique() > 0:
        fallback_class = int(y_all.mode().iloc[0])
    else:
        fallback_class = 0
    val_pred = np.full(len(y_val), fallback_class, dtype=int)
    test_pred = np.full(len(X_test_fe), fallback_class, dtype=int)
    score = float(accuracy_score(y_val, val_pred)) if len(y_val) > 0 else float("nan")
else:
    if y_train.nunique() == 1:
        single_class = int(y_train.iloc[0])
        val_pred = np.full(len(y_val), single_class, dtype=int)
        test_pred = np.full(len(X_test_fe), single_class, dtype=int)
        score = (
            float(accuracy_score(y_val, val_pred)) if len(y_val) > 0 else float("nan")
        )
    else:
        model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=5,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(X_train_fe, y_train)

        if len(y_val) > 0:
            val_pred = model.predict(X_val_fe).astype(int)
            score = float(accuracy_score(y_val, val_pred))
        else:
            val_pred = np.array([], dtype=int)
            score = float("nan")

        test_pred = model.predict(X_test_fe).astype(int)

# ------------------------------------------------------------------
# 5. Submission: id is 0-based integer row index, as specified.
# ------------------------------------------------------------------
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        TARGET: test_pred.astype(int),
    }
)
submission.to_csv(os.path.join(SUB_DIR, "submission.csv"), index=False)

print(f"Final Validation Score: {score}")