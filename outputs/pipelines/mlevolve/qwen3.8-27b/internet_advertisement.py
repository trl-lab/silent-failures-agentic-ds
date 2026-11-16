import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
VAL_FRACTION = 0.20

INPUT_DIR = Path("./input")
WORKING_DIR = Path("./working")
SUBMISSION_DIR = Path("./submission")

TRAIN_PATH = INPUT_DIR / "train.parquet"
TEST_PATH = INPUT_DIR / "test.parquet"

WORKING_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
train_raw = pd.read_parquet(TRAIN_PATH)
test_raw = pd.read_parquet(TEST_PATH)

if "class" not in train_raw.columns:
    raise ValueError("Target column 'class' not found in train.parquet.")

train_raw = train_raw.reset_index(drop=True)
test_raw = test_raw.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Target encoding
# ---------------------------------------------------------------------------
def encode_target(y: pd.Series) -> pd.Series:
    """Map target to 1/0. Handles common ad/nonad spellings and boolean-like values."""
    y = y.astype(str).str.strip().str.lower()
    mapping = {
        "ad": 1,
        "ads": 1,
        "advertisement": 1,
        "advertisement1": 1,
        "true": 1,
        "1": 1,
        "nonad": 0,
        "non-ad": 0,
        "non_ad": 0,
        "nonadvertisement": 0,
        "false": 0,
        "0": 0,
    }
    encoded = y.map(mapping)
    if encoded.isna().any():

        def fallback(v: str) -> int:
            if v.startswith("non") or v.startswith("no"):
                return 0
            return int("ad" in v)

        encoded = encoded.fillna(y[encoded.isna()].apply(fallback))
    return encoded.astype(int)


y_train_raw = train_raw["class"].copy()
y_train = encode_target(y_train_raw)

# ---------------------------------------------------------------------------
# 3. Create validation split FIRST, before any stateful feature engineering.
# ---------------------------------------------------------------------------
train_idx, val_idx = train_test_split(
    np.arange(len(train_raw)),
    test_size=VAL_FRACTION,
    random_state=SEED,
    stratify=y_train,
)

train_part = train_raw.loc[train_idx].copy()
val_part = train_raw.loc[val_idx].copy()
y_train_part = y_train.loc[train_idx].copy()
y_val_part = y_train.loc[val_idx].copy()

# ---------------------------------------------------------------------------
# 4. Identify feature types
# ---------------------------------------------------------------------------
feature_cols = [c for c in train_raw.columns if c != "class"]


def is_binary_series(s: pd.Series) -> bool:
    if s.dtype == bool:
        return True
    if not pd.api.types.is_numeric_dtype(s):
        return False
    vals = s.dropna().unique()
    if len(vals) == 0:
        return False
    return set(vals).issubset({0, 1, 0.0, 1.0})


binary_cols = []
continuous_cols = []

for col in feature_cols:
    s = train_raw[col]
    if pd.api.types.is_numeric_dtype(s):
        if is_binary_series(s):
            binary_cols.append(col)
        else:
            continuous_cols.append(col)
    else:
        # Non-numeric columns are unexpected for this dataset; drop them.
        continue

# Fallback if automatic detection finds fewer than three continuous columns.
if len(continuous_cols) < 3:
    continuous_cols = []
    for col in feature_cols:
        s = train_raw[col]
        if pd.api.types.is_numeric_dtype(s) and not is_binary_series(s):
            continuous_cols.append(col)

binary_cols = [c for c in binary_cols if c not in continuous_cols]


# ---------------------------------------------------------------------------
# 5. Fit feature-engineering state ONLY on the training split.
# ---------------------------------------------------------------------------
class FeatureEngineer:
    """Fit on train split, transform train/val/test identically."""

    def __init__(self, continuous_cols, binary_cols):
        self.continuous_cols = list(continuous_cols)
        self.binary_cols = list(binary_cols)
        self.fit_params = {}

    @staticmethod
    def _source_group(col: str) -> str:
        c = col.lower()
        if "imgurl" in c or "image_url" in c or "img_url" in c:
            return "image_url"
        if "url" in c:
            return "url"
        if "alt" in c:
            return "alt"
        if "anchor" in c:
            return "anchor"
        if "near" in c or "word" in c:
            return "near"
        return "other"

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        cont_stats = {}
        for col in self.continuous_cols:
            if col not in df.columns:
                continue

            vals = df[col].dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                if std == 0 or not np.isfinite(std):
                    std = 1.0
                median = float(np.median(vals))
                q01 = (
                    float(np.quantile(vals, 0.01))
                    if len(vals) >= 100
                    else float(np.min(vals))
                )
                q99 = (
                    float(np.quantile(vals, 0.99))
                    if len(vals) >= 100
                    else float(np.max(vals))
                )

                if len(vals) >= 20:
                    edges = np.quantile(vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    edges = np.unique(edges)
                    bins = edges if len(edges) >= 2 else None
                else:
                    bins = None

                cont_stats[col] = {
                    "mean": mean,
                    "std": std,
                    "median": median,
                    "q01": q01,
                    "q99": q99,
                    "bins": bins,
                }
            else:
                cont_stats[col] = {
                    "mean": 0.0,
                    "std": 1.0,
                    "median": 0.0,
                    "q01": 0.0,
                    "q99": 0.0,
                    "bins": None,
                }

        self.fit_params["cont_stats"] = cont_stats

        if len(self.binary_cols) > 0:
            freq = df[self.binary_cols].sum(axis=0).sort_values(ascending=False)
            top_n = min(18, len(freq))
            self.fit_params["top_binary_cols"] = list(freq.index[:top_n])
        else:
            self.fit_params["top_binary_cols"] = []

        self.fit_params["source_groups"] = {
            self._source_group(col): [
                c
                for c in self.binary_cols
                if self._source_group(c) == self._source_group(col)
            ]
            for col in self.binary_cols
        }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cont_stats = self.fit_params["cont_stats"]

        # Continuous features
        for col in self.continuous_cols:
            if col not in out.columns:
                continue

            s = out[col].astype(float)
            miss = s.isna()

            out[f"{col}__missing"] = miss.astype(int)

            stats = cont_stats.get(
                col,
                {
                    "mean": 0.0,
                    "std": 1.0,
                    "median": 0.0,
                    "q01": 0.0,
                    "q99": 0.0,
                    "bins": None,
                },
            )

            out[f"{col}__imputed"] = s.fillna(stats["median"])
            out[f"{col}__z"] = (s - stats["mean"]) / stats["std"]
            out[f"{col}__log1p"] = np.log1p(np.clip(s, 0.0, None))

            bins = stats.get("bins")
            if bins is not None and len(bins) >= 2:
                try:
                    out[f"{col}__bin"] = pd.cut(s, bins=bins, labels=False, right=False)
                except Exception:
                    out[f"{col}__bin"] = np.nan
            else:
                out[f"{col}__bin"] = np.nan

            out[f"{col}__extreme"] = (
                (~miss) & ((s < stats["q01"]) | (s > stats["q99"]))
            ).astype(int)

        # Aggregate continuous known/missing counts.
        if len(self.continuous_cols) > 0:
            cont_matrix = out[self.continuous_cols].astype(float)
            out["n_known_continuous"] = cont_matrix.notna().sum(axis=1)
            out["n_missing_continuous"] = cont_matrix.isna().sum(axis=1)
            out["any_missing_continuous"] = (out["n_missing_continuous"] > 0).astype(
                int
            )
            out["all_missing_continuous"] = (
                out["n_missing_continuous"] == len(self.continuous_cols)
            ).astype(int)
        else:
            out["n_known_continuous"] = 0
            out["n_missing_continuous"] = 0
            out["any_missing_continuous"] = 0
            out["all_missing_continuous"] = 0

        # Pairwise continuous interactions.
        for i, c1 in enumerate(self.continuous_cols):
            for c2 in self.continuous_cols[i + 1 :]:
                if c1 not in out.columns or c2 not in out.columns:
                    continue

                a = out[c1].astype(float)
                b = out[c2].astype(float)

                out[f"{c1}__x__{c2}"] = a * b
                out[f"{c1}__minus__{c2}"] = a - b

                ratio_ab = np.where((b != 0) & a.notna() & b.notna(), a / b, np.nan)
                ratio_ba = np.where((a != 0) & a.notna() & b.notna(), b / a, np.nan)
                out[f"{c1}__over__{c2}"] = ratio_ab
                out[f"{c2}__over__{c1}"] = ratio_ba

                out[f"{c1}__area__{c2}"] = np.where((a > 0) & (b > 0), a * b, np.nan)

        if len(self.continuous_cols) == 3:
            c1, c2, c3 = self.continuous_cols
            a = out[c1].astype(float)
            b = out[c2].astype(float)
            c = out[c3].astype(float)

            out["continuous_triple_product"] = a * b * c
            out["continuous_pairwise_mean"] = (
                out[f"{c1}__x__{c2}"].fillna(0.0)
                + out[f"{c1}__x__{c3}"].fillna(0.0)
                + out[f"{c2}__x__{c3}"].fillna(0.0)
            ) / 3.0

        # Binary phrase features
        for col in self.binary_cols:
            if col not in out.columns:
                continue

            if out[col].dtype == bool:
                out[col] = out[col].astype(int)
            elif not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = (
                    out[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({"": 0, "0": 0, "1": 1, "true": 1, "false": 0})
                    .fillna(0)
                    .astype(int)
                )
            else:
                out[col] = out[col].astype(float)

        if len(self.binary_cols) > 0:
            binary_matrix = out[self.binary_cols].astype(float)
            out["n_binary_on"] = binary_matrix.sum(axis=1)
            out["binary_density"] = out["n_binary_on"] / max(len(self.binary_cols), 1)
        else:
            out["n_binary_on"] = 0
            out["binary_density"] = 0.0

        # Source-level aggregates.
        source_groups = self.fit_params.get("source_groups", {})
        source_names = sorted(source_groups.keys())
        for src in source_names:
            cols = source_groups.get(src, [])
            if cols:
                cols_present = [c for c in cols if c in out.columns]
                if cols_present:
                    out[f"{src}_count"] = out[cols_present].astype(float).sum(axis=1)
                else:
                    out[f"{src}_count"] = 0.0
            else:
                out[f"{src}_count"] = 0.0

        # Interactions between text sources.
        if "url_count" in out.columns and "anchor_count" in out.columns:
            out["url_anchor_interaction"] = out["url_count"] * out["anchor_count"]
        if "image_url_count" in out.columns and "alt_count" in out.columns:
            out["image_url_alt_interaction"] = out["image_url_count"] * out["alt_count"]
        if "anchor_count" in out.columns and "near_count" in out.columns:
            out["anchor_near_interaction"] = out["anchor_count"] * out["near_count"]

        # Text x geometry availability interactions.
        out["text_when_geometry_missing"] = (
            out["n_binary_on"] * out["all_missing_continuous"]
        )
        out["text_when_geometry_known"] = out["n_binary_on"] * out["n_known_continuous"]
        out["text_per_known_geometry"] = out["n_binary_on"] / (
            out["n_known_continuous"] + 1.0
        )

        # Top binary pairwise co-occurrences.
        top_binary_cols = self.fit_params.get("top_binary_cols", [])
        for i, c1 in enumerate(top_binary_cols):
            for c2 in top_binary_cols[i + 1 :]:
                if c1 in out.columns and c2 in out.columns:
                    out[f"{c1}__and__{c2}"] = out[c1].astype(float) * out[c2].astype(
                        float
                    )

        return out


engineer = FeatureEngineer(continuous_cols, binary_cols)
engineer = engineer.fit(train_part)

train_fe = engineer.transform(train_part)
val_fe = engineer.transform(val_part)
test_fe = engineer.transform(test_raw)

# ---------------------------------------------------------------------------
# 6. Add target to train/val and ensure final column order.
# ---------------------------------------------------------------------------
train_fe["class"] = y_train_part.values
train_fe["target"] = y_train_part.values
val_fe["class"] = y_val_part.values
val_fe["target"] = y_val_part.values

train_fe["original_train_idx"] = train_part.index.to_numpy()
val_fe["original_train_idx"] = val_part.index.to_numpy()
test_fe["test_id"] = np.arange(len(test_fe), dtype=int)

feature_names = [
    c for c in train_fe.columns if c not in {"class", "target", "original_train_idx"}
]

val_fe = val_fe.reindex(columns=train_fe.columns)
test_fe = test_fe.reindex(
    columns=[c for c in train_fe.columns if c != "original_train_idx"] + ["test_id"]
)

# Save processed data.
train_fe.to_parquet(WORKING_DIR / "train_processed.parquet", index=False)
val_fe.to_parquet(WORKING_DIR / "val_processed.parquet", index=False)
test_fe.to_parquet(WORKING_DIR / "test_processed.parquet", index=False)

# Save metadata for downstream steps.
metadata = {
    "seed": SEED,
    "val_fraction": VAL_FRACTION,
    "continuous_cols": continuous_cols,
    "binary_cols": binary_cols,
    "feature_names": feature_names,
    "target_column": "class",
    "target_values": sorted(train_raw["class"].astype(str).unique().tolist()),
    "train_shape": train_fe.shape,
    "val_shape": val_fe.shape,
    "test_shape": test_fe.shape,
    "fit_params": {k: v for k, v in engineer.fit_params.items() if k != "cont_stats"},
    "cont_stats_keys": list(engineer.fit_params.get("cont_stats", {}).keys()),
}

with open(WORKING_DIR / "feature_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# 7. Training and evaluation
# ---------------------------------------------------------------------------
MODEL_PATH = WORKING_DIR / "lgb_best_model.txt"
METADATA_PATH = WORKING_DIR / "training_metadata.json"

MAX_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 50

# Fallbacks are not needed in this merged script, but keep them for robustness.
if (
    "train_fe" not in globals()
    or "val_fe" not in globals()
    or "test_fe" not in globals()
):
    train_fe = pd.read_parquet(WORKING_DIR / "train_processed.parquet")
    val_fe = pd.read_parquet(WORKING_DIR / "val_processed.parquet")
    test_fe = pd.read_parquet(WORKING_DIR / "test_processed.parquet")

if "feature_names" not in globals():
    metadata_path = WORKING_DIR / "feature_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            feat_meta = json.load(f)
        feature_names = list(feat_meta.get("feature_names", []))
    else:
        feature_names = [
            c
            for c in train_fe.columns
            if c not in {"class", "target", "original_train_idx", "test_id"}
        ]

feature_names = [
    c
    for c in feature_names
    if c in train_fe.columns and c in val_fe.columns and c in test_fe.columns
]

if len(feature_names) == 0:
    raise ValueError("No feature columns found for training.")


def _get_target(df: pd.DataFrame) -> pd.Series:
    if "target" in df.columns:
        return df["target"].copy()
    if "class" in df.columns:
        return encode_target(df["class"])
    raise ValueError("Target column not found in processed training table.")


y_train = _get_target(train_fe)
y_val = _get_target(val_fe)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_names].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
    return X.astype(np.float32, copy=False)


X_train = prepare_features(train_fe)
X_val = prepare_features(val_fe)
X_test = prepare_features(test_fe)

valid_train = y_train.notna()
valid_val = y_val.notna()

X_train = X_train.loc[valid_train].reset_index(drop=True)
y_train = y_train.loc[valid_train].astype(int).reset_index(drop=True)

X_val = X_val.loc[valid_val].reset_index(drop=True)
y_val = y_val.loc[valid_val].astype(int).reset_index(drop=True)

if len(X_train) == 0:
    raise ValueError("No valid training rows.")

if len(X_val) == 0:
    raise ValueError("No valid validation rows.")

pos_count = int((y_train == 1).sum())
neg_count = int((y_train == 0).sum())

if pos_count > 0 and neg_count > 0:
    scale_pos_weight = neg_count / pos_count
else:
    scale_pos_weight = 1.0

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "scale_pos_weight": float(scale_pos_weight),
    "random_state": SEED,
    "nthread": -1,
    "verbosity": -1,
    "seed": SEED,
}

dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

callbacks = [
    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
    lgb.log_evaluation(period=-1),
]

model = lgb.train(
    params,
    dtrain,
    num_boost_round=MAX_BOOST_ROUND,
    valid_sets=[dval],
    valid_names=["val"],
    callbacks=callbacks,
)

best_iteration = int(getattr(model, "best_iteration", 0) or 0)
if best_iteration <= 0:
    best_iteration = int(model.num_trees())

model.save_model(str(MODEL_PATH))

loaded_model = lgb.Booster(model_file=str(MODEL_PATH))


def predict_with_saved_model(X: pd.DataFrame) -> np.ndarray:
    probs = loaded_model.predict(X, num_iteration=best_iteration)
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(probs, 0.0, 1.0)


val_pred = predict_with_saved_model(X_val)
test_pred = predict_with_saved_model(X_test)

if len(np.unique(y_val)) >= 2:
    score = float(roc_auc_score(y_val, val_pred))
else:
    score = 0.5

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "class": test_pred,
    }
)

submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

training_metadata = {
    "model_path": str(MODEL_PATH),
    "best_iteration": best_iteration,
    "max_boost_round": MAX_BOOST_ROUND,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "seed": SEED,
    "params": params,
    "num_train_rows": int(len(X_train)),
    "num_val_rows": int(len(X_val)),
    "num_test_rows": int(len(X_test)),
    "pos_count_train": pos_count,
    "neg_count_train": neg_count,
    "scale_pos_weight": float(scale_pos_weight),
    "val_score": score,
    "feature_names": feature_names,
    "validation_metric": "roc_auc",
    "validation_split": "held-out 20% from previous data processing step",
}

with open(METADATA_PATH, "w") as f:
    json.dump(training_metadata, f, indent=2, default=str)

print(
    f"Trained LightGBM: best_iteration={best_iteration}, "
    f"train_rows={len(X_train)}, val_rows={len(X_val)}, "
    f"val_auc={score:.6f}"
)

print(f"Final Validation Score: {score}")