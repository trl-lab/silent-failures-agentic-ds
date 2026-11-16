import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

# -----------------------------
# Paths and constants
# -----------------------------
TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "review_overall"
ID_COLS = {"id", "Id", "ID", "index", "Index"}
RANDOM_STATE = 42
N_SPLITS = 5

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "best",
    "love",
    "loved",
    "liked",
    "like",
    "tasty",
    "smooth",
    "crisp",
    "refreshing",
    "balanced",
    "well",
    "perfect",
    "fantastic",
    "superb",
    "delicious",
    "enjoy",
    "enjoyed",
    "favorite",
    "favourite",
    "highly",
    "recommend",
    "recommended",
    "impressive",
    "solid",
    "clean",
    "bright",
    "complex",
    "interesting",
    "mellow",
    "subtle",
    "pleasant",
    "nice",
    "fine",
    "decent",
    "flavorful",
    "flavourful",
    "aromatic",
    "hoppy",
    "malty",
    "roasty",
    "rich",
    "bold",
    "strong",
    "full",
    "light",
    "medium",
    "dark",
    "wonderful",
    "incredible",
    "outstanding",
    "remarkable",
    "worth",
    "try",
    "tried",
    "again",
    "more",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "worst",
    "hate",
    "hated",
    "dislike",
    "disliked",
    "disgusting",
    "nasty",
    "bland",
    "watery",
    "flat",
    "stale",
    "off",
    "off-putting",
    "offputting",
    "thin",
    "harsh",
    "sharp",
    "medicinal",
    "chemical",
    "plastic",
    "cardboard",
    "vinegar",
    "acetone",
    "nail",
    "polish",
    "cloying",
    "overpowering",
    "underwhelming",
    "unpleasant",
    "poor",
    "weak",
    "watery",
    "harsh",
    "harshness",
    "mediciny",
    "medicinal",
    "chemical",
    "plastic",
    "cardboard",
    "vinegar",
    "acetone",
    "nail",
    "polish",
    "stale",
    "flat",
    "bland",
    "watery",
    "thin",
    "harsh",
}

# -----------------------------
# Load data
# -----------------------------
train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# Drop explicit id columns if present
train = train.drop(columns=[c for c in train.columns if c in ID_COLS], errors="ignore")
test = test.drop(columns=[c for c in test.columns if c in ID_COLS], errors="ignore")

# Keep only rows with target for training
valid = train[TARGET].notna()
train = train.loc[valid].reset_index(drop=True)
y = train[TARGET].astype(float)

feature_cols = [c for c in train.columns if c != TARGET]


# -----------------------------
# Column type inference
# -----------------------------
def _is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    if s.dtype == object or str(s.dtype) == "string":
        sample = s.dropna().astype(str).head(200)
        if len(sample) == 0:
            return False
        num = pd.to_numeric(sample, errors="coerce")
        return num.notna().mean() > 0.8
    return False


def _is_datetime_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object or str(s.dtype) == "string":
        sample = s.dropna().astype(str).head(200)
        if len(sample) == 0:
            return False
        # Avoid treating mostly numeric strings as dates
        num = pd.to_numeric(sample, errors="coerce")
        if num.notna().mean() > 0.8:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except TypeError:
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.8 and sample.str.len().median() < 25
    return False


def _is_text_series(s: pd.Series, col_name: str) -> bool:
    if s.dtype == object or str(s.dtype) == "string":
        sample = s.dropna().astype(str).head(200)
        if len(sample) == 0:
            return False
        avg_len = sample.str.len().mean()
        name = col_name.lower()
        text_keywords = {
            "text",
            "review",
            "description",
            "comment",
            "pros",
            "cons",
            "body",
            "taste",
            "aroma",
            "appearance",
            "notes",
            "summary",
        }
        if any(k in name for k in text_keywords):
            return True
        if avg_len > 50:
            return True
        if s.nunique() / max(len(s), 1) > 0.9 and avg_len > 20:
            return True
    return False


datetime_cols = []
numeric_cols = []
text_cols = []
categorical_cols = []

for col in feature_cols:
    s = train[col]
    if _is_datetime_series(s):
        datetime_cols.append(col)
    elif _is_numeric_series(s):
        numeric_cols.append(col)
    elif _is_text_series(s, col):
        text_cols.append(col)
    else:
        categorical_cols.append(col)

# Decide low-cardinality categoricals for one-hot encoding
low_card_cols = []
high_card_cols = []
for col in categorical_cols:
    nunique = train[col].nunique(dropna=True)
    if nunique <= 20:
        low_card_cols.append(col)
    else:
        high_card_cols.append(col)


# -----------------------------
# Feature construction
# -----------------------------
def _datetime_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    try:
        dt = pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        dt = pd.to_datetime(series, errors="coerce")

    out = pd.DataFrame(index=series.index)
    out[f"{prefix}_year"] = dt.dt.year
    out[f"{prefix}_month"] = dt.dt.month
    out[f"{prefix}_day"] = dt.dt.day
    out[f"{prefix}_dayofweek"] = dt.dt.dayofweek
    out[f"{prefix}_quarter"] = dt.dt.quarter
    out[f"{prefix}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    return out


def _text_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    s = series.fillna("").astype(str)
    out = pd.DataFrame(index=series.index)

    out[f"{prefix}_len"] = s.str.len()
    words = s.str.split()
    out[f"{prefix}_words"] = words.str.len().fillna(0)
    out[f"{prefix}_avg_word_len"] = out[f"{prefix}_len"] / (out[f"{prefix}_words"] + 1)
    out[f"{prefix}_exclaim"] = s.str.count("!")
    out[f"{prefix}_question"] = s.str.count(r"\?")
    out[f"{prefix}_digits"] = s.str.count(r"\d")
    out[f"{prefix}_upper_ratio"] = s.str.count(r"[A-Z]") / (s.str.len() + 1)

    lower = s.str.lower()
    pos = sum(lower.str.count(re.escape(w)) for w in POSITIVE_WORDS)
    neg = sum(lower.str.count(re.escape(w)) for w in NEGATIVE_WORDS)
    out[f"{prefix}_pos"] = pos
    out[f"{prefix}_neg"] = neg
    out[f"{prefix}_sentiment"] = pos - neg

    return out


def build_features(train_df: pd.DataFrame, val_df: pd.DataFrame, fit_df: pd.DataFrame):
    Xtr = pd.DataFrame(index=train_df.index)
    Xva = pd.DataFrame(index=val_df.index)

    # Numeric features
    for col in numeric_cols:
        if col not in train_df.columns or col not in val_df.columns:
            continue
        med = pd.to_numeric(fit_df[col], errors="coerce").median()
        if pd.isna(med):
            med = 0.0
        Xtr[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(med)
        Xva[col] = pd.to_numeric(val_df[col], errors="coerce").fillna(med)

    # Datetime features
    for col in datetime_cols:
        if col not in train_df.columns or col not in val_df.columns:
            continue
        Xtr = pd.concat([Xtr, _datetime_features(train_df[col], col)], axis=1)
        Xva = pd.concat([Xva, _datetime_features(val_df[col], col)], axis=1)

    # Text features
    for col in text_cols:
        if col not in train_df.columns or col not in val_df.columns:
            continue
        Xtr = pd.concat([Xtr, _text_features(train_df[col], col)], axis=1)
        Xva = pd.concat([Xva, _text_features(val_df[col], col)], axis=1)

    # Low-cardinality categoricals: one-hot
    for col in low_card_cols:
        if col not in train_df.columns or col not in val_df.columns:
            continue
        cats = sorted(fit_df[col].astype(str).unique())
        for cat in cats:
            Xtr[f"{col}_{cat}"] = (train_df[col].astype(str) == cat).astype(int)
            Xva[f"{col}_{cat}"] = (val_df[col].astype(str) == cat).astype(int)

    # High-cardinality categoricals: frequency encoding
    for col in high_card_cols:
        if col not in train_df.columns or col not in val_df.columns:
            continue
        freq = fit_df[col].astype(str).value_counts(normalize=True)
        Xtr[col] = train_df[col].astype(str).map(freq).fillna(0.0)
        Xva[col] = val_df[col].astype(str).map(freq).fillna(0.0)

    # Final cleanup
    Xtr = Xtr.fillna(0.0).astype(float)
    Xva = Xva.fillna(0.0).astype(float)

    return Xtr, Xva


# -----------------------------
# Cross-validation
# -----------------------------
# Optional group-aware CV if a beer identifier is present
group_col = None
for c in feature_cols:
    cl = c.lower()
    if cl in {"beer_id", "beerid", "beer", "beer_name", "beername", "beer_name_id"}:
        if train[c].notna().mean() > 0.5 and train[c].nunique(dropna=True) > 1:
            group_col = c
            break

use_group = group_col is not None
if use_group:
    groups = train[group_col].astype(str).fillna("unknown")
    cv = GroupKFold(n_splits=N_SPLITS)
else:
    groups = None
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros(len(train), dtype=float)
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(train, y, groups=groups)):
    tr_df = train.iloc[tr_idx]
    va_df = train.iloc[va_idx]
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    X_tr, X_va = build_features(tr_df, va_df, fit_df=tr_df)

    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        best_iter = model.n_estimators - 1
    best_iterations.append(int(best_iter))

    if (
        hasattr(model, "predict")
        and "iteration_range" in model.predict.__code__.co_varnames
    ):
        pred = model.predict(X_va, iteration_range=(0, best_iter + 1))
    else:
        pred = model.predict(X_va)

    oof_preds[va_idx] = pred

oof_rmse = root_mean_squared_error(y, oof_preds)
print(f"5-fold OOF RMSE: {oof_rmse:.6f}")

# -----------------------------
# Final model and submission
# -----------------------------
X_full, X_test = build_features(train, test, fit_df=train)

avg_best_iter = int(np.mean(best_iterations)) if best_iterations else 500
avg_best_iter = max(1, avg_best_iter)

final_model = xgb.XGBRegressor(
    n_estimators=avg_best_iter,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    objective="reg:squarederror",
    eval_metric="rmse",
)

final_model.fit(X_full, y, verbose=False)
test_preds = final_model.predict(X_test)

# Beer ratings are typically bounded between 0 and 5
test_preds = np.clip(test_preds, 0.0, 5.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "review_overall": test_preds,
    }
)

submission_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"Saved submission to {submission_path}")
