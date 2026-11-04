import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
try:
    import xgboost as xgb
except ImportError:
    xgb = None
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# -----------------------------
# Paths and constants
# -----------------------------
INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET = "marketing_status"
ID_COL = "id"
RANDOM_STATE = 42
VAL_FRACTION = 0.2

# -----------------------------
# Load raw data
# -----------------------------
train_raw = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_raw = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

if ID_COL not in train_raw.columns:
    train_raw[ID_COL] = np.arange(len(train_raw))
if ID_COL not in test_raw.columns:
    test_raw[ID_COL] = np.arange(len(test_raw))

if TARGET not in train_raw.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train data.")

train_raw = train_raw.dropna(subset=[TARGET]).reset_index(drop=True)
test_raw = test_raw.reset_index(drop=True)


# -----------------------------
# Target encoding for binary ROC AUC
# -----------------------------
def encode_binary_target(y_raw: pd.Series) -> tuple[pd.Series, dict]:
    y_raw = y_raw.astype(str)
    unique = sorted(y_raw.unique())

    if set(unique).issubset({"0", "1"}):
        mapping = {"0": 0, "1": 1}
        positive_class = "1"
    elif set(unique).issubset({"False", "True"}):
        mapping = {"False": 0, "True": 1}
        positive_class = "True"
    elif len(unique) == 2:
        mapping = {unique[0]: 0, unique[1]: 1}
        positive_class = unique[1]
    else:
        mapping = {c: i for i, c in enumerate(unique)}
        positive_class = unique[-1]

    y = y_raw.map(mapping)
    if y.isna().any():
        y = y.fillna(0)

    meta = {
        "target_column": TARGET,
        "target_mapping": mapping,
        "positive_class": positive_class,
        "unique_target_values": unique,
        "target_class_counts": y.value_counts().to_dict(),
    }
    return y.astype(int), meta


y_all, target_meta = encode_binary_target(train_raw[TARGET])

# -----------------------------
# Split FIRST: train -> train/val
# -----------------------------
if y_all.nunique() > 1 and y_all.value_counts().min() >= 5:
    train_split, val_split, y_train, y_val = train_test_split(
        train_raw,
        y_all,
        test_size=VAL_FRACTION,
        stratify=y_all,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
else:
    train_split, val_split, y_train, y_val = train_test_split(
        train_raw,
        y_all,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)

# -----------------------------
# Feature column discovery
# -----------------------------
exclude_cols = {ID_COL, TARGET}
feature_cols = [c for c in train_split.columns if c not in exclude_cols]


def parse_datetime_safe(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)
    except TypeError:
        return pd.to_datetime(s, errors="coerce", dayfirst=False)


def is_datetime_series(s: pd.Series, sample_size: int = 1000) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (s.dtype == object or str(s.dtype) == "string"):
        return False
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.head(sample_size)
    parsed = parse_datetime_safe(sample)
    return parsed.notna().mean() > 0.8


def is_text_series(s: pd.Series) -> bool:
    if not (s.dtype == object or str(s.dtype) == "string"):
        return False
    non_null = s.dropna().astype(str)
    if len(non_null) == 0:
        return False
    avg_len = non_null.str.len().mean()
    max_len = non_null.str.len().max()
    return avg_len > 30 or max_len > 100


numeric_cols = []
datetime_cols = []
text_cols = []
categorical_cols = []

for col in feature_cols:
    s = train_split[col]
    if pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(col)
    elif pd.api.types.is_datetime64_any_dtype(s):
        datetime_cols.append(col)
    elif s.dtype == object or str(s.dtype) == "string":
        if is_datetime_series(s):
            datetime_cols.append(col)
        elif is_text_series(s):
            text_cols.append(col)
        else:
            categorical_cols.append(col)
    else:
        categorical_cols.append(col)

# -----------------------------
# Fit stateful transformations ONLY on train_split
# -----------------------------
numeric_medians = {}
numeric_log1p = {}
numeric_scalers = {}

for col in numeric_cols:
    s = pd.to_numeric(train_split[col], errors="coerce")
    numeric_medians[col] = float(s.median()) if s.notna().any() else 0.0

    valid = s.dropna()
    use_log = False
    if len(valid) > 10 and (valid >= 0).all():
        skew = valid.skew()
        if np.isfinite(skew) and skew > 2.0:
            use_log = True
    numeric_log1p[col] = bool(use_log)

    s_imp = s.fillna(numeric_medians[col])
    if use_log:
        s_fit = np.log1p(s_imp.clip(lower=0))
    else:
        s_fit = s_imp.astype(float)

    scaler = RobustScaler()
    scaler.fit(s_fit.values.reshape(-1, 1))
    numeric_scalers[col] = scaler

# Datetime reference from train only
datetime_reference = None
if datetime_cols:
    all_dates = []
    for col in datetime_cols:
        s = parse_datetime_safe(train_split[col])
        all_dates.append(s)
    all_dates = pd.concat(all_dates, axis=1)
    valid_dates = all_dates.stack()
    if len(valid_dates) > 0:
        datetime_reference = valid_dates.max()

# Categorical maps fitted on train only
cat_maps = {}
freq_maps = {}
target_enc_maps = {}
global_target_means = {}
ohe_categories = {}

for col in categorical_cols:
    train_s = train_split[col].fillna("__MISSING__").astype(str)

    categories = sorted(train_s.unique().tolist())
    cat_maps[col] = {c: i for i, c in enumerate(categories)}

    freq = train_s.value_counts(normalize=True)
    freq_maps[col] = freq.to_dict()

    global_mean = float(y_train.mean())
    tmp = pd.DataFrame({"cat": train_s.values, "y": y_train.values})
    stats = tmp.groupby("cat")["y"].agg(["mean", "count"])
    smoothing = 10.0
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    target_enc_maps[col] = smoothed.to_dict()
    global_target_means[col] = global_mean

    if len(categories) <= 20:
        ohe_categories[col] = categories

# Text TF-IDF + SVD fitted on train only
text_columns = text_cols
vectorizer = None
svd = None
text_svd_components = 0
char_vectorizer = None
char_svd = None
text_char_svd_components = 0


def make_combined_text(df: pd.DataFrame) -> pd.Series:
    if not text_columns:
        return pd.Series([""] * len(df), index=df.index)
    parts = []
    for col in text_columns:
        if col in df.columns:
            values = df[col].fillna("").astype(str).values
            parts.append(pd.Series([f"{col}: {v}" for v in values], index=df.index))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    return pd.concat(parts, axis=1).agg(" ".join, axis=1)


if text_columns:
    train_texts = make_combined_text(train_split)
    if train_texts.str.len().mean() > 1:
        vectorizer = TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        X_text_train = vectorizer.fit_transform(train_texts)

        if X_text_train.shape[1] > 0:
            max_components = min(
                64,
                max(1, X_text_train.shape[1] - 1),
                max(1, X_text_train.shape[0] - 1),
            )
            svd = TruncatedSVD(n_components=max_components, random_state=RANDOM_STATE)
            X_text_train_svd = svd.fit_transform(X_text_train)
            text_svd_components = X_text_train_svd.shape[1]
        else:
            vectorizer = None
            svd = None
            text_svd_components = 0

        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=4000,
            min_df=2,
            sublinear_tf=True,
            lowercase=True,
            strip_accents="unicode",
        )
        X_char_train = char_vectorizer.fit_transform(train_texts)
        if X_char_train.shape[1] > 0:
            char_components = min(
                32,
                max(1, X_char_train.shape[1] - 1),
                max(1, X_char_train.shape[0] - 1),
            )
            char_svd = TruncatedSVD(n_components=char_components, random_state=RANDOM_STATE)
            X_char_train_svd = char_svd.fit_transform(X_char_train)
            text_char_svd_components = X_char_train_svd.shape[1]
        else:
            char_vectorizer = None
            char_svd = None
            text_char_svd_components = 0


# -----------------------------
# Feature builder
# -----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    # Numeric features
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        missing = s.isna().astype(int)
        s_imp = s.fillna(numeric_medians[col])
        if numeric_log1p[col]:
            s_imp = np.log1p(s_imp.clip(lower=0))
        scaled = numeric_scalers[col].transform(s_imp.values.reshape(-1, 1)).ravel()
        feats[col] = scaled
        feats[f"{col}__missing"] = missing

    # Datetime features
    for col in datetime_cols:
        s = parse_datetime_safe(df[col])
        missing = s.isna().astype(int)
        feats[f"{col}__year"] = s.dt.year
        feats[f"{col}__month"] = s.dt.month
        feats[f"{col}__day"] = s.dt.day
        feats[f"{col}__dayofweek"] = s.dt.dayofweek
        feats[f"{col}__dayofyear"] = s.dt.dayofyear
        feats[f"{col}__quarter"] = s.dt.quarter
        feats[f"{col}__is_weekend"] = (s.dt.dayofweek >= 5).astype(int)
        if datetime_reference is not None:
            feats[f"{col}__age_days"] = (datetime_reference - s).dt.days
        feats[f"{col}__missing"] = missing

    # Categorical features
    for col in categorical_cols:
        if col not in df.columns:
            s_raw = pd.Series([np.nan] * len(df), index=df.index)
        else:
            s_raw = df[col]
        missing = s_raw.isna().astype(int)
        s_str = s_raw.fillna("__MISSING__").astype(str)

        feats[f"{col}__code"] = s_str.map(cat_maps[col]).fillna(-1).astype(int)
        feats[f"{col}__freq"] = s_str.map(freq_maps[col]).fillna(0.0).astype(float)
        feats[f"{col}__target_enc"] = (
            s_str.map(target_enc_maps[col])
            .fillna(global_target_means[col])
            .astype(float)
        )

        if col in ohe_categories:
            for idx, cat in enumerate(ohe_categories[col]):
                safe_cat = re.sub(r"[^A-Za-z0-9_]+", "_", str(cat))[:36]
                feats[f"{col}__onehot_{idx}_{safe_cat}"] = (s_str == cat).astype(int)

    # Text lexical features
    for col in text_cols:
        if col not in df.columns:
            s = pd.Series([""] * len(df), index=df.index)
            missing = pd.Series([1] * len(df), index=df.index, dtype=int)
        else:
            missing = df[col].isna().astype(int)
            s = df[col].fillna("").astype(str)

        words = s.str.split()
        word_counts = words.str.len().fillna(0).astype(int)
        char_counts = s.str.count(r"\w")
        digit_counts = s.str.count(r"\d")
        upper_counts = s.str.count(r"[A-Z]")
        punct_counts = s.str.count(r"[^\w\s]")
        sentence_counts = s.str.count(r"[.!?]+")
        line_counts = s.str.count(r"\n")

        feats[f"{col}__len"] = s.str.len().astype(float)
        feats[f"{col}__words"] = word_counts.astype(float)
        feats[f"{col}__chars"] = char_counts.astype(float)
        feats[f"{col}__digits"] = digit_counts.astype(float)
        feats[f"{col}__upper"] = upper_counts.astype(float)
        feats[f"{col}__punct"] = punct_counts.astype(float)
        feats[f"{col}__avg_word_len"] = (char_counts / (word_counts + 1)).astype(float)
        feats[f"{col}__sentences"] = sentence_counts.astype(float)
        feats[f"{col}__lines"] = line_counts.astype(float)
        feats[f"{col}__has_url"] = s.str.contains(
            r"(http|www)", regex=True, na=False
        ).astype(int)
        feats[f"{col}__has_email"] = s.str.contains("@", regex=False, na=False).astype(
            int
        )
        feats[f"{col}__has_number"] = s.str.contains(
            r"\d", regex=True, na=False
        ).astype(int)
        feats[f"{col}__missing"] = missing.astype(int)

    # Combined text features
    if text_columns:
        combined = make_combined_text(df)
        combined_words = combined.str.split().str.len().fillna(0).astype(int)
        feats["text__total_len"] = combined.str.len().astype(float)
        feats["text__total_words"] = combined_words.astype(float)
        feats["text__total_digits"] = combined.str.count(r"\d").astype(float)
        feats["text__total_upper"] = combined.str.count(r"[A-Z]").astype(float)

        if vectorizer is not None and svd is not None:
            X_text = vectorizer.transform(combined)
            X_svd = svd.transform(X_text)
            for i in range(X_svd.shape[1]):
                feats[f"text_svd_{i}"] = X_svd[:, i]

        if char_vectorizer is not None and char_svd is not None:
            X_char = char_vectorizer.transform(combined)
            X_char_svd = char_svd.transform(X_char)
            for i in range(X_char_svd.shape[1]):
                feats[f"text_char_svd_{i}"] = X_char_svd[:, i]

    # Row-level aggregates
    if numeric_cols:
        num_feat_names = [c for c in numeric_cols if c in feats.columns]
        if num_feat_names:
            feats["numeric__mean"] = feats[num_feat_names].mean(axis=1)
            feats["numeric__std"] = feats[num_feat_names].std(axis=1).fillna(0.0)
            feats["numeric__min"] = feats[num_feat_names].min(axis=1)
            feats["numeric__max"] = feats[num_feat_names].max(axis=1)
            missing_names = [f"{c}__missing" for c in num_feat_names]
            feats["numeric__missing_count"] = feats[missing_names].sum(axis=1)

    if categorical_cols:
        cat_missing_names = [
            f"{c}__missing"
            for c in categorical_cols
            if f"{c}__missing" in feats.columns
        ]
        if cat_missing_names:
            feats["categorical__missing_count"] = feats[cat_missing_names].sum(axis=1)

    if text_cols:
        text_missing_names = [
            f"{c}__missing" for c in text_cols if f"{c}__missing" in feats.columns
        ]
        if text_missing_names:
            feats["text__missing_count"] = feats[text_missing_names].sum(axis=1)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    return feats


# -----------------------------
# Build matrices
# -----------------------------
train_feats = build_features(train_split)
val_feats = build_features(val_split)
test_feats = build_features(test_raw)

feature_medians = train_feats.median(numeric_only=True).fillna(0.0)

train_feats = train_feats.fillna(feature_medians)
val_feats = val_feats.fillna(feature_medians)
test_feats = test_feats.fillna(feature_medians)

feature_names = [c for c in train_feats.columns]
train_feats = train_feats[feature_names]
val_feats = val_feats.reindex(columns=feature_names)
test_feats = test_feats.reindex(columns=feature_names)

variances = train_feats.var()
non_constant_features = variances[variances > 1e-12].index.tolist()
if len(non_constant_features) > 0:
    train_feats = train_feats[non_constant_features]
    val_feats = val_feats[non_constant_features]
    test_feats = test_feats[non_constant_features]
    feature_names = non_constant_features

train_feats = train_feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
val_feats = val_feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
test_feats = test_feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

if not feature_names:
    raise ValueError("No usable features were created.")

# -----------------------------
# Model design
# -----------------------------
pos_count = int((y_train == 1).sum())
neg_count = int((y_train == 0).sum())
total_count = int(len(y_train))

if pos_count > 0 and neg_count > 0:
    scale_pos_weight = float(neg_count) / float(pos_count)
else:
    scale_pos_weight = 1.0
scale_pos_weight = float(np.clip(scale_pos_weight, 0.1, 10.0))

base_lgbm_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=4000,
    learning_rate=0.02,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=25,
    min_child_weight=1e-3,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    importance_type="gain",
    verbosity=-1,
)

regularized_lgbm_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=4000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    min_child_weight=1e-3,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    importance_type="gain",
    verbosity=-1,
)

xgb_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eta": 0.02,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "alpha": 0.1,
    "lambda": 1.0,
    "scale_pos_weight": scale_pos_weight,
    "seed": RANDOM_STATE,
    "eval_metric": "auc",
    "nthread": -1,
}

model_params = base_lgbm_params
model = None


def rank_average_proba(proba_list):
    rank_list = []
    for p in proba_list:
        p = np.asarray(p, dtype=float).ravel()
        p = np.where(np.isfinite(p), p, np.nan)
        rank_list.append(pd.Series(p).rank(pct=True).fillna(0.5).values)
    if not rank_list:
        return np.array([], dtype=float)
    return np.clip(np.mean(rank_list, axis=0), 0.0, 1.0)


# -----------------------------
# Training and validation
# -----------------------------
def prepare_X(df: pd.DataFrame) -> np.ndarray:
    X = df.reindex(columns=feature_names)
    X = X.astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.values


X_train = prepare_X(train_feats)
X_val = prepare_X(val_feats)
X_test = prepare_X(test_feats)

early_stopping_rounds = 100

if y_train.nunique() <= 1:
    raise ValueError(
        "Training target has only one class; cannot train a binary classifier."
    )

can_early_stop = len(y_val) > 0 and y_val.nunique() > 1

lgbm_base_model = lgb.LGBMClassifier(**base_lgbm_params)
lgbm_reg_model = lgb.LGBMClassifier(**regularized_lgbm_params)

lgbm_val_probas = []
lgbm_best_iterations = []

for model_obj in (lgbm_base_model, lgbm_reg_model):
    if can_early_stop:
        model_obj.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    else:
        model_obj.fit(X_train, y_train)

    best_iter = getattr(model_obj, "best_iteration_", None)
    if best_iter is not None:
        best_iter = int(best_iter)
        if best_iter <= 0:
            best_iter = None
    lgbm_best_iterations.append(best_iter)

    if best_iter is not None:
        val_p = model_obj.predict_proba(X_val, num_iteration=best_iter)[:, 1]
    else:
        val_p = model_obj.predict_proba(X_val)[:, 1]
    lgbm_val_probas.append(np.clip(np.asarray(val_p, dtype=float), 0.0, 1.0))

xgb_best_iteration = None
xgb_val_proba = np.full(len(y_val), 0.5, dtype=float)

if xgb is not None:
    dtrain = xgb.DMatrix(X_train, label=y_train.values, feature_names=feature_names)
    if can_early_stop:
        dval = xgb.DMatrix(X_val, label=y_val.values, feature_names=feature_names)
        xgb_booster = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=4000,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            maximize=True,
            verbose_eval=False,
        )
        xgb_best_iteration = getattr(xgb_booster, "best_iteration", None)
        if xgb_best_iteration is None:
            xgb_best_iteration = 4000 - 1
        xgb_best_iteration = int(xgb_best_iteration)
        dval_pred = xgb.DMatrix(X_val, feature_names=feature_names)
        xgb_val_proba = xgb_booster.predict(
            dval_pred, iteration_range=(0, xgb_best_iteration + 1)
        )
    else:
        xgb_booster = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=4000,
            verbose_eval=False,
        )
        xgb_best_iteration = 4000 - 1
        dval_pred = xgb.DMatrix(X_val, feature_names=feature_names)
        xgb_val_proba = xgb_booster.predict(
            dval_pred, iteration_range=(0, xgb_best_iteration + 1)
        )
    xgb_val_proba = np.clip(np.asarray(xgb_val_proba, dtype=float).ravel(), 0.0, 1.0)
else:
    xgb_best_iteration = 4000 - 1

rank_averaged_val_proba = rank_average_proba(lgbm_val_probas)
if xgb is not None:
    rank_averaged_val_proba = rank_average_proba(lgbm_val_probas + [xgb_val_proba])

if y_val.nunique() > 1 and len(y_val) > 0:
    score = float(roc_auc_score(y_val, rank_averaged_val_proba))
else:
    score = 0.5

# -----------------------------
# Final refit on all labeled data, using the same architecture and best iteration
# -----------------------------
full_train_feats = build_features(train_raw)
full_train_feats = full_train_feats.reindex(columns=feature_names)
full_train_feats = full_train_feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_full = full_train_feats.values

if y_all.nunique() > 1:
    X_test_final = prepare_X(test_feats)

    final_base_n = (
        int(lgbm_best_iterations[0])
        if lgbm_best_iterations[0] is not None
        else int(base_lgbm_params["n_estimators"])
    )
    final_reg_n = (
        int(lgbm_best_iterations[1])
        if lgbm_best_iterations[1] is not None
        else int(regularized_lgbm_params["n_estimators"])
    )

    final_base_params = dict(base_lgbm_params)
    final_base_params["n_estimators"] = final_base_n
    final_reg_params = dict(regularized_lgbm_params)
    final_reg_params["n_estimators"] = final_reg_n

    final_lgbm_base = lgb.LGBMClassifier(**final_base_params)
    final_lgbm_base.fit(X_full, y_all.values)
    final_lgbm_reg = lgb.LGBMClassifier(**final_reg_params)
    final_lgbm_reg.fit(X_full, y_all.values)

    base_test_proba = final_lgbm_base.predict_proba(X_test_final)[:, 1]
    reg_test_proba = final_lgbm_reg.predict_proba(X_test_final)[:, 1]

    test_proba_list = [
        np.clip(np.asarray(base_test_proba, dtype=float), 0.0, 1.0),
        np.clip(np.asarray(reg_test_proba, dtype=float), 0.0, 1.0),
    ]

    if xgb is not None:
        dtrain_full = xgb.DMatrix(X_full, label=y_all.values, feature_names=feature_names)
        final_xgb_rounds = (
            int(xgb_best_iteration) + 1
            if xgb_best_iteration is not None
            else 4000
        )
        final_xgb_booster = xgb.train(
            xgb_params,
            dtrain_full,
            num_boost_round=final_xgb_rounds,
            verbose_eval=False,
        )
        dtest = xgb.DMatrix(X_test_final, feature_names=feature_names)
        xgb_test_proba = final_xgb_booster.predict(
            dtest, iteration_range=(0, final_xgb_rounds)
        )
        test_proba_list.append(
            np.clip(np.asarray(xgb_test_proba, dtype=float).ravel(), 0.0, 1.0)
        )

    test_proba = rank_average_proba(test_proba_list)
else:
    test_proba = np.full(len(test_raw), float(y_all.mean()), dtype=float)

# -----------------------------
# Submission
# -----------------------------
submission = pd.DataFrame(
    {
        "id": np.arange(len(test_raw), dtype=int),
        "marketing_status": test_proba,
    }
)
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(f"Final Validation Score: {score}")
