import json
import os
import re
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Setup Directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 2. Load Datasets
print("Loading datasets from ./input/...")
train_df = pd.read_parquet("./input/train.parquet").reset_index(drop=True)
test_df = pd.read_parquet("./input/test.parquet").reset_index(drop=True)

target_col = "awardFor"
if target_col not in train_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in train.parquet")

# Encode Target Labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[target_col].astype(str))
target_classes = list(label_encoder.classes_)
joblib.dump(label_encoder, "./working/label_encoder.pkl")

print(f"Dataset shapes: Train={train_df.shape}, Test={test_df.shape}")
print(f"Target classes ({len(target_classes)}): {target_classes}")

# 3. Geospatial & Domain Constants
CULINARY_CAPITALS = {
    "paris": (48.8566, 2.3522),
    "tokyo": (35.6762, 139.6503),
    "new_york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "san_francisco": (37.7749, -122.4194),
    "kyoto": (35.0116, 135.7681),
    "rome": (41.9028, 12.4964),
    "hong_kong": (22.3193, 114.1694),
    "singapore": (1.3521, 103.8198),
    "madrid": (40.4168, -3.7038),
}


def haversine_np(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth in kilometers."""
    r = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return r * c


def extract_price_level(val):
    """Extract numeric price tier from symbols or strings."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    if any(sym in s for sym in ["$", "€", "£", "¥", "₩"]):
        return float(len(s))
    match = re.search(r"\d+(\.\d+)?", s)
    if match:
        return float(match.group(0))
    return np.nan


# 4. Feature Extraction
def build_base_features(df):
    f_df = pd.DataFrame(index=df.index)
    text_fields = []

    price_col = next((c for c in df.columns if "price" in c.lower()), None)
    if price_col:
        f_df["price_tier"] = df[price_col].apply(extract_price_level)
    else:
        f_df["price_tier"] = np.nan

    lat_col = next(
        (c for c in df.columns if c.lower() in ["latitude", "lat", "lat_deg"]), None
    )
    lon_col = next(
        (
            c
            for c in df.columns
            if c.lower() in ["longitude", "lon", "lng", "long", "lon_deg"]
        ),
        None,
    )

    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")

        f_df["lat"] = lat
        f_df["lon"] = lon
        f_df["geo_is_missing"] = (lat.isna() | lon.isna()).astype(int)

        lat_rad = np.radians(lat.fillna(0.0))
        lon_rad = np.radians(lon.fillna(0.0))
        f_df["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
        f_df["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
        f_df["geo_z"] = np.sin(lat_rad)

        for city_name, (clat, clon) in CULINARY_CAPITALS.items():
            f_df[f"dist_to_{city_name}"] = haversine_np(
                lat.fillna(clat), lon.fillna(clon), clat, clon
            )

        f_df["min_dist_to_capital"] = f_df[
            [f"dist_to_{c}" for c in CULINARY_CAPITALS]
        ].min(axis=1)
    else:
        f_df["lat"] = 0.0
        f_df["lon"] = 0.0
        f_df["geo_is_missing"] = 1
        f_df["geo_x"] = 0.0
        f_df["geo_y"] = 0.0
        f_df["geo_z"] = 0.0
        f_df["min_dist_to_capital"] = 0.0

    for col in df.columns:
        if col == target_col or col.lower() in ["id", "index", "restaurant_id"]:
            continue
        c_series = df[col].astype(str).fillna("")
        n_unique = df[col].nunique(dropna=True)
        is_text = (
            c_series.apply(len).mean() > 40
            or "description" in col.lower()
            or "review" in col.lower()
            or "detail" in col.lower()
        )

        if is_text:
            text_fields.append(c_series)
            f_df[f"{col}_char_len"] = c_series.apply(len)
            f_df[f"{col}_word_count"] = c_series.apply(lambda x: len(x.split()))
            f_df[f"{col}_num_digits"] = c_series.apply(
                lambda x: sum(c.isdigit() for c in x)
            )
            f_df[f"{col}_num_caps"] = c_series.apply(
                lambda x: sum(c.isupper() for c in x)
            )
            f_df[f"{col}_cap_ratio"] = f_df[f"{col}_num_caps"] / (
                f_df[f"{col}_char_len"] + 1
            )
        elif "cuisine" in col.lower() or "style" in col.lower():
            text_fields.append(c_series)
            f_df[f"{col}_item_count"] = c_series.apply(
                lambda x: len(re.split(r"[,;/|]", x)) if x else 0
            )
            f_df[f"{col}_char_len"] = c_series.apply(len)
        elif n_unique < 5000:
            f_df[f"{col}_is_missing"] = df[col].isna().astype(int)

    if text_fields:
        combined_text = text_fields[0].copy()
        for tf in text_fields[1:]:
            combined_text = combined_text + " " + tf
    else:
        combined_text = pd.Series([""] * len(df), index=df.index)

    # Extract domain-specific culinary & prestige keyword features
    culinary_keywords = [
        "tasting menu", "wine pairing", "sommelier", "bistro", "terroir",
        "caviar", "truffle", "omakase", "signature", "farm-to-table",
        "farm to table", "cellar", "fine dining", "casual", "street food",
        "budget", "affordable", "value", "luxury", "exceptional", "artisan",
        "craft", "seasonal", "contemporary", "traditional", "cocktail",
        "seafood", "wagyu", "panoramic", "garden", "historic", "intimate",
        "cozy", "modern", "heritage", "chef", "prix fixe", "kaiseki"
    ]
    lower_text = combined_text.str.lower()
    for kw in culinary_keywords:
        kw_col = f"kw_{kw.replace(' ', '_').replace('-', '_')}"
        f_df[kw_col] = lower_text.str.count(kw)

    return f_df, combined_text


print("Extracting base domain features...")
train_features_base, train_text = build_base_features(train_df)
test_features_base, test_text = build_base_features(test_df)

price_median = train_features_base["price_tier"].median()
if pd.isna(price_median):
    price_median = 1.0
train_features_base["price_tier"] = train_features_base["price_tier"].fillna(
    price_median
)
test_features_base["price_tier"] = test_features_base["price_tier"].fillna(price_median)

lat_median = train_features_base["lat"].median()
lon_median = train_features_base["lon"].median()
train_features_base["lat"] = train_features_base["lat"].fillna(lat_median)
test_features_base["lat"] = test_features_base["lat"].fillna(lat_median)
train_features_base["lon"] = train_features_base["lon"].fillna(lon_median)
test_features_base["lon"] = test_features_base["lon"].fillna(lon_median)

# 5. TF-IDF, SVD & Out-of-Fold Text Probability Meta-Features
print("Vectorizing semantic text and generating out-of-fold text meta-probabilities...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=2500,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    strip_accents="unicode",
)
train_tfidf = tfidf_vectorizer.fit_transform(train_text.fillna(""))
test_tfidf = tfidf_vectorizer.transform(test_text.fillna(""))

n_svd_components = min(30, max(1, train_tfidf.shape[1] - 1))
svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

for i in range(n_svd_components):
    train_features_base[f"svd_text_{i}"] = train_svd[:, i]
    test_features_base[f"svd_text_{i}"] = test_svd[:, i]

# High-dimensional word & char TF-IDF for linear text meta-classifier
word_vec = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode",
)
char_vec = TfidfVectorizer(
    max_features=15000,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    sublinear_tf=True,
    strip_accents="unicode",
)
train_word_tfidf = word_vec.fit_transform(train_text.fillna(""))
test_word_tfidf = word_vec.transform(test_text.fillna(""))
train_char_tfidf = char_vec.fit_transform(train_text.fillna(""))
test_char_tfidf = char_vec.transform(test_text.fillna(""))

train_text_sparse = sparse.hstack([train_word_tfidf, train_char_tfidf]).tocsr()
test_text_sparse = sparse.hstack([test_word_tfidf, test_char_tfidf]).tocsr()

oof_text_probs = np.zeros((len(train_df), len(target_classes)), dtype=np.float64)
test_text_probs = np.zeros((len(test_df), len(target_classes)), dtype=np.float64)

text_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for t_tr_idx, t_va_idx in text_skf.split(train_text_sparse, y_train):
    lr_text = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    lr_text.fit(train_text_sparse[t_tr_idx], y_train[t_tr_idx])
    oof_text_probs[t_va_idx] = lr_text.predict_proba(train_text_sparse[t_va_idx])
    test_text_probs += lr_text.predict_proba(test_text_sparse) / 5.0

for c_idx in range(len(target_classes)):
    train_features_base[f"text_meta_prob_class_{c_idx}"] = oof_text_probs[:, c_idx]
    test_features_base[f"text_meta_prob_class_{c_idx}"] = test_text_probs[:, c_idx]

# 6. Frequency Encodings, Group Aggregations & Out-Of-Fold Bayesian Target Encoding
print("Extracting out-of-fold target encodings and multi-level price aggregations...")
categorical_cols = [
    c for c in train_df.columns if train_df[c].dtype == "object" and c != target_col
]
for col in categorical_cols:
    freq_map = train_df[col].astype(str).value_counts(normalize=True).to_dict()
    train_features_base[f"{col}_freq"] = (
        train_df[col].astype(str).map(freq_map).fillna(0.0)
    )
    test_features_base[f"{col}_freq"] = (
        test_df[col].astype(str).map(freq_map).fillna(0.0)
    )

# Out-of-fold Bayesian smoothed target encoding for categorical columns
te_cols = [
    c for c in train_df.columns
    if train_df[c].dtype == "object"
    and c != target_col
    and c.lower() not in ["id", "index", "restaurant_id"]
    and train_df[c].nunique() < 2000
]

smoothing_weight = 10.0
te_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for c_idx in range(len(target_classes)):
    y_binary = (y_train == c_idx).astype(float)
    global_mean = float(y_binary.mean())

    for col in te_cols:
        col_series_tr = train_df[col].astype(str)
        col_series_te = test_df[col].astype(str)

        oof_te = np.full(len(train_df), global_mean, dtype=np.float64)
        for t_tr_idx, t_va_idx in te_skf.split(train_df, y_train):
            sub_tr = train_df.iloc[t_tr_idx]
            sub_y = y_binary[t_tr_idx]

            stats = (
                pd.DataFrame({"cat": sub_tr[col].astype(str), "target": sub_y})
                .groupby("cat")["target"]
                .agg(["count", "mean"])
            )
            smoothed_map = (
                (stats["count"] * stats["mean"] + smoothing_weight * global_mean)
                / (stats["count"] + smoothing_weight)
            ).to_dict()

            oof_te[t_va_idx] = (
                col_series_tr.iloc[t_va_idx].map(smoothed_map).fillna(global_mean).values
            )

        full_stats = (
            pd.DataFrame({"cat": col_series_tr, "target": y_binary})
            .groupby("cat")["target"]
            .agg(["count", "mean"])
        )
        full_smoothed_map = (
            (full_stats["count"] * full_stats["mean"] + smoothing_weight * global_mean)
            / (full_stats["count"] + smoothing_weight)
        ).to_dict()
        test_te = col_series_te.map(full_smoothed_map).fillna(global_mean).values

        train_features_base[f"{col}_te_class_{c_idx}"] = oof_te
        test_features_base[f"{col}_te_class_{c_idx}"] = test_te

# Multi-level price relative aggregations across cuisine and geo fields
group_cols = [
    c for c in train_df.columns
    if any(k in c.lower() for k in ["cuisine", "city", "region", "country", "location", "area", "neighborhood"])
    and train_df[c].dtype == "object"
]

for col in group_cols:
    price_stats = (
        train_features_base.groupby(train_df[col].astype(str))["price_tier"]
        .agg(["mean", "std"])
        .to_dict()
    )
    mean_map = price_stats["mean"]
    tr_cat = train_df[col].astype(str)
    te_cat = test_df[col].astype(str)

    train_features_base[f"{col}_price_mean"] = tr_cat.map(mean_map).fillna(price_median)
    test_features_base[f"{col}_price_mean"] = te_cat.map(mean_map).fillna(price_median)

    train_features_base[f"price_diff_from_{col}_mean"] = (
        train_features_base["price_tier"] - train_features_base[f"{col}_price_mean"]
    )
    test_features_base[f"price_diff_from_{col}_mean"] = (
        test_features_base["price_tier"] - test_features_base[f"{col}_price_mean"]
    )

train_features = train_features_base.fillna(0.0)
test_features = test_features_base.fillna(0.0)

common_cols = [c for c in train_features.columns if c in test_features.columns]
train_features = train_features[common_cols]
test_features = test_features[common_cols]

print(f"Engineered Feature Matrix: {len(common_cols)} total features.")

# 7. Model Architectures & Configurations
num_classes = len(np.unique(y_train))
num_samples, num_features = train_features.shape
num_test = len(test_features)

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 2,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

cb_params = {
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "random_seed": 42,
    "verbose": False,
    "early_stopping_rounds": 50,
}

# 8. 5-Fold Stratified Cross-Validation Training (Tri-Model Architecture)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb_probs = np.zeros((num_samples, num_classes), dtype=np.float64)
oof_xgb_probs = np.zeros((num_samples, num_classes), dtype=np.float64)
oof_cb_probs = np.zeros((num_samples, num_classes), dtype=np.float64)
test_lgb_probs = np.zeros((num_test, num_classes), dtype=np.float64)
test_xgb_probs = np.zeros((num_test, num_classes), dtype=np.float64)
test_cb_probs = np.zeros((num_test, num_classes), dtype=np.float64)

X_train_np = train_features.values
X_test_np = test_features.values

print(
    f"Beginning 5-fold training on {num_samples} samples across {num_classes} classes..."
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train)):
    X_tr, y_tr = X_train_np[train_idx], y_train[train_idx]
    X_va, y_va = X_train_np[val_idx], y_train[val_idx]

    # --- Train LightGBM ---
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=common_cols)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=common_cols)

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_preds_lgb = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
    oof_lgb_probs[val_idx] = val_preds_lgb
    test_lgb_probs += (
        lgb_model.predict(X_test_np, num_iteration=lgb_model.best_iteration) / 5.0
    )

    # --- Train XGBoost ---
    xgb_model = xgb.XGBClassifier(
        **xgb_params, n_estimators=1500, early_stopping_rounds=50
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_preds_xgb = xgb_model.predict_proba(X_va)
    oof_xgb_probs[val_idx] = val_preds_xgb
    test_xgb_probs += xgb_model.predict_proba(X_test_np) / 5.0

    # --- Train CatBoost ---
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

    val_preds_cb = cb_model.predict_proba(X_va)
    oof_cb_probs[val_idx] = val_preds_cb
    test_cb_probs += cb_model.predict_proba(X_test_np) / 5.0

    # Fold validation evaluation
    fold_blend_probs = (val_preds_lgb + val_preds_xgb + val_preds_cb) / 3.0
    fold_preds = np.argmax(fold_blend_probs, axis=1)
    fold_acc = accuracy_score(y_va, fold_preds)
    print(f"Fold {fold} - Validation Accuracy (Equal Tri-Blend): {fold_acc:.5f}")

# 9. Out-Of-Fold Optimization & Probability Blending
print("Optimizing ensemble weights on OOF predictions...")
best_acc = 0.0
best_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

for w1_i in range(0, 101, 2):
    w1 = w1_i / 100.0
    for w2_i in range(0, 101 - w1_i, 2):
        w2 = w2_i / 100.0
        w3 = round(1.0 - w1 - w2, 4)
        if w3 < 0:
            continue
        blend_probs = w1 * oof_lgb_probs + w2 * oof_xgb_probs + w3 * oof_cb_probs
        acc = accuracy_score(y_train, np.argmax(blend_probs, axis=1))
        if acc > best_acc:
            best_acc = acc
            best_weights = [w1, w2, w3]

def loss_func(weights):
    w = np.maximum(weights, 0.0)
    if w.sum() == 0:
        return 1.0
    w = w / w.sum()
    blend = w[0] * oof_lgb_probs + w[1] * oof_xgb_probs + w[2] * oof_cb_probs
    return -accuracy_score(y_train, np.argmax(blend, axis=1))

res = minimize(loss_func, best_weights, method="Nelder-Mead")
if -res.fun > best_acc:
    best_acc = -res.fun
    opt_w = np.maximum(res.x, 0.0)
    best_weights = (opt_w / opt_w.sum()).tolist()

w_lgb, w_xgb, w_cb = best_weights
print(f"Optimized Ensemble Weights: LightGBM={w_lgb:.4f}, XGBoost={w_xgb:.4f}, CatBoost={w_cb:.4f}")

oof_final_blend = w_lgb * oof_lgb_probs + w_xgb * oof_xgb_probs + w_cb * oof_cb_probs
oof_final_preds = np.argmax(oof_final_blend, axis=1)
overall_acc = accuracy_score(y_train, oof_final_preds)

print(f"Completed 5-Fold Tri-Ensemble Training. Overall OOF Accuracy: {overall_acc:.5f}")

# 10. Generate Test Predictions & Submission
test_blend_probs = (
    w_lgb * test_lgb_probs + w_xgb * test_xgb_probs + w_cb * test_cb_probs
)
test_class_indices = np.argmax(test_blend_probs, axis=1)
test_award_labels = label_encoder.inverse_transform(test_class_indices)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_award_labels)), "awardFor": test_award_labels}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(
    f"Submission successfully written to {submission_path} with shape {submission_df.shape}"
)

# Final Metric Output
print(f"Final Validation Score: {overall_acc}")
