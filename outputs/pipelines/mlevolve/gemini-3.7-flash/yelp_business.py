import ast
import json
import os
import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ---------------------------------------------------------
# 1. Setup & Directory Paths
# ---------------------------------------------------------
INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using compute device: {device}")
print("Loading raw datasets from ./input ...")
train_df = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
test_df = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

print(f"Raw train shape: {train_df.shape}, Raw test shape: {test_df.shape}")

# ---------------------------------------------------------
# 2. Target Variable Processing & Stratified Split
# ---------------------------------------------------------
target_col = "stars"
unique_stars = sorted(train_df[target_col].unique())
num_classes = len(unique_stars)
star_to_idx = {star: i for i, star in enumerate(unique_stars)}
idx_to_star = {i: float(star) for i, star in enumerate(unique_stars)}

train_df["target_class"] = train_df[target_col].map(star_to_idx)

# 80/20 Stratified Split
train_split_df, val_split_df = train_test_split(
    train_df, test_size=0.20, random_state=42, stratify=train_df["target_class"]
)
train_split_df = train_split_df.reset_index(drop=True)
val_split_df = val_split_df.reset_index(drop=True)
test_clean_df = test_df.reset_index(drop=True)

print(
    f"Train split: {len(train_split_df)} rows | Val split: {len(val_split_df)} rows | Test: {len(test_clean_df)} rows"
)


# ---------------------------------------------------------
# 3. Helper Parsers for Nested Yelp Attributes & Hours
# ---------------------------------------------------------
def parse_dict_string(val):
    """Robustly parse stringified dictionaries or return dict as is."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if not isinstance(val, str):
        try:
            if pd.isna(val):
                return {}
        except Exception:
            pass
        return {}

    val_clean = val.strip()
    if not val_clean or val_clean in ["{}", "None", "nan", "null"]:
        return {}

    # Method 1: ast.literal_eval
    try:
        parsed = ast.literal_eval(val_clean)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Method 2: Standard JSON loads
    try:
        parsed = json.loads(val_clean)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Method 3: Clean u'...' prefix
    try:
        cleaned = re.sub(r"\bu'([^']*)'", r"'\1'", val_clean)
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Method 4: Regex-based extraction of key-value pairs
    res = {}
    try:
        matches = re.finditer(
            r"['\"]?([a-zA-Z0-9_]+)['\"]?\s*:\s*([^,{}]+|\{[^{}]*\})", val_clean
        )
        for m in matches:
            k = m.group(1).strip().strip("'\"")
            v = m.group(2).strip().strip("'\"")
            res[k] = v
    except Exception:
        pass
    return res


def flatten_attributes(attr_dict):
    """Extract flat key-value pairs from nested Yelp business attributes."""
    flattened = {}
    if not isinstance(attr_dict, dict):
        return flattened

    for k, v in attr_dict.items():
        k_clean = str(k).strip().strip("'\"")
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flattened[f"{k_clean}_{str(sub_k).strip()}"] = sub_v
        elif isinstance(v, str) and ("{" in v and "}" in v):
            sub_dict = parse_dict_string(v)
            if sub_dict:
                for sub_k, sub_v in sub_dict.items():
                    flattened[f"{k_clean}_{str(sub_k).strip()}"] = sub_v
            else:
                flattened[k_clean] = v
        else:
            flattened[k_clean] = v
    return flattened


def parse_time_str(time_str):
    """Convert 'H:M' or 'HH:MM' string to decimal hours (0.0 - 24.0+)."""
    try:
        parts = time_str.split(":")
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        return h + m / 60.0
    except Exception:
        return np.nan


def parse_hours_entry(hours_dict):
    """Parse weekly opening hours into rich operational features."""
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    features = {}

    total_hours = 0.0
    days_open = 0
    late_night_count = 0
    early_morning_count = 0

    for day in days:
        day_sched = hours_dict.get(day, None) if isinstance(hours_dict, dict) else None
        if day_sched and isinstance(day_sched, str) and "-" in day_sched:
            try:
                start_str, end_str = day_sched.split("-")
                start_h = parse_time_str(start_str.strip())
                end_h = parse_time_str(end_str.strip())

                if end_h < start_h:
                    end_h += 24.0  # Open past midnight

                duration = end_h - start_h
                features[f"hours_{day}_open"] = 1
                features[f"hours_{day}_start"] = start_h
                features[f"hours_{day}_end"] = end_h
                features[f"hours_{day}_duration"] = duration

                total_hours += duration
                days_open += 1
                if end_h >= 23.0:
                    late_night_count += 1
                if start_h <= 7.0:
                    early_morning_count += 1
            except Exception:
                features[f"hours_{day}_open"] = 0
                features[f"hours_{day}_start"] = np.nan
                features[f"hours_{day}_end"] = np.nan
                features[f"hours_{day}_duration"] = 0.0
        else:
            features[f"hours_{day}_open"] = 0
            features[f"hours_{day}_start"] = np.nan
            features[f"hours_{day}_end"] = np.nan
            features[f"hours_{day}_duration"] = 0.0

    features["hours_total_weekly"] = total_hours
    features["hours_days_open_count"] = days_open
    features["hours_avg_daily_duration"] = (
        (total_hours / days_open) if days_open > 0 else 0.0
    )
    features["hours_open_weekend"] = int(
        features.get("hours_Saturday_open", 0) > 0
        or features.get("hours_Sunday_open", 0) > 0
    )
    features["hours_late_night_days"] = late_night_count
    features["hours_early_morning_days"] = early_morning_count
    return features


# ---------------------------------------------------------
# 4. Feature Extraction & Engineering
# ---------------------------------------------------------
print("Extracting attribute and operational schedule features...")
train_raw_attrs = [
    flatten_attributes(parse_dict_string(x))
    for x in train_split_df.get("attributes", [None] * len(train_split_df))
]
val_raw_attrs = [
    flatten_attributes(parse_dict_string(x))
    for x in val_split_df.get("attributes", [None] * len(val_split_df))
]
test_raw_attrs = [
    flatten_attributes(parse_dict_string(x))
    for x in test_clean_df.get("attributes", [None] * len(test_clean_df))
]

# Identify top attribute keys present in training set to avoid noisy sparse attributes
attr_key_counts = {}
for attr_dict in train_raw_attrs:
    for k in attr_dict.keys():
        attr_key_counts[k] = attr_key_counts.get(k, 0) + 1

top_attr_keys = sorted([k for k, count in attr_key_counts.items() if count >= 15])
print(f"Discovered {len(top_attr_keys)} prominent attribute keys from train.")


def process_attribute_values(attr_list, selected_keys):
    records = []
    cat_enc_maps = {
        "alcohol": {"none": 0.0, "beer_and_wine": 1.0, "full_bar": 2.0},
        "noiselevel": {
            "quiet": 1.0,
            "average": 2.0,
            "loud": 3.0,
            "very_loud": 4.0,
        },
        "wifi": {"no": 0.0, "free": 1.0, "paid": 2.0},
        "restaurantsattire": {"casual": 1.0, "dressy": 2.0, "formal": 3.0},
    }

    for d in attr_list:
        row = {}
        for k in selected_keys:
            val = d.get(k, None)
            col_name = f"attr_{k}"
            if val is None:
                row[col_name] = np.nan
            elif isinstance(val, bool):
                row[col_name] = 1.0 if val else 0.0
            elif isinstance(val, (int, float)):
                row[col_name] = float(val)
            elif isinstance(val, str):
                val_str = (
                    val.strip().lower().strip("'\"").replace("u'", "").strip("'\"")
                )
                k_lower = k.lower()

                # Specific categorical maps
                if "alcohol" in k_lower and val_str in cat_enc_maps["alcohol"]:
                    row[col_name] = cat_enc_maps["alcohol"][val_str]
                elif (
                    "noiselevel" in k_lower
                    and val_str in cat_enc_maps["noiselevel"]
                ):
                    row[col_name] = cat_enc_maps["noiselevel"][val_str]
                elif "wifi" in k_lower and val_str in cat_enc_maps["wifi"]:
                    row[col_name] = cat_enc_maps["wifi"][val_str]
                elif (
                    "restaurantsattire" in k_lower
                    and val_str in cat_enc_maps["restaurantsattire"]
                ):
                    row[col_name] = cat_enc_maps["restaurantsattire"][val_str]
                elif val_str in ["true", "t", "yes", "free", "all", "1", "1.0"]:
                    row[col_name] = 1.0
                elif val_str in [
                    "false",
                    "f",
                    "no",
                    "none",
                    "null",
                    "0",
                    "0.0",
                ]:
                    row[col_name] = 0.0
                else:
                    try:
                        row[col_name] = float(val_str)
                    except ValueError:
                        row[col_name] = float(abs(hash(val_str)) % 20 + 1)
            else:
                row[col_name] = np.nan
        records.append(row)
    return pd.DataFrame(records)


train_attr_df = process_attribute_values(train_raw_attrs, top_attr_keys)
val_attr_df = process_attribute_values(val_raw_attrs, top_attr_keys)
test_attr_df = process_attribute_values(test_raw_attrs, top_attr_keys)

# Hours parsing
train_raw_hours = [
    parse_hours_entry(parse_dict_string(x))
    for x in train_split_df.get("hours", [None] * len(train_split_df))
]
val_raw_hours = [
    parse_hours_entry(parse_dict_string(x))
    for x in val_split_df.get("hours", [None] * len(val_split_df))
]
test_raw_hours = [
    parse_hours_entry(parse_dict_string(x))
    for x in test_clean_df.get("hours", [None] * len(test_clean_df))
]

train_hours_df = pd.DataFrame(train_raw_hours)
val_hours_df = pd.DataFrame(val_raw_hours)
test_hours_df = pd.DataFrame(test_raw_hours)

# ---------------------------------------------------------
# 5. Spatial & Geographic Feature Engineering
# ---------------------------------------------------------
print("Computing spatial, geographic, and density features...")


def extract_base_tabular(df):
    feats = pd.DataFrame(index=df.index)

    lat = (
        df["latitude"].fillna(df["latitude"].median()).values
        if "latitude" in df
        else np.zeros(len(df))
    )
    lon = (
        df["longitude"].fillna(df["longitude"].median()).values
        if "longitude" in df
        else np.zeros(len(df))
    )

    feats["latitude"] = lat
    feats["longitude"] = lon
    feats["lat_rad"] = np.radians(lat)
    feats["lon_rad"] = np.radians(lon)
    feats["coord_x"] = np.cos(feats["lat_rad"]) * np.cos(feats["lon_rad"])
    feats["coord_y"] = np.cos(feats["lat_rad"]) * np.sin(feats["lon_rad"])
    feats["coord_z"] = np.sin(feats["lat_rad"])

    rev = (
        df["review_count"].fillna(0).values
        if "review_count" in df
        else np.zeros(len(df))
    )
    feats["review_count"] = rev
    feats["log_review_count"] = np.log1p(np.maximum(0, rev))
    feats["is_open"] = (
        df["is_open"].fillna(1).astype(float).values
        if "is_open" in df
        else np.ones(len(df))
    )

    name_str = df["name"].fillna("").astype(str)
    feats["name_len"] = name_str.str.len()
    feats["name_words"] = name_str.str.split().apply(len)
    feats["name_has_inc"] = name_str.str.contains(
        r"\b(inc|llc|corp|co|ltd)\b", case=False, regex=True
    ).astype(float)
    feats["name_has_digits"] = name_str.str.contains(r"\d", regex=True).astype(float)

    addr_str = df["address"].fillna("").astype(str)
    feats["addr_len"] = addr_str.str.len()
    feats["addr_has_suite"] = addr_str.str.contains(
        r"\b(ste|suite|apt|unit|fl|floor|#)\b", case=False, regex=True
    ).astype(float)

    return feats


train_base_df = extract_base_tabular(train_split_df)
val_base_df = extract_base_tabular(val_split_df)
test_base_df = extract_base_tabular(test_clean_df)

# Distance to City Center computed strictly from Train
city_centers = (
    train_split_df.groupby("city")[["latitude", "longitude"]]
    .agg(["median", "count"])
    .dropna()
)
city_center_coords = {}
for city_name, row in city_centers.iterrows():
    if row[("latitude", "count")] >= 3:
        city_center_coords[str(city_name).lower().strip()] = (
            row[("latitude", "median")],
            row[("longitude", "median")],
        )


def compute_city_distance(df):
    dists = []
    for _, row in df.iterrows():
        city_key = str(row.get("city", "")).lower().strip()
        if city_key in city_center_coords:
            c_lat, c_lon = city_center_coords[city_key]
            b_lat = row.get("latitude", c_lat)
            b_lon = row.get("longitude", c_lon)
            if pd.isna(b_lat) or pd.isna(b_lon):
                dists.append(0.0)
            else:
                d = np.sqrt((b_lat - c_lat) ** 2 + (b_lon - c_lon) ** 2)
                dists.append(d)
        else:
            dists.append(-1.0)
    return pd.Series(dists)


train_base_df["dist_to_city_center"] = compute_city_distance(train_split_df)
val_base_df["dist_to_city_center"] = compute_city_distance(val_split_df)
test_base_df["dist_to_city_center"] = compute_city_distance(test_clean_df)

# Frequency Encoding for categorical location columns
for cat_col in ["city", "state", "postal_code"]:
    freq_map = (
        train_split_df[cat_col].fillna("MISSING").value_counts(normalize=True).to_dict()
    )
    train_base_df[f"{cat_col}_freq"] = (
        train_split_df[cat_col].fillna("MISSING").map(freq_map).fillna(0.0)
    )
    val_base_df[f"{cat_col}_freq"] = (
        val_split_df[cat_col].fillna("MISSING").map(freq_map).fillna(0.0)
    )
    test_base_df[f"{cat_col}_freq"] = (
        test_clean_df[cat_col].fillna("MISSING").map(freq_map).fillna(0.0)
    )

# Chain business indicator (business name frequency from train)
name_freq_map = train_split_df["name"].fillna("").value_counts().to_dict()
train_base_df["is_chain"] = (
    train_split_df["name"].fillna("").map(name_freq_map).fillna(0) > 1
).astype(float)
val_base_df["is_chain"] = (
    val_split_df["name"].fillna("").map(name_freq_map).fillna(0) > 1
).astype(float)
test_base_df["is_chain"] = (
    test_clean_df["name"].fillna("").map(name_freq_map).fillna(0) > 1
).astype(float)

# ---------------------------------------------------------
# 6. Categories Text Feature Engineering (TF-IDF + SVD)
# ---------------------------------------------------------
print("Extracting category semantics with TF-IDF & TruncatedSVD...")
train_cats_text = train_split_df["categories"].fillna("").astype(str)
val_cats_text = val_split_df["categories"].fillna("").astype(str)
test_cats_text = test_clean_df["categories"].fillna("").astype(str)

train_base_df["category_count"] = train_cats_text.apply(
    lambda x: len(x.split(",")) if x else 0
)
val_base_df["category_count"] = val_cats_text.apply(
    lambda x: len(x.split(",")) if x else 0
)
test_base_df["category_count"] = test_cats_text.apply(
    lambda x: len(x.split(",")) if x else 0
)

macro_categories = [
    "Restaurants",
    "Food",
    "Nightlife",
    "Bars",
    "Shopping",
    "Beauty & Spas",
    "Health & Medical",
    "Home Services",
    "Automotive",
    "Active Life",
    "Event Planning & Services",
    "Coffee & Tea",
    "Fast Food",
    "Sandwiches",
    "Pizza",
    "Burgers",
    "Mexican",
    "Italian",
    "American (Traditional)",
    "American (New)",
    "Breakfast & Brunch",
    "Specialty Food",
    "Bakeries",
]
for cat in macro_categories:
    cat_clean_col = "cat_is_" + re.sub(r"[^a-zA-Z0-9]", "_", cat).lower()
    train_base_df[cat_clean_col] = train_cats_text.str.contains(
        re.escape(cat), case=False, regex=True
    ).astype(float)
    val_base_df[cat_clean_col] = val_cats_text.str.contains(
        re.escape(cat), case=False, regex=True
    ).astype(float)
    test_base_df[cat_clean_col] = test_cats_text.str.contains(
        re.escape(cat), case=False, regex=True
    ).astype(float)

# TF-IDF & SVD Decomposition strictly fit on train split
tfidf = TfidfVectorizer(
    token_pattern=r"(?u)\b[\w-]+\b", min_df=5, max_features=1000, ngram_range=(1, 2)
)
tfidf_train = tfidf.fit_transform(train_cats_text)
tfidf_val = tfidf.transform(val_cats_text)
tfidf_test = tfidf.transform(test_cats_text)

n_svd_components = 25
svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
svd_train = svd.fit_transform(tfidf_train)
svd_val = svd.transform(tfidf_val)
svd_test = svd.transform(tfidf_test)

svd_cols = [f"cat_svd_{i}" for i in range(n_svd_components)]
train_svd_df = pd.DataFrame(svd_train, columns=svd_cols)
val_svd_df = pd.DataFrame(svd_val, columns=svd_cols)
test_svd_df = pd.DataFrame(svd_test, columns=svd_cols)

# ---------------------------------------------------------
# 7. Leak-Free Out-Of-Fold Target Encoding on Train
# ---------------------------------------------------------
print("Applying strictly leak-free Out-Of-Fold Target Encoding...")
target_num = train_split_df[target_col].values
global_mean_target = float(np.mean(target_num))

te_cols = ["city", "state", "postal_code"]
train_te_feats = pd.DataFrame(index=train_split_df.index)
val_te_feats = pd.DataFrame(index=val_split_df.index)
test_te_feats = pd.DataFrame(index=test_clean_df.index)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for col in te_cols:
    train_te_feats[f"{col}_target_enc"] = np.nan
    val_te_feats[f"{col}_target_enc"] = np.nan
    test_te_feats[f"{col}_target_enc"] = np.nan

    # OOF on train split
    for tr_idx, oof_idx in skf.split(train_split_df, train_split_df["target_class"]):
        fold_train = train_split_df.iloc[tr_idx]
        fold_stats = fold_train.groupby(col)[target_col].agg(["count", "mean"])

        smooth_weight = 10
        smoothed = (
            fold_stats["count"] * fold_stats["mean"]
            + smooth_weight * global_mean_target
        ) / (fold_stats["count"] + smooth_weight)

        oof_col_val = train_split_df.iloc[oof_idx][col]
        train_te_feats.iloc[
            oof_idx, train_te_feats.columns.get_loc(f"{col}_target_enc")
        ] = (oof_col_val.map(smoothed).fillna(global_mean_target).values)

    # Full train split mapping for Val and Test
    full_stats = train_split_df.groupby(col)[target_col].agg(["count", "mean"])
    full_smoothed = (
        full_stats["count"] * full_stats["mean"] + 10 * global_mean_target
    ) / (full_stats["count"] + 10)

    val_te_feats[f"{col}_target_enc"] = (
        val_split_df[col].map(full_smoothed).fillna(global_mean_target).values
    )
    test_te_feats[f"{col}_target_enc"] = (
        test_clean_df[col].map(full_smoothed).fillna(global_mean_target).values
    )

# ---------------------------------------------------------
# 8. Feature Concatenation & Consistency Alignment
# ---------------------------------------------------------
print("Assembling final feature matrices...")
X_train = pd.concat(
    [train_base_df, train_hours_df, train_attr_df, train_svd_df, train_te_feats], axis=1
)
X_val = pd.concat(
    [val_base_df, val_hours_df, val_attr_df, val_svd_df, val_te_feats], axis=1
)
X_test = pd.concat(
    [test_base_df, test_hours_df, test_attr_df, test_svd_df, test_te_feats], axis=1
)

feature_columns = list(X_train.columns)
X_val = X_val[feature_columns]
X_test = X_test[feature_columns]

y_train = train_split_df["target_class"].values.astype(np.int64)
y_val = val_split_df["target_class"].values.astype(np.int64)

num_features = len(feature_columns)
print(f"Final Feature Matrix Dimension: {num_features} engineered features.")
print(
    f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}"
)


# ---------------------------------------------------------
# 9. Model Architectures & Configurations
# ---------------------------------------------------------
class TabularResidualBlock(nn.Module):
    """Residual Block for tabular deep learning with LayerNorm, SiLU, and Dropout."""

    def __init__(self, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = self.activation(out + residual)
        return self.dropout(out)


class TabularResNet(nn.Module):
    """Deep Tabular ResNet model designed for multi-class rating classification."""

    def __init__(
        self,
        input_dim,
        num_classes=9,
        hidden_dims=[256, 256, 128],
        dropout_rate=0.2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(hidden_dims[0], dropout_rate=dropout_rate)
                for _ in range(2)
            ]
        )

        self.transition = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dims[2], num_classes),
        )

    def forward(self, x):
        h = self.input_layer(x)
        for res_block in self.res_blocks:
            h = res_block(h)
        h = self.transition(h)
        logits = self.head(h)
        return logits


def get_torch_criterion(label_smoothing=0.05):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def build_torch_optimizer_and_scheduler(
    model, lr=2e-3, weight_decay=1e-4, max_epochs=20, steps_per_epoch=100
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    return optimizer, scheduler


def get_lgbm_params(num_classes=9, random_state=42):
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 45,
        "max_depth": 7,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 25,
        "lambda_l1": 0.1,
        "lambda_l2": 1.5,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(num_classes=9, random_state=42):
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }


# ---------------------------------------------------------
# 10. Model Training & Evaluation
# ---------------------------------------------------------
# 10.1 LightGBM
print("--- Training LightGBM Model ---")
lgb_train_data = lgb.Dataset(X_train, label=y_train)
lgb_val_data = lgb.Dataset(X_val, label=y_val, reference=lgb_train_data)

lgb_params = get_lgbm_params(num_classes=num_classes, random_state=42)
lgb_callbacks = [
    lgb.early_stopping(stopping_rounds=60, verbose=False),
    lgb.log_evaluation(period=0),
]

lgb_model = lgb.train(
    lgb_params,
    lgb_train_data,
    num_boost_round=1200,
    valid_sets=[lgb_val_data],
    callbacks=lgb_callbacks,
)

lgb_val_probs = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
lgb_test_probs = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
lgb_val_acc = accuracy_score(y_val, np.argmax(lgb_val_probs, axis=1))
print(
    f"LightGBM Best Iteration: {lgb_model.best_iteration} | Validation Accuracy: {lgb_val_acc:.5f}"
)

# 10.2 XGBoost
print("--- Training XGBoost Model ---")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

xgb_params = get_xgb_params(num_classes=num_classes, random_state=42)
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1200,
    evals=[(dval, "val")],
    early_stopping_rounds=60,
    verbose_eval=False,
)

xgb_val_probs = xgb_model.predict(dval)
xgb_test_probs = xgb_model.predict(dtest)
xgb_val_acc = accuracy_score(y_val, np.argmax(xgb_val_probs, axis=1))
print(
    f"XGBoost Best Iteration: {xgb_model.best_iteration} | Validation Accuracy: {xgb_val_acc:.5f}"
)

# 10.3 PyTorch TabularResNet
print("--- Training TabularResNet ---")
train_medians = X_train.median()
X_train_imp = X_train.fillna(train_medians).fillna(0.0).values
X_val_imp = X_val.fillna(train_medians).fillna(0.0).values
X_test_imp = X_test.fillna(train_medians).fillna(0.0).values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)
X_val_scaled = scaler.transform(X_val_imp).astype(np.float32)
X_test_scaled = scaler.transform(X_test_imp).astype(np.float32)

torch_train_ds = TensorDataset(
    torch.from_numpy(X_train_scaled), torch.from_numpy(y_train)
)
torch_val_ds = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val))
torch_test_tensor = torch.from_numpy(X_test_scaled).to(device)

train_loader = DataLoader(torch_train_ds, batch_size=256, shuffle=True, drop_last=False)
val_loader = DataLoader(torch_val_ds, batch_size=512, shuffle=False)

torch_model = TabularResNet(input_dim=num_features, num_classes=num_classes).to(device)
max_epochs = 20
optimizer, scheduler = build_torch_optimizer_and_scheduler(
    torch_model,
    lr=2e-3,
    weight_decay=1e-4,
    max_epochs=max_epochs,
    steps_per_epoch=len(train_loader),
)
criterion = get_torch_criterion(label_smoothing=0.05)

best_val_loss = float("inf")
best_model_weights = None

for epoch in range(max_epochs):
    torch_model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = torch_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(torch_model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(batch_y)

    train_loss = total_loss / len(torch_train_ds)

    torch_model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = torch_model(batch_x)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()

    val_loss /= len(torch_val_ds)
    val_acc = correct / len(torch_val_ds)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = {
            k: v.cpu().clone() for k, v in torch_model.state_dict().items()
        }

    if (epoch + 1) % 5 == 0 or epoch == max_epochs - 1:
        print(
            f"Epoch [{epoch+1:02d}/{max_epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.5f}"
        )

if best_model_weights is not None:
    torch_model.load_state_dict(
        {k: v.to(device) for k, v in best_model_weights.items()}
    )

torch_model.eval()
with torch.no_grad():
    val_logits = torch_model(torch.from_numpy(X_val_scaled).to(device))
    torch_val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()

    test_logits = torch_model(torch_test_tensor)
    torch_test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

torch_val_acc = accuracy_score(y_val, np.argmax(torch_val_probs, axis=1))
print(f"TabularResNet Best Validation Accuracy: {torch_val_acc:.5f}")

# ---------------------------------------------------------
# 11. Probability Blending & Ensemble Validation
# ---------------------------------------------------------
print("--- Computing Multi-Model Ensemble Predictions ---")
w_lgb, w_xgb, w_nn = 0.45, 0.40, 0.15
ensemble_val_probs = (
    w_lgb * lgb_val_probs + w_xgb * xgb_val_probs + w_nn * torch_val_probs
)
ensemble_test_probs = (
    w_lgb * lgb_test_probs + w_xgb * xgb_test_probs + w_nn * torch_test_probs
)

final_val_preds_idx = np.argmax(ensemble_val_probs, axis=1)
final_val_score = accuracy_score(y_val, final_val_preds_idx)
print(f"Ensemble Hold-out Validation Accuracy: {final_val_score:.6f}")

# ---------------------------------------------------------
# 12. Generate Submission File
# ---------------------------------------------------------
final_test_preds_idx = np.argmax(ensemble_test_probs, axis=1)
final_test_stars = [idx_to_star[int(idx)] for idx in final_test_preds_idx]

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_stars), dtype=int), "stars": final_test_stars}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

# Sanity assertions
assert os.path.exists(submission_path), "Submission file was not created!"
assert len(submission_df) == len(
    X_test
), f"Row count mismatch: expected {len(X_test)}, got {len(submission_df)}"
assert list(submission_df.columns) == [
    "id",
    "stars",
], f"Unexpected columns: {submission_df.columns}"
assert not submission_df.isnull().any().any(), "Submission contains null values!"

print(
    f"Submission successfully written to {submission_path} with {len(submission_df)} predictions."
)
print(
    f"Submission distribution preview:\n{submission_df['stars'].value_counts(normalize=True).sort_index()}"
)

# Mandatory Final Validation Score Output
print(f"Final Validation Score: {final_val_score:.6f}")
