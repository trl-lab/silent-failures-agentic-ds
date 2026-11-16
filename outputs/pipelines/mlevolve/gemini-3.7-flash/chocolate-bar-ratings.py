import json
import os
import re
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# Set random seeds for strict reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# =========================================================================
# 1. Data Loading & Domain Feature Engineering
# =========================================================================

train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")


def clean_col_name(c):
    return re.sub(r"[^\w\s]", "", c).strip().replace(" ", "_").lower()


train_df = train_raw.copy()
test_df = test_raw.copy()
train_df.columns = [clean_col_name(c) for c in train_df.columns]
test_df.columns = [clean_col_name(c) for c in test_df.columns]

# Locate target column
target_col = None
for col in ["rating", "ratings", "target"]:
    if col in train_df.columns:
        target_col = col
        break

if target_col is None:
    for col in train_df.columns:
        if "rat" in col:
            target_col = col
            break

assert target_col is not None, "Target column 'Rating' not found in training data."

# Standardize column mapping references
col_map = {
    "cocoa": [
        c
        for c in train_df.columns
        if ("cocoa" in c and ("percent" in c or "pct" in c)) or c == "cocoa_percent"
    ],
    "ingredients": [c for c in train_df.columns if "ingredient" in c],
    "review_date": [
        c for c in train_df.columns if "date" in c or "year" in c or c == "review_date"
    ],
    "ref": [c for c in train_df.columns if c == "ref"],
    "company": [
        c
        for c in train_df.columns
        if (
            ("company" in c and "location" not in c)
            or "maker" in c
            or "manufacturer" in c
        )
    ],
    "company_loc": [
        c
        for c in train_df.columns
        if "company_location" in c or ("company" in c and "loc" in c)
    ],
    "origin": [
        c
        for c in train_df.columns
        if (
            ("origin" in c and "specific" not in c)
            or "country_of_bean_origin" in c
            or "broad_bean_origin" in c
        )
    ],
    "specific_origin": [
        c for c in train_df.columns if "specific" in c or "bar_name" in c
    ],
    "characteristics": [
        c
        for c in train_df.columns
        if "characteristic" in c or "tasting" in c or "notes" in c or "memorable" in c
    ],
    "bean_type": [c for c in train_df.columns if "bean_type" in c or "variety" in c],
}


def get_col(key):
    cols = col_map.get(key, [])
    return cols[0] if len(cols) > 0 else None


c_cocoa = get_col("cocoa")
c_ing = get_col("ingredients")
c_date = get_col("review_date")
c_ref = get_col("ref")
c_comp = get_col("company")
c_loc = get_col("company_loc")
c_origin = get_col("origin")
c_specific = get_col("specific_origin")
c_char = get_col("characteristics")
c_bean = get_col("bean_type")


def parse_cocoa_percent(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).replace("%", "").strip()
    try:
        f = float(val_str)
        return f if f > 1.0 else f * 100.0
    except:
        return np.nan


def parse_ingredient_features(df, col_name):
    if col_name is None or col_name not in df.columns:
        return pd.DataFrame(index=df.index)

    feats = pd.DataFrame(index=df.index)
    raw_str = df[col_name].fillna("").astype(str)

    def extract_count(x):
        match = re.match(r"^\s*(\d+)", x)
        if match:
            return float(match.group(1))
        tokens = [
            t.strip() for t in re.sub(r"^\d+\s*[-–:]\s*", "", x).split(",") if t.strip()
        ]
        return float(len(tokens)) if tokens else np.nan

    feats["feat_ing_count"] = raw_str.apply(extract_count)

    for code, name in [
        ("B", "has_beans"),
        ("S", "has_sugar"),
        ("S*", "has_alt_sweetener"),
        ("C", "has_cocoa_butter"),
        ("V", "has_vanilla"),
        ("L", "has_lecithin"),
        ("Sa", "has_salt"),
    ]:
        if code == "S*":
            feats[f"feat_{name}"] = raw_str.apply(
                lambda x: 1 if ("S*" in x or "s*" in x) else 0
            )
        elif code == "S":
            feats[f"feat_{name}"] = raw_str.apply(
                lambda x: (
                    1
                    if (
                        re.search(r"(?<!\*)\bS\b(?!\*)", x)
                        or ("S" in x and "S*" not in x)
                    )
                    else 0
                )
            )
        else:
            feats[f"feat_{name}"] = raw_str.apply(
                lambda x: 1 if re.search(rf"\b{code}\b", x, re.IGNORECASE) else 0
            )

    feats["feat_is_pure_chocolate"] = (
        (feats["feat_has_vanilla"] == 0)
        & (feats["feat_has_lecithin"] == 0)
        & (feats["feat_has_salt"] == 0)
    ).astype(int)
    feats["feat_is_single_additive"] = (
        feats["feat_has_vanilla"] + feats["feat_has_lecithin"] + feats["feat_has_salt"]
        == 1
    ).astype(int)
    feats["feat_has_additives"] = (
        feats["feat_has_vanilla"] + feats["feat_has_lecithin"] + feats["feat_has_salt"]
        > 0
    ).astype(int)

    return feats


def parse_text_characteristics(df, col_name):
    if col_name is None or col_name not in df.columns:
        return pd.DataFrame(index=df.index), pd.Series("", index=df.index)

    feats = pd.DataFrame(index=df.index)
    raw_str = df[col_name].fillna("").astype(str).str.lower()

    feats["feat_char_len"] = raw_str.apply(len)
    feats["feat_char_word_count"] = raw_str.apply(
        lambda x: len(x.split()) if x.strip() else 0
    )
    feats["feat_char_comma_count"] = raw_str.apply(lambda x: x.count(","))

    sensory_lexicon = {
        "flavor_fruity": [
            "fruit",
            "fruity",
            "berry",
            "cherry",
            "citrus",
            "lemon",
            "orange",
            "plum",
            "raisin",
            "banana",
            "fig",
            "apple",
            "grape",
            "raspberry",
            "blackberry",
            "strawberry",
            "tropical",
            "mango",
            "apricot",
            "melon",
            "pear",
        ],
        "flavor_nutty": [
            "nut",
            "nutty",
            "hazelnut",
            "almond",
            "peanut",
            "cashew",
            "walnut",
            "pecan",
            "macadamia",
            "pistachio",
            "praline",
        ],
        "flavor_floral": [
            "floral",
            "flower",
            "jasmine",
            "rose",
            "lavender",
            "blossom",
            "violet",
        ],
        "flavor_sweet_caramel": [
            "sweet",
            "caramel",
            "honey",
            "molasses",
            "toffee",
            "butterscotch",
            "maple",
            "sugar",
            "marshmallow",
            "fudge",
            "syrup",
        ],
        "flavor_roasty_cocoa": [
            "roast",
            "roasted",
            "cocoa",
            "dark",
            "intense",
            "deep",
            "brownie",
            "coffee",
            "espresso",
            "mocha",
            "toast",
            "biscuit",
            "malt",
        ],
        "flavor_earthy_woody": [
            "earthy",
            "earth",
            "wood",
            "woody",
            "tobacco",
            "leather",
            "forest",
            "soil",
            "moss",
            "cedar",
            "oak",
            "hay",
            "grass",
            "herbal",
        ],
        "flavor_spicy": [
            "spice",
            "spicy",
            "cinnamon",
            "clove",
            "nutmeg",
            "pepper",
            "ginger",
            "cardamom",
            "anise",
            "licorice",
        ],
        "flavor_creamy_dairy": [
            "creamy",
            "cream",
            "milk",
            "milky",
            "butter",
            "buttery",
            "dairy",
            "fatty",
            "smooth",
            "rich",
            "silky",
        ],
        "flavor_acidic_sour": [
            "sour",
            "tart",
            "acidic",
            "acid",
            "citric",
            "tangy",
            "vinegar",
            "ferment",
            "fermented",
        ],
        "flavor_bitter_astringent": [
            "bitter",
            "astringent",
            "dry",
            "chalky",
            "harsh",
            "sharp",
            "pungent",
            "tannic",
            "drying",
        ],
        "flavor_off_defect": [
            "chemical",
            "rubber",
            "burnt",
            "smoke",
            "smokey",
            "smoky",
            "musty",
            "mold",
            "moldy",
            "dusty",
            "stale",
            "metallic",
            "medicinal",
            "plastic",
            "hammy",
            "dirty",
            "flat",
            "bland",
            "cardboard",
        ],
    }

    for cat, keywords in sensory_lexicon.items():
        pattern = r"\b(" + "|".join(keywords) + r")\b"
        feats[f"feat_{cat}"] = raw_str.apply(lambda x: int(bool(re.search(pattern, x))))

    return feats, raw_str


train_feat_ing = parse_ingredient_features(train_df, c_ing)
test_feat_ing = parse_ingredient_features(test_df, c_ing)

train_feat_char, train_char_text = parse_text_characteristics(train_df, c_char)
test_feat_char, test_char_text = parse_text_characteristics(test_df, c_char)

train_processed = pd.DataFrame(index=train_df.index)
test_processed = pd.DataFrame(index=test_df.index)

# Cocoa features
if c_cocoa:
    train_processed["feat_cocoa_percent"] = train_df[c_cocoa].apply(parse_cocoa_percent)
    test_processed["feat_cocoa_percent"] = test_df[c_cocoa].apply(parse_cocoa_percent)

    mean_cocoa = float(train_processed["feat_cocoa_percent"].median())
    train_processed["feat_cocoa_percent"] = train_processed[
        "feat_cocoa_percent"
    ].fillna(mean_cocoa)
    test_processed["feat_cocoa_percent"] = test_processed["feat_cocoa_percent"].fillna(
        mean_cocoa
    )

    for df_p in [train_processed, test_processed]:
        df_p["feat_cocoa_dev_from_70"] = (df_p["feat_cocoa_percent"] - 70.0).abs()
        df_p["feat_is_high_cocoa"] = (df_p["feat_cocoa_percent"] >= 80.0).astype(int)
        df_p["feat_is_standard_cocoa"] = (
            (df_p["feat_cocoa_percent"] >= 68.0) & (df_p["feat_cocoa_percent"] <= 75.0)
        ).astype(int)
        df_p["feat_is_low_cocoa"] = (df_p["feat_cocoa_percent"] < 68.0).astype(int)

# Date / REF features
if c_date and c_date in train_df.columns:
    train_processed["feat_review_date"] = (
        pd.to_numeric(train_df[c_date], errors="coerce").fillna(2015).astype(float)
    )
    test_processed["feat_review_date"] = (
        pd.to_numeric(test_df[c_date], errors="coerce").fillna(2015).astype(float)
    )

if c_ref and c_ref in train_df.columns:
    train_processed["feat_ref"] = (
        pd.to_numeric(train_df[c_ref], errors="coerce").fillna(0).astype(float)
    )
    test_processed["feat_ref"] = (
        pd.to_numeric(test_df[c_ref], errors="coerce").fillna(0).astype(float)
    )

# Geography & Terroir interaction
if c_loc and c_origin and c_loc in train_df.columns and c_origin in train_df.columns:

    def check_domestic(row, loc_c, orig_c):
        loc = str(row[loc_c]).strip().lower() if pd.notna(row[loc_c]) else ""
        orig = str(row[orig_c]).strip().lower() if pd.notna(row[orig_c]) else ""
        if not loc or not orig:
            return 0
        return int(loc == orig or loc in orig or orig in loc)

    train_processed["feat_is_domestic_bean"] = train_df.apply(
        lambda r: check_domestic(r, c_loc, c_origin), axis=1
    )
    test_processed["feat_is_domestic_bean"] = test_df.apply(
        lambda r: check_domestic(r, c_loc, c_origin), axis=1
    )

    train_processed["feat_is_blend"] = (
        train_df[c_origin]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(lambda x: 1 if ("blend" in x or "," in x or "/" in x or "&" in x) else 0)
    )
    test_processed["feat_is_blend"] = (
        test_df[c_origin]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(lambda x: 1 if ("blend" in x or "," in x or "/" in x or "&" in x) else 0)
    )

# Combine engineered frames
train_processed = pd.concat([train_processed, train_feat_ing, train_feat_char], axis=1)
test_processed = pd.concat([test_processed, test_feat_ing, test_feat_char], axis=1)

# Fit TF-IDF & SVD exclusively on train characteristics
tfidf = TfidfVectorizer(
    max_features=250, ngram_range=(1, 2), min_df=2, stop_words="english"
)
train_tfidf = tfidf.fit_transform(train_char_text)
test_tfidf = tfidf.transform(test_char_text)

svd = TruncatedSVD(n_components=12, random_state=42)
train_svd = svd.fit_transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

for i in range(12):
    train_processed[f"feat_text_svd_{i}"] = train_svd[:, i]
    test_processed[f"feat_text_svd_{i}"] = test_svd[:, i]

# Categorical encodings
cat_cols = [
    c
    for c in [c_comp, c_loc, c_origin, c_specific, c_bean]
    if c is not None and c in train_df.columns
]

for col in cat_cols:
    freq_map = (
        train_df[col]
        .fillna("Unknown")
        .astype(str)
        .value_counts(normalize=True)
        .to_dict()
    )
    train_processed[f"feat_{col}_freq"] = (
        train_df[col]
        .fillna("Unknown")
        .astype(str)
        .map(freq_map)
        .fillna(0.0)
        .astype(float)
    )
    test_processed[f"feat_{col}_freq"] = (
        test_df[col]
        .fillna("Unknown")
        .astype(str)
        .map(freq_map)
        .fillna(0.0)
        .astype(float)
    )

    le = LabelEncoder()
    train_vals = train_df[col].fillna("Unknown").astype(str).tolist()
    all_known_cats = list(set(train_vals))
    le.fit(all_known_cats + ["__UNKNOWN__"])

    cat_to_id = {val: idx for idx, val in enumerate(le.classes_)}
    unknown_id = cat_to_id["__UNKNOWN__"]

    train_processed[f"cat_{col}"] = (
        train_df[col]
        .fillna("Unknown")
        .astype(str)
        .apply(lambda x: cat_to_id.get(x, unknown_id))
        .astype(int)
    )
    test_processed[f"cat_{col}"] = (
        test_df[col]
        .fillna("Unknown")
        .astype(str)
        .apply(lambda x: cat_to_id.get(x, unknown_id))
        .astype(int)
    )

# Target labels and Stratified 5-Fold setup
raw_target = train_df[target_col].copy()
unique_ratings = sorted(list(raw_target.unique()), key=lambda x: str(x))
rating_to_class = {val: idx for idx, val in enumerate(unique_ratings)}
class_to_rating = {idx: val for idx, val in enumerate(unique_ratings)}
num_classes = len(unique_ratings)

y_classes = raw_target.map(rating_to_class).values
train_processed["target_class"] = y_classes
train_processed["raw_rating"] = raw_target.values


def rating_to_numeric(val):
    val_str = str(val).strip()
    if "under" in val_str.lower():
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", val_str)
        if nums:
            return float(nums[0]) - 0.25
        return 2.5
    if "over" in val_str.lower():
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", val_str)
        if nums:
            return float(nums[0]) + 0.25
        return 4.0
    try:
        return float(val_str)
    except:
        return 3.0


numeric_target = raw_target.map(rating_to_numeric).astype(float)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_processed["fold"] = -1
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_processed, y_classes)):
    train_processed.loc[val_idx, "fold"] = fold_idx

# Out-of-fold target statistics (leak-free)
for group_col in [c_comp, c_origin, c_loc]:
    if group_col is None or group_col not in train_df.columns:
        continue

    feat_name = f"feat_{group_col}_oof_mean_rating"
    feat_cnt_name = f"feat_{group_col}_oof_count"
    train_processed[feat_name] = np.nan
    train_processed[feat_cnt_name] = 0.0
    global_mean = float(numeric_target.mean())

    for f in range(5):
        tr_mask = train_processed["fold"] != f
        va_mask = train_processed["fold"] == f

        tr_group = train_df.loc[tr_mask, group_col].fillna("Unknown").astype(str)
        tr_y = numeric_target.loc[tr_mask]

        group_stats = tr_y.groupby(tr_group).agg(["mean", "count"])
        m = 5.0
        smooth_stat = (group_stats["mean"] * group_stats["count"] + global_mean * m) / (
            group_stats["count"] + m
        )
        smooth_map = smooth_stat.to_dict()
        cnt_map = group_stats["count"].to_dict()

        val_group = train_df.loc[va_mask, group_col].fillna("Unknown").astype(str)
        train_processed.loc[va_mask, feat_name] = (
            val_group.map(smooth_map).fillna(global_mean).astype(float)
        )
        train_processed.loc[va_mask, feat_cnt_name] = (
            val_group.map(cnt_map).fillna(0).astype(float)
        )

    full_group = train_df[group_col].fillna("Unknown").astype(str)
    full_stats = numeric_target.groupby(full_group).agg(["mean", "count"])
    full_smooth = (full_stats["mean"] * full_stats["count"] + global_mean * 5.0) / (
        full_stats["count"] + 5.0
    )
    full_smooth_map = full_smooth.to_dict()
    full_cnt_map = full_stats["count"].to_dict()

    test_group = test_df[group_col].fillna("Unknown").astype(str)
    test_processed[feat_name] = (
        test_group.map(full_smooth_map).fillna(global_mean).astype(float)
    )
    test_processed[feat_cnt_name] = test_group.map(full_cnt_map).fillna(0).astype(float)

feature_columns = [
    c
    for c in train_processed.columns
    if c not in ["fold", "target_class", "raw_rating"]
]
categorical_columns = [c for c in feature_columns if c.startswith("cat_")]
numeric_columns = [c for c in feature_columns if not c.startswith("cat_")]

for col in feature_columns:
    med = float(train_processed[col].median())
    train_processed[col] = train_processed[col].fillna(med)
    if col in test_processed.columns:
        test_processed[col] = test_processed[col].fillna(med)

# =========================================================================
# 2. Model Definitions
# =========================================================================


def get_lgb_params(num_classes: int, seed: int = 42) -> dict:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 15,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def get_xgb_params(num_classes: int, seed: int = 42) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "random_state": seed,
        "n_jobs": -1,
    }


class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.drop1(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.drop2(out)
        return residual + out


class TabularResNet(nn.Module):

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout_rate: float = 0.2,
        emb_dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for _, num_cats, emb_dim in categorical_cardinalities:
            emb = nn.Embedding(num_embeddings=num_cats + 2, embedding_dim=emb_dim)
            nn.init.kaiming_normal_(emb.weight, mode="fan_out")
            self.embeddings.append(emb)
            total_emb_dim += emb_dim

        self.emb_dropout = (
            nn.Dropout(emb_dropout) if total_emb_dim > 0 else nn.Identity()
        )
        self.num_numeric = num_numeric_features
        if num_numeric_features > 0:
            self.num_norm = nn.BatchNorm1d(num_numeric_features)
            self.num_proj = nn.Linear(num_numeric_features, hidden_dim)
        else:
            self.num_norm = nn.Identity()
            self.num_proj = None

        input_dim = total_emb_dim + (hidden_dim if num_numeric_features > 0 else 0)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim=hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head_act = nn.SiLU()
        self.head_drop = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor = None) -> torch.Tensor:
        emb_outs = []
        if x_cat is not None and len(self.embeddings) > 0:
            for i, emb_layer in enumerate(self.embeddings):
                cat_col = x_cat[:, i].long()
                emb_outs.append(emb_layer(cat_col))
            embs = torch.cat(emb_outs, dim=1)
            embs = self.emb_dropout(embs)
        else:
            embs = None

        if self.num_numeric > 0 and x_num is not None:
            normed_num = self.num_norm(x_num)
            proj_num = F.silu(self.num_proj(normed_num))
            if embs is not None:
                combined = torch.cat([embs, proj_num], dim=1)
            else:
                combined = proj_num
        else:
            combined = embs

        h = F.silu(self.input_proj(combined))
        for block in self.blocks:
            h = block(h)

        h = self.head_norm(h)
        h = self.head_act(h)
        h = self.head_drop(h)
        return self.classifier(h)


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction="mean")
        return (self.smoothing / c) * loss + (1 - self.smoothing) * nll


def get_criterion(smoothing: float = 0.05) -> nn.Module:
    if smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    return nn.CrossEntropyLoss()


def get_nn_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    epochs: int = 35,
    min_lr: float = 1e-5,
):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    return optimizer, scheduler


# =========================================================================
# 3. Model Training & Cross-Validation
# =========================================================================

n_splits = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_all = train_processed[feature_columns].values.astype(np.float32)
y_all = train_processed["target_class"].values.astype(np.int64)
folds = train_processed["fold"].values
X_test = test_processed[feature_columns].values.astype(np.float32)

num_idx = [feature_columns.index(c) for c in numeric_columns]
cat_idx = [feature_columns.index(c) for c in categorical_columns]

cat_cardinalities = []
for c in categorical_columns:
    max_c = int(max(train_processed[c].max(), test_processed[c].max())) + 2
    emb_dim = min(32, max(4, int(round(max_c**0.5))))
    cat_cardinalities.append((c, max_c, emb_dim))

oof_lgb = np.zeros((len(train_processed), num_classes), dtype=np.float32)
test_lgb = np.zeros((len(test_processed), num_classes), dtype=np.float32)

oof_xgb = np.zeros((len(train_processed), num_classes), dtype=np.float32)
test_xgb = np.zeros((len(test_processed), num_classes), dtype=np.float32)

oof_nn = np.zeros((len(train_processed), num_classes), dtype=np.float32)
test_nn = np.zeros((len(test_processed), num_classes), dtype=np.float32)

for fold_idx in range(n_splits):
    tr_mask = folds != fold_idx
    va_mask = folds == fold_idx

    X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
    X_va, y_va = X_all[va_mask], y_all[va_mask]

    # --- LightGBM ---
    lgb_params = get_lgb_params(num_classes=num_classes, seed=42 + fold_idx)
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    gbm = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=800,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[va_mask] = gbm.predict(X_va, num_iteration=gbm.best_iteration)
    test_lgb += (
        gbm.predict(X_test, num_iteration=gbm.best_iteration) / n_splits
    ).astype(np.float32)

    # --- XGBoost ---
    xgb_params = get_xgb_params(num_classes=num_classes, seed=42 + fold_idx)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    dtest = xgb.DMatrix(X_test)

    bst = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=800,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    oof_xgb[va_mask] = bst.predict(dval)
    test_xgb += (bst.predict(dtest) / n_splits).astype(np.float32)

    # --- Deep Tabular ResNet ---
    nn_model = TabularResNet(
        num_numeric_features=len(num_idx),
        categorical_cardinalities=cat_cardinalities,
        num_classes=num_classes,
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.2,
        emb_dropout=0.1,
    ).to(device)

    criterion = get_criterion(smoothing=0.05)
    epochs = 35
    batch_size = 64
    optimizer, scheduler = get_nn_optimizer_and_scheduler(
        nn_model, lr=2e-3, weight_decay=1e-4, epochs=epochs
    )

    tr_num = torch.tensor(X_tr[:, num_idx], dtype=torch.float32)
    tr_cat = torch.tensor(X_tr[:, cat_idx], dtype=torch.int64)
    tr_y_t = torch.tensor(y_tr, dtype=torch.int64)

    va_num = torch.tensor(X_va[:, num_idx], dtype=torch.float32).to(device)
    va_cat = torch.tensor(X_va[:, cat_idx], dtype=torch.int64).to(device)
    va_y_t = torch.tensor(y_va, dtype=torch.int64).to(device)

    te_num = torch.tensor(X_test[:, num_idx], dtype=torch.float32).to(device)
    te_cat = torch.tensor(X_test[:, cat_idx], dtype=torch.int64).to(device)

    train_ds = TensorDataset(tr_num, tr_cat, tr_y_t)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_val_loss = float("inf")
    best_nn_preds = None
    best_test_fold_preds = None

    for epoch in range(epochs):
        nn_model.train()
        for b_num, b_cat, b_y in train_loader:
            b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(b_num, b_cat)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_logits = nn_model(va_num, va_cat)
            val_loss = nn.CrossEntropyLoss()(val_logits, va_y_t).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_nn_preds = (
                    torch.softmax(val_logits, dim=-1).cpu().numpy().astype(np.float32)
                )
                test_logits = nn_model(te_num, te_cat)
                best_test_fold_preds = (
                    torch.softmax(test_logits, dim=-1).cpu().numpy().astype(np.float32)
                )

    oof_nn[va_mask] = best_nn_preds
    test_nn += best_test_fold_preds / n_splits

    fold_acc_lgb = accuracy_score(y_va, np.argmax(oof_lgb[va_mask], axis=1))
    fold_acc_xgb = accuracy_score(y_va, np.argmax(oof_xgb[va_mask], axis=1))
    fold_acc_nn = accuracy_score(y_va, np.argmax(oof_nn[va_mask], axis=1))
    print(
        f"Fold {fold_idx + 1}/{n_splits} | LGB Acc: {fold_acc_lgb:.4f} | XGB Acc: {fold_acc_xgb:.4f} | ResNet Acc: {fold_acc_nn:.4f}"
    )

# =========================================================================
# 4. Ensemble Weight Optimization & Final Predictions
# =========================================================================


def loss_fn(weights):
    w1, w2, w3 = weights
    w_sum = max(w1 + w2 + w3, 1e-8)
    w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum
    blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_nn
    eps = 1e-7
    blend_norm = np.clip(blend, eps, 1.0 - eps)
    blend_norm = blend_norm / blend_norm.sum(axis=1, keepdims=True)
    ce = -np.mean(np.log(blend_norm[np.arange(len(y_all)), y_all]))
    return ce


opt_res = minimize(
    loss_fn,
    [0.4, 0.4, 0.2],
    method="Nelder-Mead",
    bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
)
opt_w = opt_res.x
opt_w = opt_w / np.sum(opt_w)

oof_blend = opt_w[0] * oof_lgb + opt_w[1] * oof_xgb + opt_w[2] * oof_nn
oof_preds = np.argmax(oof_blend, axis=1)
final_val_accuracy = float(accuracy_score(y_all, oof_preds))

test_blend = opt_w[0] * test_lgb + opt_w[1] * test_xgb + opt_w[2] * test_nn
test_preds = np.argmax(test_blend, axis=1)
test_ratings = [class_to_rating[int(c)] for c in test_preds]

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_processed), dtype=int), "Rating": test_ratings}
)

submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Final Validation Score: {final_val_accuracy}")
