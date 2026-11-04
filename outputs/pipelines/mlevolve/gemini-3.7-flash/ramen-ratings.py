import json
import os
import random
import re
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 0. Deterministic Seed Setup
# ---------------------------------------------------------
RANDOM_STATE = 42


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(RANDOM_STATE)

# ---------------------------------------------------------
# 1. Setup Directories and Load Raw Data
# ---------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Ensure ID columns are preserved
if "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df))
if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))

target_col = "Stars"
for col in ["Brand", "Style", "Country", "Variety"]:
    if col not in train_df.columns:
        train_df[col] = "Missing"
    train_df[col] = train_df[col].fillna("Missing").astype(str)
    if col not in test_df.columns:
        test_df[col] = "Missing"
    test_df[col] = test_df[col].fillna("Missing").astype(str)

if "Top Ten" in train_df.columns:
    train_df["Top Ten"] = train_df["Top Ten"].fillna("Missing").astype(str)
if "Top Ten" in test_df.columns:
    test_df["Top Ten"] = test_df["Top Ten"].fillna("Missing").astype(str)
categorical_cols = [c for c in ["Brand", "Style", "Country"] if c in train_df.columns]

# ---------------------------------------------------------
# 2. Target Encoding and Label Preparation
# ---------------------------------------------------------
train_df[target_col] = train_df[target_col].astype(str).str.strip()

# Remove invalid target rows if any exist
valid_mask = ~train_df[target_col].isin(["", "nan", "None", "Unrated", "NaN"])
train_df = train_df[valid_mask].reset_index(drop=True)

label_encoder = LabelEncoder()
train_df["target"] = label_encoder.fit_transform(train_df[target_col])
num_classes = len(label_encoder.classes_)
np.save("./working/target_classes.npy", label_encoder.classes_)

# ---------------------------------------------------------
# 3. Stratified Split and Cross-Validation Strategy
# ---------------------------------------------------------
class_counts = train_df["target"].value_counts()
rare_classes = class_counts[class_counts < 2].index
strat_target = train_df["target"].copy()
if len(rare_classes) > 0:
    most_common_class = class_counts.index[0]
    strat_target[strat_target.isin(rare_classes)] = most_common_class

train_idx, val_idx = train_test_split(
    train_df.index, test_size=0.20, random_state=RANDOM_STATE, stratify=strat_target
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
folds = np.zeros(len(train_df), dtype=int)
for fold, (_, f_val_idx) in enumerate(skf.split(train_df, strat_target)):
    folds[f_val_idx] = fold
train_df["fold"] = folds

# ---------------------------------------------------------
# 4. Feature Engineering Functions
# ---------------------------------------------------------
FLAVOR_KEYWORDS = [
    "spicy",
    "hot",
    "chili",
    "miso",
    "shoyu",
    "tonkotsu",
    "pork",
    "chicken",
    "beef",
    "seafood",
    "shrimp",
    "curry",
    "kimchi",
    "tom yum",
    "laksa",
    "sesame",
    "garlic",
    "onion",
    "vegetable",
    "veggie",
    "mushroom",
    "duck",
    "cheese",
    "black garlic",
    "sour",
    "sweet",
    "creamy",
    "rich",
    "oriental",
    "chow mein",
    "yakisoba",
    "udon",
    "soba",
    "vermicelli",
    "rice",
    "fried",
    "artificial",
    "flavor",
    "flavour",
    "sauce",
    "soup",
    "instant",
    "bowl",
    "cup",
    "pack",
    "tray",
    "premium",
    "gourmet",
    "special",
    "mega",
    "mini",
]


def extract_text_and_domain_features(df):
    feats = pd.DataFrame(index=df.index)
    variety_text = df["Variety"].fillna("").astype(str).str.lower()

    feats["variety_char_len"] = variety_text.str.len()
    feats["variety_word_count"] = variety_text.apply(lambda s: len(s.split()))
    feats["variety_avg_word_len"] = feats["variety_char_len"] / (
        feats["variety_word_count"] + 1e-5
    )
    feats["variety_digit_count"] = variety_text.apply(
        lambda s: sum(c.isdigit() for c in s)
    )
    feats["variety_upper_count"] = (
        df["Variety"].fillna("").apply(lambda s: sum(c.isupper() for c in s))
    )
    feats["variety_upper_ratio"] = feats["variety_upper_count"] / (
        feats["variety_char_len"] + 1e-5
    )

    for kw in FLAVOR_KEYWORDS:
        col_name = f"kw_{kw.replace(' ', '_')}"
        feats[col_name] = variety_text.apply(lambda s: int(kw in s))

    if "Top Ten" in df.columns:
        top_ten_str = df["Top Ten"].fillna("").astype(str).str.lower()
        feats["is_top_ten"] = top_ten_str.apply(
            lambda s: 0 if s in ["", "missing", "nan", "\n", "none"] else 1
        )
        feats["top_ten_year"] = top_ten_str.apply(
            lambda s: (
                int(re.search(r"(20\d\d)", s).group(1)) if re.search(r"(20\d\d)", s) else 0
            )
        )
        feats["top_ten_rank"] = top_ten_str.apply(
            lambda s: (
                int(re.search(r"#(\d+)", s).group(1)) if re.search(r"#(\d+)", s) else 0
            )
        )
    else:
        feats["is_top_ten"] = 0
        feats["top_ten_year"] = 0
        feats["top_ten_rank"] = 0
    return feats


def create_interaction_features(df):
    feats = pd.DataFrame(index=df.index)
    brand = df["Brand"].fillna("Missing").astype(str)
    country = df["Country"].fillna("Missing").astype(str)
    style = df["Style"].fillna("Missing").astype(str)

    feats["brand_country"] = brand + "__" + country
    feats["brand_style"] = brand + "__" + style
    feats["style_country"] = style + "__" + country
    return feats


train_domain_feats = extract_text_and_domain_features(train_df)
test_domain_feats = extract_text_and_domain_features(test_df)

train_inter_cat = create_interaction_features(train_df)
test_inter_cat = create_interaction_features(test_df)

# ---------------------------------------------------------
# 5. Frequency & Target Encodings (Strict Isolation)
# ---------------------------------------------------------
all_cats = [
    "Brand",
    "Style",
    "Country",
    "brand_country",
    "brand_style",
    "style_country",
]

train_cat_all = pd.concat(
    [train_df[["Brand", "Style", "Country"]], train_inter_cat], axis=1
)
test_cat_all = pd.concat(
    [test_df[["Brand", "Style", "Country"]], test_inter_cat], axis=1
)

freq_train = pd.DataFrame(index=train_df.index)
freq_test = pd.DataFrame(index=test_df.index)

for col in all_cats:
    counts = train_cat_all.loc[train_idx, col].value_counts()
    freq_train[f"{col}_freq"] = train_cat_all[col].map(counts).fillna(0).astype(float)
    freq_test[f"{col}_freq"] = test_cat_all[col].map(counts).fillna(0).astype(float)

target_stats_train = pd.DataFrame(index=train_df.index)
target_stats_test = pd.DataFrame(index=test_df.index)

for col in ["Brand", "Country", "Style", "brand_country"]:
    oof_mean = pd.Series(index=train_df.index, dtype=float)
    oof_std = pd.Series(index=train_df.index, dtype=float)
    oof_min = pd.Series(index=train_df.index, dtype=float)
    oof_max = pd.Series(index=train_df.index, dtype=float)

    for fold_num in range(5):
        tr_f = train_df["fold"] != fold_num
        va_f = train_df["fold"] == fold_num

        tr_data = train_df.loc[tr_f]
        tr_cats = train_cat_all.loc[tr_f, col]
        va_cats = train_cat_all.loc[va_f, col]

        grouped = tr_data.groupby(tr_cats)["target"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)
        mins = grouped.min()
        maxs = grouped.max()
        counts = tr_cats.value_counts()

        global_mean = tr_data["target"].mean()
        global_std = tr_data["target"].std()
        global_min = tr_data["target"].min()
        global_max = tr_data["target"].max()

        smoothed_mean = (means * counts + global_mean * 10) / (counts + 10)
        smoothed_std = (stds * counts + global_std * 10) / (counts + 10)

        oof_mean.loc[va_f] = va_cats.map(smoothed_mean).fillna(global_mean)
        oof_std.loc[va_f] = va_cats.map(smoothed_std).fillna(global_std)
        oof_min.loc[va_f] = va_cats.map(mins).fillna(global_min)
        oof_max.loc[va_f] = va_cats.map(maxs).fillna(global_max)

    target_stats_train[f"{col}_te_mean"] = oof_mean
    target_stats_train[f"{col}_te_std"] = oof_std
    target_stats_train[f"{col}_te_min"] = oof_min
    target_stats_train[f"{col}_te_max"] = oof_max

    full_grouped = train_df.groupby(train_cat_all[col])["target"]
    full_means = full_grouped.mean()
    full_stds = full_grouped.std().fillna(0)
    full_mins = full_grouped.min()
    full_maxs = full_grouped.max()
    full_counts = train_cat_all[col].value_counts()

    full_global_mean = train_df["target"].mean()
    full_global_std = train_df["target"].std()
    full_global_min = train_df["target"].min()
    full_global_max = train_df["target"].max()

    full_smoothed_mean = (full_means * full_counts + full_global_mean * 10) / (
        full_counts + 10
    )
    full_smoothed_std = (full_stds * full_counts + full_global_std * 10) / (
        full_counts + 10
    )

    target_stats_test[f"{col}_te_mean"] = (
        test_cat_all[col].map(full_smoothed_mean).fillna(full_global_mean).astype(float)
    )
    target_stats_test[f"{col}_te_std"] = (
        test_cat_all[col].map(full_smoothed_std).fillna(full_global_std).astype(float)
    )
    target_stats_test[f"{col}_te_min"] = (
        test_cat_all[col].map(full_mins).fillna(full_global_min).astype(float)
    )
    target_stats_test[f"{col}_te_max"] = (
        test_cat_all[col].map(full_maxs).fillna(full_global_max).astype(float)
    )

target_means_train = target_stats_train
target_means_test = target_stats_test

# ---------------------------------------------------------
# 6. TF-IDF and Contextual Sentence Embeddings
# ---------------------------------------------------------
train_df["composite_text"] = (
    "Brand: "
    + train_df["Brand"]
    + " | Country: "
    + train_df["Country"]
    + " | Style: "
    + train_df["Style"]
    + " | Variety: "
    + train_df["Variety"]
)
test_df["composite_text"] = (
    "Brand: "
    + test_df["Brand"]
    + " | Country: "
    + test_df["Country"]
    + " | Style: "
    + test_df["Style"]
    + " | Variety: "
    + test_df["Variety"]
)

# Multi-view TF-IDF (Word + Char)
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=1500,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    lowercase=True,
)
tfidf_char = TfidfVectorizer(
    ngram_range=(2, 5),
    analyzer="char_wb",
    max_features=1500,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    lowercase=True,
)

tfidf_w_tr = tfidf_word.fit_transform(train_df["composite_text"])
tfidf_w_te = tfidf_word.transform(test_df["composite_text"])

tfidf_c_tr = tfidf_char.fit_transform(train_df["Variety"])
tfidf_c_te = tfidf_char.transform(test_df["Variety"])

from scipy.sparse import hstack

tfidf_combined_tr = hstack([tfidf_w_tr, tfidf_c_tr])
tfidf_combined_te = hstack([tfidf_w_te, tfidf_c_te])

n_svd_dim = 32
svd = TruncatedSVD(n_components=n_svd_dim, random_state=RANDOM_STATE)
svd_train = svd.fit_transform(tfidf_combined_tr)
svd_test = svd.transform(tfidf_combined_te)

svd_cols = [f"svd_feat_{i}" for i in range(n_svd_dim)]
svd_train_df = pd.DataFrame(svd_train, columns=svd_cols, index=train_df.index)
svd_test_df = pd.DataFrame(svd_test, columns=svd_cols, index=test_df.index)

# Transformer Sentence Embeddings Extraction
transformer_embeddings_extracted = False
try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model.eval()

    def compute_embeddings(text_series, b_size=64):
        emb_list = []
        texts = list(text_series)
        for idx in range(0, len(texts), b_size):
            b_texts = texts[idx : idx + b_size]
            encoded = tokenizer(
                b_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                emb_list.append(pooled.cpu().numpy())
        return np.vstack(emb_list)

    emb_train = compute_embeddings(train_df["composite_text"])
    emb_test = compute_embeddings(test_df["composite_text"])

    emb_svd = TruncatedSVD(n_components=24, random_state=RANDOM_STATE)
    emb_train_svd = emb_svd.fit_transform(emb_train)
    emb_test_svd = emb_svd.transform(emb_test)

    emb_cols = [f"trans_emb_{i}" for i in range(24)]
    emb_train_df = pd.DataFrame(emb_train_svd, columns=emb_cols, index=train_df.index)
    emb_test_df = pd.DataFrame(emb_test_svd, columns=emb_cols, index=test_df.index)

    svd_train_df = pd.concat([svd_train_df, emb_train_df], axis=1)
    svd_test_df = pd.concat([svd_test_df, emb_test_df], axis=1)
    transformer_embeddings_extracted = True
    print("Transformer contextual embeddings successfully generated.")
except Exception as exc:
    print(f"Transformers embedding fallback active ({exc}).")

# ---------------------------------------------------------
# 7. Categorical Ordinal Encoding
# ---------------------------------------------------------
cat_encoded_train = pd.DataFrame(index=train_df.index)
cat_encoded_test = pd.DataFrame(index=test_df.index)

for col in all_cats:
    le = LabelEncoder()
    le.fit(train_cat_all.loc[train_idx, col].astype(str))

    train_cat_encoded = (
        train_cat_all[col]
        .astype(str)
        .map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    )
    test_cat_encoded = (
        test_cat_all[col]
        .astype(str)
        .map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    )

    cat_encoded_train[f"{col}_cat_code"] = train_cat_encoded
    cat_encoded_test[f"{col}_cat_code"] = test_cat_encoded

# ---------------------------------------------------------
# 8. Combine Features & Scale Numeric Features
# ---------------------------------------------------------
X_train_full = pd.concat(
    [
        train_domain_feats,
        freq_train,
        target_means_train,
        svd_train_df,
        cat_encoded_train,
    ],
    axis=1,
)

X_test = pd.concat(
    [test_domain_feats, freq_test, target_means_test, svd_test_df, cat_encoded_test],
    axis=1,
)

numeric_feature_cols = [
    col for col in X_train_full.columns if not col.endswith("_cat_code")
]
categorical_feature_cols = [
    col for col in X_train_full.columns if col.endswith("_cat_code")
]

scaler = StandardScaler()
scaler.fit(X_train_full.loc[train_idx, numeric_feature_cols])

X_train_full[numeric_feature_cols] = scaler.transform(
    X_train_full[numeric_feature_cols]
)
X_test[numeric_feature_cols] = scaler.transform(X_test[numeric_feature_cols])

X_train_full["id"] = train_df["id"]
X_train_full["target"] = train_df["target"]
X_train_full["fold"] = train_df["fold"]
X_test["id"] = test_df["id"]

train_processed = X_train_full.loc[train_idx].copy().reset_index(drop=True)
val_processed = X_train_full.loc[val_idx].copy().reset_index(drop=True)
test_processed = X_test.copy().reset_index(drop=True)

feature_list = [
    col for col in train_processed.columns if col not in ["id", "target", "fold"]
]
with open("./working/feature_names.json", "w") as f:
    json.dump(
        {
            "feature_names": feature_list,
            "categorical_features": categorical_feature_cols,
            "numeric_features": numeric_feature_cols,
            "num_classes": int(num_classes),
        },
        f,
        indent=2,
    )


# ---------------------------------------------------------
# 9. 5-Fold Stratified CV Multi-Model Ensemble (LGBM, XGB, CatBoost, MLP)
# ---------------------------------------------------------
n_splits = 5
oof_lgb_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
oof_xgb_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
oof_cat_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)
oof_mlp_probs = np.zeros((len(train_df), num_classes), dtype=np.float64)

test_lgb_probs = np.zeros((len(test_df), num_classes), dtype=np.float64)
test_xgb_probs = np.zeros((len(test_df), num_classes), dtype=np.float64)
test_cat_probs = np.zeros((len(test_df), num_classes), dtype=np.float64)
test_mlp_probs = np.zeros((len(test_df), num_classes), dtype=np.float64)

y_all = train_df["target"].values
X_all = X_train_full[feature_list].copy()
X_test_feats = X_test[feature_list].copy()

print(f"Training feature set dimension: {X_all.shape}, num_classes: {num_classes}")

for fold in range(n_splits):
    train_mask = (train_df["fold"] != fold).values
    val_mask = (train_df["fold"] == fold).values

    X_tr, y_tr = X_all.iloc[train_mask], y_all[train_mask]
    X_va, y_va = X_all.iloc[val_mask], y_all[val_mask]

    # --- Model 1: Multi-class LightGBM ---
    lgb_clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        boosting_type="gbdt",
        colsample_bytree=0.70,
        subsample=0.85,
        subsample_freq=1,
        learning_rate=0.03,
        n_estimators=1000,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    lgb_val_pred = lgb_clf.predict_proba(X_va)
    oof_lgb_probs[val_mask] = lgb_val_pred
    test_lgb_probs += lgb_clf.predict_proba(X_test_feats) / n_splits
    lgb_fold_acc = accuracy_score(y_va, np.argmax(lgb_val_pred, axis=1))

    # --- Model 2: Multi-class XGBoost ---
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        max_depth=5,
        colsample_bytree=0.70,
        subsample=0.85,
        learning_rate=0.03,
        n_estimators=1000,
        early_stopping_rounds=50,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    xgb_val_pred = xgb_clf.predict_proba(X_va)
    oof_xgb_probs[val_mask] = xgb_val_pred
    test_xgb_probs += xgb_clf.predict_proba(X_test_feats) / n_splits
    xgb_fold_acc = accuracy_score(y_va, np.argmax(xgb_val_pred, axis=1))

    # --- Model 3: Multi-class CatBoost ---
    cat_clf = CatBoostClassifier(
        iterations=900,
        learning_rate=0.04,
        depth=5,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=RANDOM_STATE + fold,
        verbose=False,
        early_stopping_rounds=50,
        thread_count=-1,
    )
    cat_clf.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        verbose=False,
    )
    cat_val_pred = cat_clf.predict_proba(X_va)
    oof_cat_probs[val_mask] = cat_val_pred
    test_cat_probs += cat_clf.predict_proba(X_test_feats) / n_splits
    cat_fold_acc = accuracy_score(y_va, np.argmax(cat_val_pred, axis=1))

    # --- Model 4: Multi-class Regularized Neural MLP ---
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=RANDOM_STATE + fold,
    )
    mlp_clf.fit(X_tr, y_tr)
    mlp_val_pred = mlp_clf.predict_proba(X_va)
    oof_mlp_probs[val_mask] = mlp_val_pred
    test_mlp_probs += mlp_clf.predict_proba(X_test_feats) / n_splits
    mlp_fold_acc = accuracy_score(y_va, np.argmax(mlp_val_pred, axis=1))

    fold_blend = (
        0.30 * lgb_val_pred
        + 0.30 * xgb_val_pred
        + 0.25 * cat_val_pred
        + 0.15 * mlp_val_pred
    )
    blend_fold_acc = accuracy_score(y_va, np.argmax(fold_blend, axis=1))

    print(
        f"Fold {fold+1}/{n_splits} | LGB: {lgb_fold_acc:.4f} | XGB: {xgb_fold_acc:.4f} | CAT: {cat_fold_acc:.4f} | MLP: {mlp_fold_acc:.4f} | Blend: {blend_fold_acc:.4f}"
    )

# ---------------------------------------------------------
# 10. Out-of-Fold Evaluation and Optimal Ensemble Blending
# ---------------------------------------------------------
lgb_oof_acc = accuracy_score(y_all, np.argmax(oof_lgb_probs, axis=1))
xgb_oof_acc = accuracy_score(y_all, np.argmax(oof_xgb_probs, axis=1))
cat_oof_acc = accuracy_score(y_all, np.argmax(oof_cat_probs, axis=1))
mlp_oof_acc = accuracy_score(y_all, np.argmax(oof_mlp_probs, axis=1))

model_oof_list = [oof_lgb_probs, oof_xgb_probs, oof_cat_probs, oof_mlp_probs]
model_test_list = [test_lgb_probs, test_xgb_probs, test_cat_probs, test_mlp_probs]
model_names = ["LightGBM", "XGBoost", "CatBoost", "MLP"]

# Simplex Weight Optimization on OOF predictions
def optimize_blend_weights(oof_list, targets):
    n_models = len(oof_list)

    def objective(w_logits):
        w = np.exp(w_logits) / np.sum(np.exp(w_logits))
        blended = sum(w[i] * oof_list[i] for i in range(n_models))
        preds = np.argmax(blended, axis=1)
        return -accuracy_score(targets, preds)

    initial_weights = np.zeros(n_models)
    opt_res = minimize(objective, initial_weights, method="Nelder-Mead", options={"maxiter": 600})
    best_weights = np.exp(opt_res.x) / np.sum(np.exp(opt_res.x))
    return best_weights

opt_weights = optimize_blend_weights(model_oof_list, y_all)

oof_optimal_blend = sum(opt_weights[i] * model_oof_list[i] for i in range(len(model_oof_list)))
ensemble_oof_acc = accuracy_score(y_all, np.argmax(oof_optimal_blend, axis=1))

print("=" * 60)
for name, acc, w in zip(model_names, [lgb_oof_acc, xgb_oof_acc, cat_oof_acc, mlp_oof_acc], opt_weights):
    print(f"Model: {name:10s} | OOF Acc: {acc:.4f} | Optimal Weight: {w:.4f}")
print(f"Optimal Blended Ensemble OOF Accuracy: {ensemble_oof_acc:.4f}")
print("=" * 60)

# ---------------------------------------------------------
# 11. Final Test Inference and Submission Generation
# ---------------------------------------------------------
test_ensemble_probs = sum(
    opt_weights[i] * model_test_list[i] for i in range(len(model_test_list))
)
test_pred_indices = np.argmax(test_ensemble_probs, axis=1)
test_stars_pred = label_encoder.inverse_transform(test_pred_indices)

submission_df = pd.DataFrame(
    {"id": test_df["id"].values, "Stars": test_stars_pred}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

assert os.path.exists(submission_path), "Submission file was not created!"
assert len(submission_df) == len(test_df), "Submission row count mismatch!"
assert list(submission_df.columns) == ["id", "Stars"], "Submission format mismatch!"

final_val_score = ensemble_oof_acc
print(f"Final Validation Score: {final_val_score}")
