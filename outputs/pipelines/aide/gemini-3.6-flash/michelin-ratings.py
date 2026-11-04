import os
import re
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "awardFor"
y_train = train_df[target_col].values
train_df = train_df.drop(columns=[target_col])

n_train = len(train_df)
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

features_list = []

# 2. Identify Column Types
num_cols = []
cat_cols = []
text_cols = []

for col in combined_df.columns:
    if col.lower() in ["id", "index"]:
        continue
    dtype = combined_df[col].dtype
    if np.issubdtype(dtype, np.number):
        num_cols.append(col)
    else:
        non_null = combined_df[col].dropna().astype(str)
        if len(non_null) > 0 and non_null.str.len().mean() > 25:
            text_cols.append(col)
        else:
            cat_cols.append(col)

# 3. Domain Specific Feature Engineering: Price & Coordinates
domain_feats = []
domain_cols = []

# Price parsing
price_col = [c for c in combined_df.columns if "price" in c.lower()]
num_price_vec = None
if price_col:
    p_col = price_col[0]
    p_str = combined_df[p_col].astype(str).fillna("")

    sym_count = p_str.apply(
        lambda x: len(re.findall(r"[\$\€\£\¥\₹]", x))
    ).values.reshape(-1, 1)

    def parse_num(x):
        nums = re.findall(r"\d+", x)
        return float(nums[0]) if nums else np.nan

    num_price = p_str.apply(parse_num).values.astype(np.float32)
    med_p = np.nanmedian(num_price)
    num_price_vec = np.nan_to_num(num_price, nan=med_p if not np.isnan(med_p) else 0.0)

    domain_feats.append(sym_count)
    domain_feats.append(num_price_vec.reshape(-1, 1))
    domain_cols.extend(["price_symbol_count", "price_numeric_val"])

# Lat / Lon spatial features
coords_pca = None
lat_cols = [c for c in combined_df.columns if "lat" in c.lower()]
lon_cols = [c for c in combined_df.columns if "lon" in c.lower() or "lng" in c.lower()]

if lat_cols and lon_cols:
    lat_val = combined_df[lat_cols[0]].values.astype(np.float32)
    lon_val = combined_df[lon_cols[0]].values.astype(np.float32)

    lat_val = np.nan_to_num(lat_val, nan=np.nanmedian(lat_val))
    lon_val = np.nan_to_num(lon_val, nan=np.nanmedian(lon_val))

    coords = np.column_stack([lat_val, lon_val])
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(coords)

    domain_feats.append(coords_pca)
    domain_cols.extend(["coord_pca_1", "coord_pca_2"])

if domain_feats:
    domain_mat = np.hstack(domain_feats)
    features_list.append(pd.DataFrame(domain_mat, columns=domain_cols))

# 4. Group Aggregation Features (Contextual Baselines)
group_agg_list = []
df_grp = combined_df.copy()
if num_price_vec is not None:
    df_grp["_price_num"] = num_price_vec
if coords_pca is not None:
    df_grp["_coord_pca_1"] = coords_pca[:, 0]
    df_grp["_coord_pca_2"] = coords_pca[:, 1]

for col in cat_cols:
    s = df_grp[col].astype(str).fillna("missing")
    n_uniq = s.nunique()
    if 2 <= n_uniq <= 500:
        freq = s.map(s.value_counts()).values.reshape(-1, 1)
        group_agg_list.append(pd.DataFrame(freq, columns=[f"grp_cnt_{col}"]))
        for target_agg in ["_price_num", "_coord_pca_1", "_coord_pca_2"]:
            if target_agg in df_grp.columns:
                g_mean = df_grp.groupby(s)[target_agg].transform("mean").values
                diff = df_grp[target_agg].values - g_mean
                group_agg_list.append(
                    pd.DataFrame(
                        np.column_stack([g_mean, diff]),
                        columns=[
                            f"grp_{col}_{target_agg}_mean",
                            f"grp_{col}_{target_agg}_diff",
                        ],
                    )
                )

if group_agg_list:
    features_list.append(pd.concat(group_agg_list, axis=1))

# 5. Numerical Features Processing
for col in num_cols:
    val = combined_df[col].values.astype(np.float32).reshape(-1, 1)
    median_val = np.nanmedian(val)
    val = np.nan_to_num(val, nan=median_val if not np.isnan(median_val) else 0.0)
    features_list.append(pd.DataFrame(val, columns=[f"num_{col}"]))

# 6. Categorical Features (Label & Frequency Encoding)
for col in cat_cols:
    s = combined_df[col].astype(str).fillna("missing")
    freq = s.map(s.value_counts(normalize=True)).values.reshape(-1, 1)
    le_cat = LabelEncoder()
    encoded = le_cat.fit_transform(s).reshape(-1, 1)
    features_list.append(
        pd.DataFrame(
            np.hstack([encoded, freq]), columns=[f"cat_le_{col}", f"cat_freq_{col}"]
        )
    )

# 7. Multi-Scale Text Processing (Word + Char TF-IDF + Dual SVD)
all_text_cols = text_cols if text_cols else cat_cols
if all_text_cols:
    combined_text = (
        combined_df[all_text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    )
    text_len = combined_text.str.len().values.reshape(-1, 1)
    word_count = combined_text.str.split().str.len().values.reshape(-1, 1)

    # Word-level TF-IDF & SVD
    tfidf_word = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 3), sublinear_tf=True
    )
    tfidf_word_mat = tfidf_word.fit_transform(combined_text)
    svd_word = TruncatedSVD(n_components=128, random_state=42)
    svd_word_feats = svd_word.fit_transform(tfidf_word_mat)

    # Character-level TF-IDF & SVD (captures morphological and foreign language nuances)
    tfidf_char = TfidfVectorizer(
        max_features=15000, ngram_range=(3, 5), analyzer="char_wb", sublinear_tf=True
    )
    tfidf_char_mat = tfidf_char.fit_transform(combined_text)
    svd_char = TruncatedSVD(n_components=64, random_state=42)
    svd_char_feats = svd_char.fit_transform(tfidf_char_mat)

    text_stat_feats = np.hstack([text_len, word_count, svd_word_feats, svd_char_feats])
    stat_cols = (
        ["text_len", "word_count"]
        + [f"svd_w_{i}" for i in range(128)]
        + [f"svd_c_{i}" for i in range(64)]
    )
    features_list.append(pd.DataFrame(text_stat_feats, columns=stat_cols))

# Assemble Feature Matrix & Standardize Unique Column Names
X_all = pd.concat(features_list, axis=1)
X_all.columns = [f"f_{i}_{col}" for i, col in enumerate(X_all.columns)]

X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

# Encode Target Column
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train)
num_classes = len(le_target.classes_)

# 8. Stratified 5-Fold Cross Validation & XGBoost + LightGBM Ensemble Training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((n_train, num_classes))
test_preds = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train_encoded[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train_encoded[val_idx]

    # XGBoost Model
    clf_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        random_state=42 + fold,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )
    clf_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # LightGBM Model
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=15,
        random_state=42 + fold,
        objective="multiclass",
        n_jobs=-1,
        verbose=-1,
    )
    clf_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    pred_xgb_va = clf_xgb.predict_proba(X_va)
    pred_lgb_va = clf_lgb.predict_proba(X_va)

    pred_xgb_te = clf_xgb.predict_proba(X_test)
    pred_lgb_te = clf_lgb.predict_proba(X_test)

    oof_preds[val_idx] = 0.5 * pred_xgb_va + 0.5 * pred_lgb_va
    test_preds += (0.5 * pred_xgb_te + 0.5 * pred_lgb_te) / skf.n_splits

oof_class_preds = np.argmax(oof_preds, axis=1)
oof_acc = accuracy_score(y_train_encoded, oof_class_preds)
print(f"Validation Accuracy: {oof_acc:.5f}")

# 9. Save Test Predictions
os.makedirs("./working", exist_ok=True)
test_class_preds = np.argmax(test_preds, axis=1)
test_pred_labels = le_target.inverse_transform(test_class_preds)

sub = pd.DataFrame(
    {"id": np.arange(len(test_pred_labels)), "awardFor": test_pred_labels}
)
sub.to_csv("./working/submission.csv", index=False)
