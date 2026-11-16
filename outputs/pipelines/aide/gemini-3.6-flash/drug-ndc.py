import os
import re
import warnings
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

n_train = len(train_df)
target_col = "dea_schedule"

# Handle target column
train_df[target_col] = train_df[target_col].fillna("NONE").astype(str)
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df[target_col])
num_classes = len(le_target.classes_)

# Combine data for unified feature engineering
df_all = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df],
    ignore_index=True,
)

# Identify column types
text_cols = [c for c in df_all.columns if df_all[c].dtype == "object" and c != "id"]
date_cols = [c for c in df_all.columns if "date" in c.lower() or "time" in c.lower()]

# 2. Text Feature Processing
for c in text_cols:
    df_all[c] = df_all[c].fillna("").astype(str)

full_text = df_all[text_cols].apply(lambda row: " ".join(row.values), axis=1)

# TF-IDF Vectorization for concatenated text
char_vec = TfidfVectorizer(
    ngram_range=(3, 5), max_features=20000, analyzer="char_wb", sublinear_tf=True
)
word_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=20000, sublinear_tf=True)

char_tfidf = char_vec.fit_transform(full_text)
word_tfidf = word_vec.fit_transform(full_text)

# Truncated SVD Features
svd_char = TruncatedSVD(n_components=40, random_state=42)
svd_word = TruncatedSVD(n_components=40, random_state=42)

char_svd_feats = svd_char.fit_transform(char_tfidf)
word_svd_feats = svd_word.fit_transform(word_tfidf)

svd_cols = [f"char_svd_{i}" for i in range(40)] + [f"word_svd_{i}" for i in range(40)]
svd_df = pd.DataFrame(np.hstack([char_svd_feats, word_svd_feats]), columns=svd_cols)

# Controlled Substance Keyword Flags
keywords = [
    "controlled",
    "schedule",
    "narcotic",
    "opioid",
    "benzodiazepine",
    "stimulant",
    "barbiturate",
    "cannabinoid",
    "sedative",
    "analgesic",
    "amphetamine",
    "codeine",
    "morphine",
    "fentanyl",
    "oxycodone",
    "hydrocodone",
    "alprazolam",
    "lorazepam",
    "zolpidem",
    "pregabalin",
    "buprenorphine",
    "ketamine",
    "tramadol",
    "tapentadol",
    "hydromorphone",
    "oxymorphone",
    "methadone",
    "diazepam",
    "clonazepam",
    "midazolam",
    "carisoprodol",
    "modafinil",
    "methylphenidate",
    "ephedrine",
    "pseudoephedrine",
]

kw_df = pd.DataFrame()
full_text_lower = full_text.str.lower()
for kw in keywords:
    kw_df[f"kw_{kw}"] = full_text_lower.str.contains(kw).astype(int)


# Regex Active Dosage Numeric Features
def extract_dosages(text):
    nums = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:mg|mcg|ug|g|ml|unit|%)", str(text), re.IGNORECASE
    )
    if not nums:
        return [0.0, 0.0, 0.0, 0.0]
    vals = [float(v) for v in nums]
    return [max(vals), min(vals), float(np.mean(vals)), float(len(vals))]


dosage_stats = full_text.apply(extract_dosages).tolist()
dosage_df = pd.DataFrame(
    dosage_stats, columns=["dosage_max", "dosage_min", "dosage_mean", "dosage_count"]
)

# 3. Date Features
date_df = pd.DataFrame()
for c in date_cols:
    dt_series = pd.to_datetime(df_all[c], errors="coerce")
    date_df[f"{c}_year"] = dt_series.dt.year.fillna(-1)
    date_df[f"{c}_month"] = dt_series.dt.month.fillna(-1)
    date_df[f"{c}_day"] = dt_series.dt.day.fillna(-1)
    date_df[f"{c}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1)

if len(date_cols) >= 2:
    dt1 = pd.to_datetime(df_all[date_cols[0]], errors="coerce")
    dt2 = pd.to_datetime(df_all[date_cols[1]], errors="coerce")
    date_df["date_diff_days"] = (dt2 - dt1).dt.days.fillna(-1)

# 4. Categorical Frequency Encodings & Label Encodings
cat_df = pd.DataFrame()
for c in text_cols:
    freq_map = df_all[c].value_counts(normalize=True).to_dict()
    cat_df[f"{c}_freq"] = df_all[c].map(freq_map).fillna(0)
    cat_df[f"{c}_le"] = LabelEncoder().fit_transform(df_all[c])

# 5. Out-of-Fold Multi-Class Target Encoding
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cat_cols_te = [c for c in text_cols if df_all[c].nunique() > 1]
alpha = 10.0  # Bayesian smoothing weight
global_priors = np.bincount(y_train, minlength=num_classes) / n_train

te_cols = [f"te_{col}_cls_{c}" for col in cat_cols_te for c in range(num_classes)]
te_oof = np.zeros((n_train, len(te_cols)), dtype=np.float32)
te_test = np.zeros((len(test_df), len(te_cols)), dtype=np.float32)

train_cats = df_all[cat_cols_te].iloc[:n_train].reset_index(drop=True)
test_cats = df_all[cat_cols_te].iloc[n_train:].reset_index(drop=True)

for col_idx, col in enumerate(cat_cols_te):
    test_col_te = np.zeros((len(test_df), num_classes), dtype=np.float32)

    for tr_idx, va_idx in skf.split(train_cats, y_train):
        tr_cat = train_cats[col].iloc[tr_idx]
        tr_y = y_train[tr_idx]
        va_cat = train_cats[col].iloc[va_idx]

        df_tr_col = pd.DataFrame({"cat": tr_cat.values, "target": tr_y})
        grouped = (
            df_tr_col.groupby("cat")["target"].value_counts().unstack(fill_value=0)
        )
        tot_counts = df_tr_col["cat"].value_counts()

        start_c = col_idx * num_classes
        for cls in range(num_classes):
            cls_counts = (
                grouped[cls]
                if cls in grouped.columns
                else pd.Series(0, index=tot_counts.index)
            )
            smoothed_prob = (cls_counts + alpha * global_priors[cls]) / (
                tot_counts + alpha
            )
            prob_dict = smoothed_prob.to_dict()

            va_mapped = va_cat.map(prob_dict).fillna(global_priors[cls]).values
            te_oof[va_idx, start_c + cls] = va_mapped

            test_mapped = (
                test_cats[col].map(prob_dict).fillna(global_priors[cls]).values
            )
            test_col_te[:, cls] += test_mapped / 5.0

    start_c = col_idx * num_classes
    te_test[:, start_c : start_c + num_classes] = test_col_te

te_train_df = pd.DataFrame(te_oof, columns=te_cols)
te_test_df = pd.DataFrame(te_test, columns=te_cols)
te_df = pd.concat([te_train_df, te_test_df], ignore_index=True)

# Combine Tabular Features
X_dense = pd.concat([svd_df, kw_df, dosage_df, date_df, cat_df, te_df], axis=1)

# 6. Global & Enhanced Column-Specific Out-of-Fold Linear Meta Features (LR, MNB, SGD, CNB)
tfidf_full = hstack([char_tfidf, word_tfidf]).tocsr()
tfidf_train = tfidf_full[:n_train]
tfidf_test = tfidf_full[n_train:]

oof_global_meta = np.zeros((n_train, num_classes * 4))
test_global_meta = np.zeros((len(test_df), num_classes * 4))

test_preds_lr = np.zeros((len(test_df), num_classes))
test_preds_mnb = np.zeros((len(test_df), num_classes))
test_preds_sgd = np.zeros((len(test_df), num_classes))
test_preds_cnb = np.zeros((len(test_df), num_classes))

for tr_idx, va_idx in skf.split(tfidf_train, y_train):
    X_tr_t, y_tr_t = tfidf_train[tr_idx], y_train[tr_idx]
    X_va_t, y_va_t = tfidf_train[va_idx], y_train[va_idx]

    # Global Logistic Regression
    lr = LogisticRegression(C=2.0, max_iter=500, random_state=42)
    lr.fit(X_tr_t, y_tr_t)
    oof_global_meta[va_idx, :num_classes] = lr.predict_proba(X_va_t)
    test_preds_lr += lr.predict_proba(tfidf_test) / 5.0

    # Global Multinomial Naive Bayes
    mnb = MultinomialNB(alpha=0.5)
    mnb.fit(X_tr_t, y_tr_t)
    oof_global_meta[va_idx, num_classes : num_classes * 2] = mnb.predict_proba(X_va_t)
    test_preds_mnb += mnb.predict_proba(tfidf_test) / 5.0

    # Global SGD Classifier (Log Loss)
    sgd = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, random_state=42)
    sgd.fit(X_tr_t, y_tr_t)
    oof_global_meta[va_idx, num_classes * 2 : num_classes * 3] = sgd.predict_proba(
        X_va_t
    )
    test_preds_sgd += sgd.predict_proba(tfidf_test) / 5.0

    # Global Complement Naive Bayes
    cnb = ComplementNB(alpha=0.5)
    cnb.fit(X_tr_t, y_tr_t)
    oof_global_meta[va_idx, num_classes * 3 :] = cnb.predict_proba(X_va_t)
    test_preds_cnb += cnb.predict_proba(tfidf_test) / 5.0

test_global_meta[:, :num_classes] = test_preds_lr
test_global_meta[:, num_classes : num_classes * 2] = test_preds_mnb
test_global_meta[:, num_classes * 2 : num_classes * 3] = test_preds_sgd
test_global_meta[:, num_classes * 3 :] = test_preds_cnb

global_meta_cols = [f"linear_meta_{i}" for i in range(num_classes * 4)]
global_meta_tr_df = pd.DataFrame(oof_global_meta, columns=global_meta_cols)
global_meta_te_df = pd.DataFrame(test_global_meta, columns=global_meta_cols)

# Column-specific OOF linear meta-features using combined Word + Char TF-IDF with LR, MNB, SGD & CNB
col_meta_tr_list = []
col_meta_te_list = []

for c in text_cols:
    if df_all[c].nunique() > 1:
        vec_word = TfidfVectorizer(
            ngram_range=(1, 2), max_features=5000, sublinear_tf=True
        )
        vec_char = TfidfVectorizer(
            ngram_range=(3, 5), analyzer="char_wb", max_features=5000, sublinear_tf=True
        )
        tfidf_word = vec_word.fit_transform(df_all[c])
        tfidf_char = vec_char.fit_transform(df_all[c])
        tfidf_col = hstack([tfidf_word, tfidf_char]).tocsr()

        tfidf_col_tr = tfidf_col[:n_train]
        tfidf_col_te = tfidf_col[n_train:]

        oof_col = np.zeros((n_train, num_classes * 4), dtype=np.float32)
        test_col = np.zeros((len(test_df), num_classes * 4), dtype=np.float32)

        for tr_idx, va_idx in skf.split(tfidf_col_tr, y_train):
            X_tr_c, y_tr_c = tfidf_col_tr[tr_idx], y_train[tr_idx]
            X_va_c, y_va_c = tfidf_col_tr[va_idx], y_train[va_idx]

            # Logistic Regression
            lr_col = LogisticRegression(C=2.0, max_iter=500, random_state=42)
            lr_col.fit(X_tr_c, y_tr_c)
            oof_col[va_idx, :num_classes] = lr_col.predict_proba(X_va_c)
            test_col[:, :num_classes] += lr_col.predict_proba(tfidf_col_te) / 5.0

            # Multinomial Naive Bayes
            mnb_col = MultinomialNB(alpha=0.5)
            mnb_col.fit(X_tr_c, y_tr_c)
            oof_col[va_idx, num_classes : num_classes * 2] = mnb_col.predict_proba(
                X_va_c
            )
            test_col[:, num_classes : num_classes * 2] += (
                mnb_col.predict_proba(tfidf_col_te) / 5.0
            )

            # SGD Classifier
            sgd_col = SGDClassifier(
                loss="log_loss", alpha=1e-4, max_iter=1000, random_state=42
            )
            sgd_col.fit(X_tr_c, y_tr_c)
            oof_col[va_idx, num_classes * 2 : num_classes * 3] = sgd_col.predict_proba(
                X_va_c
            )
            test_col[:, num_classes * 2 : num_classes * 3] += (
                sgd_col.predict_proba(tfidf_col_te) / 5.0
            )

            # Complement Naive Bayes
            cnb_col = ComplementNB(alpha=0.5)
            cnb_col.fit(X_tr_c, y_tr_c)
            oof_col[va_idx, num_classes * 3 :] = cnb_col.predict_proba(X_va_c)
            test_col[:, num_classes * 3 :] += cnb_col.predict_proba(tfidf_col_te) / 5.0

        col_names = (
            [f"meta_lr_{c}_cls_{cls}" for cls in range(num_classes)]
            + [f"meta_mnb_{c}_cls_{cls}" for cls in range(num_classes)]
            + [f"meta_sgd_{c}_cls_{cls}" for cls in range(num_classes)]
            + [f"meta_cnb_{c}_cls_{cls}" for cls in range(num_classes)]
        )
        col_meta_tr_list.append(pd.DataFrame(oof_col, columns=col_names))
        col_meta_te_list.append(pd.DataFrame(test_col, columns=col_names))

col_meta_tr_df = pd.concat(col_meta_tr_list, axis=1)
col_meta_te_df = pd.concat(col_meta_te_list, axis=1)

meta_tr_df = pd.concat([global_meta_tr_df, col_meta_tr_df], axis=1)
meta_te_df = pd.concat([global_meta_te_df, col_meta_te_df], axis=1)
meta_df = pd.concat([meta_tr_df, meta_te_df], ignore_index=True)

# Final Feature Sets
X_final = pd.concat([X_dense, meta_df], axis=1)
X_train = X_final.iloc[:n_train].reset_index(drop=True)
X_test = X_final.iloc[n_train:].reset_index(drop=True)

# Handle missing/inf values
X_train = X_train.fillna(-999).replace([np.inf, -np.inf], -999)
X_test = X_test.fillna(-999).replace([np.inf, -np.inf], -999)

# 7. Multi-Model Ensemble Training via 5-Fold Stratified K-Fold
oof_lgb = np.zeros((n_train, num_classes))
oof_xgb = np.zeros((n_train, num_classes))
oof_cb = np.zeros((n_train, num_classes))
oof_et = np.zeros((n_train, num_classes))
oof_mlp = np.zeros((n_train, num_classes))

test_lgb = np.zeros((len(test_df), num_classes))
test_xgb = np.zeros((len(test_df), num_classes))
test_cb = np.zeros((len(test_df), num_classes))
test_et = np.zeros((len(test_df), num_classes))
test_mlp = np.zeros((len(test_df), num_classes))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]

    # LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    oof_lgb[va_idx] = model_lgb.predict_proba(X_va)
    test_lgb += model_lgb.predict_proba(X_test) / 5.0

    # XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        eval_metric="mlogloss",
        early_stopping_rounds=40,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = model_xgb.predict_proba(X_va)
    test_xgb += model_xgb.predict_proba(X_test) / 5.0

    # CatBoost
    model_cb = CatBoostClassifier(
        iterations=800,
        learning_rate=0.04,
        depth=6,
        early_stopping_rounds=40,
        random_seed=42 + fold,
        verbose=0,
        thread_count=-1,
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[va_idx] = model_cb.predict_proba(X_va)
    test_cb += model_cb.predict_proba(X_test) / 5.0

    # ExtraTrees
    model_et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        random_state=42 + fold,
        n_jobs=-1,
    )
    model_et.fit(X_tr, y_tr)
    oof_et[va_idx] = model_et.predict_proba(X_va)
    test_et += model_et.predict_proba(X_test) / 5.0

    # MLP
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)

    model_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=150,
        early_stopping=True,
        random_state=42 + fold,
    )
    model_mlp.fit(X_tr_sc, y_tr)
    oof_mlp[va_idx] = model_mlp.predict_proba(X_va_sc)
    test_mlp += model_mlp.predict_proba(X_te_sc) / 5.0

# 8. Optimize Blend Weights
models_oof = [oof_lgb, oof_xgb, oof_cb, oof_et, oof_mlp]
models_test = [test_lgb, test_xgb, test_cb, test_et, test_mlp]


def loss_func(weights):
    weights = np.array(weights)
    weights = np.maximum(0, weights)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(len(models_oof)) / len(models_oof)

    blended_oof = sum(w * oof for w, oof in zip(weights, models_oof))
    blended_preds = np.argmax(blended_oof, axis=1)
    return -accuracy_score(y_train, blended_preds)


init_weights = [0.25, 0.25, 0.25, 0.15, 0.10]
bounds = [(0, 1)] * len(models_oof)
res = minimize(loss_func, init_weights, bounds=bounds, method="Nelder-Mead")

best_weights = np.maximum(0, res.x)
best_weights /= best_weights.sum()

# Compute Final OOF Prediction & Evaluation
oof_ensemble = sum(w * oof for w, oof in zip(best_weights, models_oof))
oof_predictions = np.argmax(oof_ensemble, axis=1)
cv_accuracy = accuracy_score(y_train, oof_predictions)

# 9. Generate & Save Submission
test_ensemble = sum(w * test_p for w, test_p in zip(best_weights, models_test))
final_test_classes = np.argmax(test_ensemble, axis=1)
final_test_labels = le_target.inverse_transform(final_test_classes)

submission = pd.DataFrame(
    {
        "id": test_df["id"] if "id" in test_df.columns else np.arange(len(test_df)),
        "dea_schedule": final_test_labels,
    }
)
submission.to_csv("./working/submission.csv", index=False)

print(f"Validation Metric (Accuracy): {cv_accuracy:.5f}")
