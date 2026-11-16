import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

# Directory Setup
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Raw Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Financial Assistance"
if target_col not in train_df.columns:
    for col in train_df.columns:
        if col.lower() == target_col.lower():
            target_col = col
            break

# Target Transformation (Log1p for right-skewed monetary target)
y_train_raw = train_df[target_col].copy().values
y_train_clipped = np.clip(y_train_raw, a_min=0, a_max=None)
y_train_log = np.log1p(y_train_clipped)

train_features = train_df.drop(columns=[target_col]).copy()
test_features = test_df.copy()

# Identify or assign ID column
id_col = None
for possible_id in ["id", "ID", "Grant_ID", "GrantID", "Row_ID"]:
    if possible_id in train_features.columns:
        id_col = possible_id
        break

# Dynamic Column Classification
numeric_cols = []
categorical_cols = []
text_cols = []
datetime_cols = []

combined_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)

for col in train_features.columns:
    if id_col and col == id_col:
        continue

    if pd.api.types.is_datetime64_any_dtype(combined_df[col]):
        datetime_cols.append(col)
        continue

    if pd.api.types.is_numeric_dtype(combined_df[col]):
        numeric_cols.append(col)
        continue

    if pd.api.types.is_object_dtype(combined_df[col]) or pd.api.types.is_string_dtype(
        combined_df[col]
    ):
        sample_non_null = combined_df[col].dropna().astype(str).head(100)
        if len(sample_non_null) > 0:
            avg_len = sample_non_null.str.len().mean()
            avg_words = sample_non_null.str.split().str.len().mean()

            if (
                any(term in col.lower() for term in ["date", "time", "year", "dt"])
                and avg_len < 30
            ):
                try:
                    pd.to_datetime(sample_non_null, errors="raise")
                    datetime_cols.append(col)
                    continue
                except (ValueError, TypeError):
                    pass

            if avg_words > 4 or avg_len > 60:
                text_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            categorical_cols.append(col)

train_processed = pd.DataFrame(index=train_features.index)
test_processed = pd.DataFrame(index=test_features.index)

if id_col and id_col in train_features.columns:
    train_processed["id"] = train_features[id_col].values
    test_processed["id"] = test_features[id_col].values
else:
    train_processed["id"] = np.arange(len(train_features))
    test_processed["id"] = np.arange(len(test_features))

# 2. Datetime Features Extraction
for col in datetime_cols:
    tr_dt = pd.to_datetime(train_features[col], errors="coerce")
    te_dt = pd.to_datetime(test_features[col], errors="coerce")

    for dt_series, target_df in [(tr_dt, train_processed), (te_dt, test_processed)]:
        target_df[f"{col}_year"] = dt_series.dt.year.fillna(-1).astype(float)
        target_df[f"{col}_month"] = dt_series.dt.month.fillna(-1).astype(float)
        target_df[f"{col}_day"] = dt_series.dt.day.fillna(-1).astype(float)
        target_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(-1).astype(float)
        target_df[f"{col}_quarter"] = dt_series.dt.quarter.fillna(-1).astype(float)

if len(datetime_cols) >= 2:
    for i in range(len(datetime_cols)):
        for j in range(i + 1, len(datetime_cols)):
            col1, col2 = datetime_cols[i], datetime_cols[j]
            tr_diff = (
                pd.to_datetime(train_features[col2], errors="coerce")
                - pd.to_datetime(train_features[col1], errors="coerce")
            ).dt.days
            te_diff = (
                pd.to_datetime(test_features[col2], errors="coerce")
                - pd.to_datetime(test_features[col1], errors="coerce")
            ).dt.days
            train_processed[f"{col2}_minus_{col1}_days"] = tr_diff.fillna(-999).astype(
                float
            )
            test_processed[f"{col2}_minus_{col1}_days"] = te_diff.fillna(-999).astype(
                float
            )

# 3. Stratified 5-Fold Splitting & Unstructured Text Features (with ST Embeddings & Dual OOF Ridge Features)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_processed["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
    train_processed.loc[val_idx, "fold"] = fold

oof_ridge_log = np.zeros(len(train_features))
test_ridge_log = np.zeros(len(test_features))
oof_ridge_raw = np.zeros(len(train_features))
test_ridge_raw = np.zeros(len(test_features))

tfidf_matrices_tr = []
tfidf_matrices_te = []

if HAS_ST and len(text_cols) > 0:
    try:
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        for col in text_cols:
            tr_text = train_features[col].fillna("").astype(str).tolist()
            te_text = test_features[col].fillna("").astype(str).tolist()
            tr_emb = st_model.encode(tr_text, show_progress_bar=False, batch_size=64)
            te_emb = st_model.encode(te_text, show_progress_bar=False, batch_size=64)

            n_st_comp = min(32, tr_emb.shape[1])
            svd_st = TruncatedSVD(n_components=n_st_comp, random_state=42)
            tr_st_svd = svd_st.fit_transform(tr_emb)
            te_st_svd = svd_st.transform(te_emb)
            for d in range(n_st_comp):
                train_processed[f"{col}_st_{d}"] = tr_st_svd[:, d]
                test_processed[f"{col}_st_{d}"] = te_st_svd[:, d]
    except Exception:
        pass

for col in text_cols:
    tr_text = train_features[col].fillna("").astype(str)
    te_text = test_features[col].fillna("").astype(str)

    train_processed[f"{col}_char_len"] = tr_text.str.len().astype(float)
    test_processed[f"{col}_char_len"] = te_text.str.len().astype(float)

    train_processed[f"{col}_word_count"] = tr_text.str.split().str.len().astype(float)
    test_processed[f"{col}_word_count"] = te_text.str.split().str.len().astype(float)

    train_processed[f"{col}_digit_count"] = tr_text.apply(
        lambda s: sum(c.isdigit() for c in s)
    ).astype(float)
    test_processed[f"{col}_digit_count"] = te_text.apply(
        lambda s: sum(c.isdigit() for c in s)
    ).astype(float)

    tfidf_word = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
    )
    tr_tfidf_w = tfidf_word.fit_transform(tr_text)
    te_tfidf_w = tfidf_word.transform(te_text)

    tfidf_char = TfidfVectorizer(
        max_features=15000, analyzer="char", ngram_range=(2, 4), sublinear_tf=True
    )
    tr_tfidf_c = tfidf_char.fit_transform(tr_text)
    te_tfidf_c = tfidf_char.transform(te_text)

    tfidf_matrices_tr.extend([tr_tfidf_w, tr_tfidf_c])
    tfidf_matrices_te.extend([te_tfidf_w, te_tfidf_c])

    n_comp = min(30, tr_tfidf_w.shape[1] - 1) if tr_tfidf_w.shape[1] > 1 else 1
    if n_comp > 1:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        tr_svd = svd.fit_transform(tr_tfidf_w)
        te_svd = svd.transform(te_tfidf_w)

        for k in range(n_comp):
            train_processed[f"{col}_svd_{k}"] = tr_svd[:, k]
            test_processed[f"{col}_svd_{k}"] = te_svd[:, k]

if len(tfidf_matrices_tr) > 0:
    X_tfidf_tr = hstack(tfidf_matrices_tr).tocsr()
    X_tfidf_te = hstack(tfidf_matrices_te).tocsr()

    for fold in range(5):
        val_mask = (train_processed["fold"] == fold).values
        tr_mask = ~val_mask

        ridge_log = Ridge(alpha=1.0, random_state=42)
        ridge_log.fit(X_tfidf_tr[tr_mask], y_train_log[tr_mask])
        oof_ridge_log[val_mask] = ridge_log.predict(X_tfidf_tr[val_mask])
        test_ridge_log += ridge_log.predict(X_tfidf_te) / 5.0

        ridge_raw = Ridge(alpha=100.0, random_state=42)
        ridge_raw.fit(X_tfidf_tr[tr_mask], y_train_clipped[tr_mask])
        oof_ridge_raw[val_mask] = ridge_raw.predict(X_tfidf_tr[val_mask])
        test_ridge_raw += ridge_raw.predict(X_tfidf_te) / 5.0

    train_processed["tfidf_ridge_oof"] = oof_ridge_log
    test_processed["tfidf_ridge_oof"] = test_ridge_log
    train_processed["tfidf_ridge_raw_oof"] = oof_ridge_raw
    test_processed["tfidf_ridge_raw_oof"] = test_ridge_raw

# 4. Categorical Feature Encoding

for col in categorical_cols:
    tr_cat = train_features[col].fillna("Missing").astype(str)
    te_cat = test_features[col].fillna("Missing").astype(str)

    freq_map = tr_cat.value_counts(normalize=True).to_dict()
    train_processed[f"{col}_freq"] = tr_cat.map(freq_map).fillna(0).astype(float)
    test_processed[f"{col}_freq"] = te_cat.map(freq_map).fillna(0).astype(float)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_processed[f"{col}_ord"] = (
        enc.fit_transform(tr_cat.values.reshape(-1, 1)).ravel().astype(float)
    )
    test_processed[f"{col}_ord"] = (
        enc.transform(te_cat.values.reshape(-1, 1)).ravel().astype(float)
    )

    train_processed[f"{col}_te_mean"] = np.nan
    test_processed[f"{col}_te_mean"] = np.nan
    global_mean = y_train_log.mean()

    for fold in range(5):
        val_mask = (train_processed["fold"] == fold).values
        tr_mask = ~val_mask

        tr_cat_fold = tr_cat.iloc[tr_mask]
        y_fold = pd.Series(y_train_log).iloc[tr_mask]

        target_map = y_fold.groupby(tr_cat_fold).mean().to_dict()
        train_processed.loc[val_mask, f"{col}_te_mean"] = (
            tr_cat.iloc[val_mask].map(target_map).fillna(global_mean)
        )

    full_target_map = pd.Series(y_train_log).groupby(tr_cat).mean().to_dict()
    test_processed[f"{col}_te_mean"] = te_cat.map(full_target_map).fillna(global_mean)

# 5. Numerical Imputation, Scaling, and Log Transformations
for col in numeric_cols:
    tr_num = train_features[col].astype(float)
    te_num = test_features[col].astype(float)

    if tr_num.isna().sum() > 0 or te_num.isna().sum() > 0:
        train_processed[f"{col}_isnan"] = tr_num.isna().astype(float)
        test_processed[f"{col}_isnan"] = te_num.isna().astype(float)

    med = tr_num.median()
    if np.isnan(med):
        med = 0.0

    tr_imputed = tr_num.fillna(med)
    te_imputed = te_num.fillna(med)

    train_processed[col] = tr_imputed
    test_processed[col] = te_imputed

    if (tr_imputed >= 0).all() and tr_imputed.skew() > 1.5:
        train_processed[f"{col}_log1p"] = np.log1p(tr_imputed)
        test_processed[f"{col}_log1p"] = np.log1p(te_imputed)

# 6. Group Aggregation Features (All Categorical and Numerical pairs)
for grp_cat in categorical_cols:
    for grp_num in numeric_cols:
        grp_stats = (
            train_features.groupby(train_features[grp_cat].fillna("Missing"))[grp_num]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        grp_stats.columns = [
            grp_cat,
            f"{grp_cat}_{grp_num}_mean",
            f"{grp_cat}_{grp_num}_std",
            f"{grp_cat}_{grp_num}_min",
            f"{grp_cat}_{grp_num}_max",
        ]

        tr_grp = (
            train_features[[grp_cat]]
            .fillna("Missing")
            .merge(grp_stats, on=grp_cat, how="left")
        )
        te_grp = (
            test_features[[grp_cat]]
            .fillna("Missing")
            .merge(grp_stats, on=grp_cat, how="left")
        )

        for stat in ["mean", "std", "min", "max"]:
            train_processed[f"{grp_cat}_{grp_num}_grp_{stat}"] = (
                tr_grp[f"{grp_cat}_{grp_num}_{stat}"].fillna(0).values
            )
            test_processed[f"{grp_cat}_{grp_num}_grp_{stat}"] = (
                te_grp[f"{grp_cat}_{grp_num}_{stat}"].fillna(0).values
            )

# Prepare Model Arrays
ignore_cols = ["id", "fold"]
feature_cols = [c for c in train_processed.columns if c not in ignore_cols]

X = train_processed[feature_cols].values
X_test = test_processed[feature_cols].values
folds = train_processed["fold"].values

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

n_samples = len(train_processed)
n_test = len(test_processed)
n_folds = 5

oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_cb = np.zeros(n_samples)

test_lgb = np.zeros(n_test)
test_xgb = np.zeros(n_test)
test_cb = np.zeros(n_test)

# 7. 5-Fold Model Training Loop
for fold in range(n_folds):
    val_mask = folds == fold
    tr_mask = ~val_mask

    X_tr, y_tr_log = X[tr_mask], y_train_log[tr_mask]
    X_va, y_va_log = X[val_mask], y_train_log[val_mask]

    # Model 1: LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr_log,
        eval_set=[(X_va, y_va_log)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_mask] = model_lgb.predict(X_va)
    test_lgb += model_lgb.predict(X_test) / n_folds

    # Model 2: XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.025,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
    oof_xgb[val_mask] = model_xgb.predict(X_va)
    test_xgb += model_xgb.predict(X_test) / n_folds

    # Model 3: CatBoost or HistGradientBoosting
    if HAS_CATBOOST:
        model_cb = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.025,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50,
        )
        model_cb.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=False)
        oof_cb[val_mask] = model_cb.predict(X_va)
        test_cb += model_cb.predict(X_test) / n_folds
    else:
        model_cb = HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.025,
            max_depth=6,
            random_state=42 + fold,
            early_stopping=True,
        )
        model_cb.fit(X_tr, y_tr_log)
        oof_cb[val_mask] = model_cb.predict(X_va)
        test_cb += model_cb.predict(X_test) / n_folds


# 8. Ensemble Optimization & Jensen Scale Correction (Linear + GBDT Blend)
def objective(weights):
    w1, w2, w3, w4 = weights
    pred_log = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb + w4 * oof_ridge_log
    pred_raw = np.clip(np.expm1(pred_log), a_min=0, a_max=None)
    c = np.sum(y_train_raw * pred_raw) / (np.sum(pred_raw**2) + 1e-8)
    return np.sqrt(mean_squared_error(y_train_raw, c * pred_raw))


res = minimize(
    objective,
    [0.25, 0.25, 0.25, 0.25],
    bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
    constraints={"type": "eq", "fun": lambda w: 1.0 - sum(w)},
)

w1, w2, w3, w4 = res.x
oof_blend_log = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb + w4 * oof_ridge_log
test_blend_log = w1 * test_lgb + w2 * test_xgb + w3 * test_cb + w4 * test_ridge_log

oof_blend_raw = np.clip(np.expm1(oof_blend_log), a_min=0, a_max=None)
test_blend_raw = np.clip(np.expm1(test_blend_log), a_min=0, a_max=None)

c_opt = np.sum(y_train_raw * oof_blend_raw) / (np.sum(oof_blend_raw**2) + 1e-8)

oof_final = c_opt * oof_blend_raw
test_final = c_opt * test_blend_raw

final_rmse = np.sqrt(mean_squared_error(y_train_raw, oof_final))

# 9. Output Submission File
submission_df = pd.DataFrame(
    {
        "id": np.arange(len(test_df)),
        "Financial Assistance": test_final,
    }
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {final_rmse}")