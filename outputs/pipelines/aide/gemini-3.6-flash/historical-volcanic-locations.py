import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    from catboost import CatBoostClassifier

    has_catboost = True
except ImportError:
    has_catboost = False
    from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

target_col = "DEATHS_TOTAL_DESCRIPTION"
train = train.dropna(subset=[target_col]).reset_index(drop=True)

le = LabelEncoder()
y = le.fit_transform(train[target_col].astype(str))
n_classes = len(le.classes_)

# Separate target and prepare combined dataframe
df_train = train.drop(columns=[target_col]).copy()
df_test = test.copy()
n_train = len(df_train)
df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# Feature Engineering: Row-level missingness count
df_all["missing_count"] = df_all.isnull().sum(axis=1)

# 2. Geo-spatial Feature Engineering
lat_cols = [c for c in df_all.columns if "lat" in c.lower()]
lon_cols = [c for c in df_all.columns if "lon" in c.lower()]

if lat_cols and lon_cols:
    lat_c, lon_c = lat_cols[0], lon_cols[0]
    df_all[lat_c] = pd.to_numeric(df_all[lat_c], errors="coerce").fillna(0.0)
    df_all[lon_c] = pd.to_numeric(df_all[lon_c], errors="coerce").fillna(0.0)

    lat_rad = np.radians(df_all[lat_c])
    lon_rad = np.radians(df_all[lon_c])

    df_all["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    df_all["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df_all["coord_z"] = np.sin(lat_rad)
    df_all["abs_lat"] = np.abs(df_all[lat_c])

    coords = df_all[["coord_x", "coord_y", "coord_z"]].values
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df_all["geo_cluster"] = kmeans.fit_predict(coords)

# 3. Numeric Interactions & Domain Features
vei_cols = [c for c in df_all.columns if "vei" in c.lower()]
elev_cols = [
    c for c in df_all.columns if "elev" in c.lower() or "altitude" in c.lower()
]
year_cols = [c for c in df_all.columns if "year" in c.lower()]

if vei_cols and elev_cols:
    vc, ec = vei_cols[0], elev_cols[0]
    df_all[vc] = pd.to_numeric(df_all[vc], errors="coerce").fillna(0)
    df_all[ec] = pd.to_numeric(df_all[ec], errors="coerce").fillna(0)
    df_all["vei_x_elev"] = df_all[vc] * df_all[ec]
    df_all["vei_sq"] = df_all[vc] ** 2

if year_cols:
    yc = year_cols[0]
    df_all[yc] = pd.to_numeric(df_all[yc], errors="coerce").fillna(2000)
    df_all["years_ago"] = 2025 - df_all[yc]
    df_all["is_ancient"] = (df_all[yc] < 1800).astype(int)

# 4. Group Aggregations
group_keys = [
    c
    for c in [
        "COUNTRY",
        "Country",
        "TYPE",
        "Volcano Type",
        "geo_cluster",
        "LOCATION",
        "Location",
    ]
    if c in df_all.columns
]
num_targets = [
    c
    for c in ["ELEVATION", "Elevation", "VEI", "vei", "YEAR", "Year"]
    if c in df_all.columns
]

for gk in group_keys[:3]:
    for nt in num_targets[:2]:
        grp = df_all.groupby(gk)[nt].agg(["mean", "std"]).reset_index()
        grp.columns = [gk, f"{gk}_{nt}_mean", f"{gk}_{nt}_std"]
        df_all = df_all.merge(grp, on=gk, how="left")

# 5. Text & Frequency Encoding
cat_cols = df_all.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()
for col in list(cat_cols):
    df_all[col] = df_all[col].fillna("missing").astype(str)
    freq = df_all[col].value_counts(normalize=True).to_dict()
    df_all[f"{col}_freq"] = df_all[col].map(freq)

    if df_all[col].nunique() > 5:
        tfidf = TfidfVectorizer(max_features=50, stop_words="english")
        tfidf_mat = tfidf.fit_transform(df_all[col])
        if tfidf_mat.shape[1] > 1:
            n_comp = min(3, tfidf_mat.shape[1] - 1)
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd_feats = svd.fit_transform(tfidf_mat)
            for i in range(svd_feats.shape[1]):
                df_all[f"{col}_tfidf_svd_{i}"] = svd_feats[:, i]

for col in cat_cols:
    df_all[col] = LabelEncoder().fit_transform(df_all[col])

# Split back to train and test
X_train_all = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test_all = df_all.iloc[n_train:].copy().reset_index(drop=True)

# 6. Smoothed M-Estimate Target Encoding with Index Alignment
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_series = pd.Series(y, index=X_train_all.index)
m_smooth = 10.0

for col in cat_cols:
    if col in X_train_all.columns and X_train_all[col].nunique() > 3:
        for c in range(n_classes):
            col_te = f"{col}_te_c{c}"
            X_train_all[col_te] = 0.0
            X_test_all[col_te] = 0.0
            global_mean = (y_series == c).mean()

            for tr_idx, val_idx in skf.split(X_train_all, y):
                tr_x = X_train_all.iloc[tr_idx]
                tr_y_c = (y_series.iloc[tr_idx] == c).astype(float)

                grp = tr_y_c.groupby(tr_x[col])
                counts = grp.count()
                means = grp.mean()
                smoothed = (counts * means + m_smooth * global_mean) / (
                    counts + m_smooth
                )

                X_train_all.loc[X_train_all.index[val_idx], col_te] = (
                    X_train_all.iloc[val_idx][col].map(smoothed).fillna(global_mean)
                )

            full_grp = (y_series == c).astype(float).groupby(X_train_all[col])
            full_counts = full_grp.count()
            full_means = full_grp.mean()
            full_smoothed = (full_counts * full_means + m_smooth * global_mean) / (
                full_counts + m_smooth
            )
            X_test_all[col_te] = X_test_all[col].map(full_smoothed).fillna(global_mean)

# Fill any remaining NaNs
X_train_all = X_train_all.fillna(0)
X_test_all = X_test_all.fillna(0)

# 7. Model Training & Out-of-Fold Evaluation
oof_preds_lgb = np.zeros((n_train, n_classes))
oof_preds_xgb = np.zeros((n_train, n_classes))
oof_preds_m3 = np.zeros((n_train, n_classes))

test_preds_lgb = np.zeros((len(X_test_all), n_classes))
test_preds_xgb = np.zeros((len(X_test_all), n_classes))
test_preds_m3 = np.zeros((len(X_test_all), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y)):
    X_tr, y_tr = X_train_all.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_all.iloc[val_idx], y[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1,
    )
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_va)
    test_preds_lgb += model_lgb.predict_proba(X_test_all) / 5.0

    # XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_va)
    test_preds_xgb += model_xgb.predict_proba(X_test_all) / 5.0

    # CatBoost or HistGradientBoosting
    if has_catboost:
        model_m3 = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=0,
        )
        model_m3.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof_preds_m3[val_idx] = model_m3.predict_proba(X_va)
        test_preds_m3 += model_m3.predict_proba(X_test_all) / 5.0
    else:
        model_m3 = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_depth=6,
            random_state=42 + fold,
            early_stopping=True,
            n_iter_no_change=20,
        )
        model_m3.fit(X_tr, y_tr)
        oof_preds_m3[val_idx] = model_m3.predict_proba(X_va)
        test_preds_m3 += model_m3.predict_proba(X_test_all) / 5.0


# 8. Optimize Ensemble Soft-Voting Weights
def loss_func(weights):
    w = np.maximum(0, weights)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    blend = w[0] * oof_preds_lgb + w[1] * oof_preds_xgb + w[2] * oof_preds_m3
    preds = np.argmax(blend, axis=1)
    return -accuracy_score(y, preds)


res = minimize(
    loss_func, [0.40, 0.35, 0.25], method="Nelder-Mead", options={"maxiter": 300}
)
opt_w = np.maximum(0, res.x)
opt_w = opt_w / opt_w.sum()

oof_blend = (
    opt_w[0] * oof_preds_lgb + opt_w[1] * oof_preds_xgb + opt_w[2] * oof_preds_m3
)
oof_pred_labels = np.argmax(oof_blend, axis=1)

val_accuracy = accuracy_score(y, oof_pred_labels)
print(f"Validation Accuracy: {val_accuracy:.5f}")

# 9. Save Submission
test_blend = (
    opt_w[0] * test_preds_lgb + opt_w[1] * test_preds_xgb + opt_w[2] * test_preds_m3
)
test_pred_labels = np.argmax(test_blend, axis=1)
test_pred_categories = le.inverse_transform(test_pred_labels)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test)), "DEATHS_TOTAL_DESCRIPTION": test_pred_categories}
)
submission.to_csv("./working/submission.csv", index=False)
