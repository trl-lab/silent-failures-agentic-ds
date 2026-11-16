import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "RATIO_ENROLL_EMP"
y = train_df[target_col].values

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Combine datasets for consistent feature engineering
n_train = len(train_df)
X_all = pd.concat(
    [
        train_df.drop(columns=[target_col], errors="ignore"),
        test_df.drop(columns=[target_col], errors="ignore"),
    ],
    ignore_index=True,
)

# 2. Feature Engineering
num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()

# Spatial coordinates detection and transformation
lat_col = next((c for c in num_cols if c.upper() in ["LATITUDE", "LAT", "Y"]), None)
lon_col = next((c for c in num_cols if c.upper() in ["LONGITUDE", "LON", "X"]), None)

if lat_col and lon_col:
    X_all[lat_col] = X_all[lat_col].fillna(X_all[lat_col].median())
    X_all[lon_col] = X_all[lon_col].fillna(X_all[lon_col].median())

    # Rotated spatial coordinates
    for angle in [15, 30, 45, 60, 75]:
        rad = np.radians(angle)
        X_all[f"rot_{angle}_x"] = X_all[lon_col] * np.cos(rad) - X_all[
            lat_col
        ] * np.sin(rad)
        X_all[f"rot_{angle}_y"] = X_all[lon_col] * np.sin(rad) + X_all[
            lat_col
        ] * np.cos(rad)

    # Polar coordinates relative to center
    clat, clon = X_all[lat_col].mean(), X_all[lon_col].mean()
    X_all["dist_center"] = np.sqrt(
        (X_all[lat_col] - clat) ** 2 + (X_all[lon_col] - clon) ** 2
    )
    X_all["bearing_center"] = np.arctan2(X_all[lat_col] - clat, X_all[lon_col] - clon)

# Pairwise Categorical Interactions for domain context
moderate_cats = [c for c in cat_cols if 2 <= X_all[c].nunique() <= 100]
interaction_cols = []
if len(moderate_cats) >= 2:
    top_cats = moderate_cats[:4]
    for i in range(len(top_cats)):
        for j in range(i + 1, len(top_cats)):
            c1, c2 = top_cats[i], top_cats[j]
            inter_name = f"inter_{c1}_{c2}"
            X_all[inter_name] = X_all[c1].astype(str) + "_" + X_all[c2].astype(str)
            interaction_cols.append(inter_name)

all_cat_cols = cat_cols + interaction_cols

# Group-level aggregate statistics for numerical features
grp_cols = [c for c in all_cat_cols if 1 < X_all[c].nunique() < 200]
num_base_cols = [
    c for c in num_cols if "rot" not in c and c not in ["dist_center", "bearing_center"]
]

for g in grp_cols[:4]:
    for col in num_base_cols[:4]:
        grp_mean = X_all.groupby(g)[col].transform("mean")
        grp_std = X_all.groupby(g)[col].transform("std")
        X_all[f"{col}_mean_by_{g}"] = grp_mean
        X_all[f"{col}_diff_{g}"] = X_all[col] - grp_mean
        X_all[f"{col}_std_by_{g}"] = grp_std.fillna(0)

# Expanded Text Decomposition using TF-IDF + TruncatedSVD
for c in cat_cols:
    unique_cnt = X_all[c].nunique()
    if 30 <= unique_cnt <= 10000:
        text_data = X_all[c].fillna("").astype(str)
        tfidf = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), stop_words="english"
        )
        tfidf_mat = tfidf.fit_transform(text_data)
        n_comp = min(12, tfidf_mat.shape[1] - 1)
        if n_comp > 1:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd_feats = svd.fit_transform(tfidf_mat)
            for i in range(n_comp):
                X_all[f"svd_{c}_{i}"] = svd_feats[:, i]

# Frequency Encoding
for c in all_cat_cols:
    freq = X_all[c].value_counts()
    X_all[f"{c}_freq"] = X_all[c].map(freq)

# Prepare string representation for CatBoost categorical features
cb_cat_cols = []
for c in all_cat_cols:
    X_all[c] = X_all[c].fillna("Missing").astype(str)
    cb_cat_cols.append(c)

# Split back to Train and Test sets
X_train = X_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = X_all.iloc[n_train:].copy().reset_index(drop=True)

# 3. Model Training with Multi-Seed Multi-Depth CatBoost Ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
inner_kf = KFold(n_splits=5, shuffle=True, random_state=123)

oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

te_cols = [c for c in all_cat_cols if 1 < X_train[c].nunique() < 1000]
smooth = 10.0
global_mean = y.mean()

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, y_tr = X_train.iloc[train_idx].copy(), y[train_idx]
    X_va, y_va = X_train.iloc[val_idx].copy(), y[val_idx]
    X_te = X_test.copy()

    # Leak-free Target Encoding for Train, Validation, and Test
    for c in te_cols:
        df_tr_c = pd.DataFrame({"cat": X_tr[c], "target": y_tr})
        stats = df_tr_c.groupby("cat")["target"].agg(["count", "mean"])
        smooth_te = (stats["count"] * stats["mean"] + smooth * global_mean) / (
            stats["count"] + smooth
        )
        X_va[f"{c}_te"] = X_va[c].map(smooth_te).fillna(global_mean)
        X_te[f"{c}_te"] = X_te[c].map(smooth_te).fillna(global_mean)

        oof_tr_te = np.zeros(len(X_tr))
        for in_tr, in_va in inner_kf.split(X_tr):
            in_df = pd.DataFrame({"cat": X_tr.iloc[in_tr][c], "target": y_tr[in_tr]})
            in_stats = in_df.groupby("cat")["target"].agg(["count", "mean"])
            in_te = (in_stats["count"] * in_stats["mean"] + smooth * global_mean) / (
                in_stats["count"] + smooth
            )
            oof_tr_te[in_va] = X_tr.iloc[in_va][c].map(in_te).fillna(global_mean)
        X_tr[f"{c}_te"] = oof_tr_te

    # Multi-seed / Multi-depth Ensemble Configurations
    configs = [
        {"depth": 6, "l2_leaf_reg": 3.0, "lr": 0.04, "seed": 42 + fold},
        {"depth": 8, "l2_leaf_reg": 6.0, "lr": 0.03, "seed": 100 + fold},
    ]

    fold_val_pred = np.zeros(len(val_idx))
    fold_test_pred = np.zeros(len(X_test))

    for cfg in configs:
        model = CatBoostRegressor(
            iterations=1300,
            learning_rate=cfg["lr"],
            depth=cfg["depth"],
            l2_leaf_reg=cfg["l2_leaf_reg"],
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=cfg["seed"],
            verbose=0,
            cat_features=cb_cat_cols,
        )

        model.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
        )

        val_p = model.predict(X_va)
        test_p = model.predict(X_te)

        fold_val_pred += val_p / len(configs)
        fold_test_pred += test_p / len(configs)

    oof_preds[val_idx] = np.clip(fold_val_pred, 0, None)
    test_preds += np.clip(fold_test_pred, 0, None) / kf.n_splits

# 4. Evaluation Metric
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {oof_rmse:.5f}")

# 5. Save Test Predictions
submission = pd.DataFrame(
    {"id": np.arange(len(X_test)), "RATIO_ENROLL_EMP": test_preds}
)

submission_path = "./working/submission.csv"
submission.to_csv(submission_path, index=False)
