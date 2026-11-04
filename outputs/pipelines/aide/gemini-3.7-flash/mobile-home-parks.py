import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "SIZE"
y_raw = train_df[target_col]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

n_train = len(train_df)
df_all = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# 2. Feature Engineering
# Spatial coordinate features
lat_col = next((c for c in df_all.columns if "LAT" in c.upper()), None)
lon_col = next(
    (c for c in df_all.columns if "LON" in c.upper() or "LONG" in c.upper()), None
)

if lat_col and lon_col:
    df_all[lat_col] = pd.to_numeric(df_all[lat_col], errors="coerce")
    df_all[lon_col] = pd.to_numeric(df_all[lon_col], errors="coerce")

    lat_median = df_all[lat_col].median()
    lon_median = df_all[lon_col].median()
    lat_clean = df_all[lat_col].fillna(lat_median).values
    lon_clean = df_all[lon_col].fillna(lon_median).values

    coords = np.column_stack([lat_clean, lon_clean])

    # Multi-scale K-Nearest Neighbors spatial density features
    for k in [3, 7, 15, 30]:
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(coords)
        dists, _ = nn.kneighbors(coords)
        df_all[f"knn_dist_min_{k}"] = dists[:, 1]
        df_all[f"knn_dist_mean_{k}"] = dists[:, 1:].mean(axis=1)
        df_all[f"knn_dist_std_{k}"] = dists[:, 1:].std(axis=1)

    # Spatial projections & interactions
    df_all["coord_sum"] = df_all[lat_col] + df_all[lon_col]
    df_all["coord_diff"] = df_all[lat_col] - df_all[lon_col]
    df_all["coord_dist_center"] = np.sqrt(
        (df_all[lat_col] - lat_median) ** 2 + (df_all[lon_col] - lon_median) ** 2
    )
    lat_rad = np.radians(lat_clean)
    lon_rad = np.radians(lon_clean)
    df_all["coord_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    df_all["coord_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df_all["coord_z"] = np.sin(lat_rad)

# Process categorical columns
cat_cols = []
for col in list(df_all.columns):
    if df_all[col].dtype == "object" or df_all[col].dtype.name == "category":
        cat_cols.append(col)
        freq = df_all[col].value_counts(dropna=False)
        df_all[f"{col}_freq"] = df_all[col].map(freq).fillna(0).astype(np.float32)

        if lat_col and lon_col and df_all[col].nunique() < 1000:
            grp_lat_mean = df_all.groupby(col)[lat_col].transform("mean")
            grp_lon_mean = df_all.groupby(col)[lon_col].transform("mean")
            df_all[f"{col}_lat_diff"] = df_all[lat_col] - grp_lat_mean
            df_all[f"{col}_lon_diff"] = df_all[lon_col] - grp_lon_mean

        df_all[col] = df_all[col].astype("category").cat.codes

df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
X = df_all.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_all.iloc[n_train:].copy().reset_index(drop=True)

# 3. Stratified 5-Fold Setup with Multiclass Out-Of-Fold Target Encoding
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Select categorical columns for target encoding
te_cols = [
    c for c in cat_cols if df_all[c].nunique() > 1 and df_all[c].nunique() < 5000
]

# Add multiclass target encodings per fold
X_te = X.copy()
X_test_te = X_test.copy()
smoothing = 10.0

for c in range(num_classes):
    y_binary = (y == c).astype(float)
    prior = y_binary.mean()

    for col in te_cols:
        col_te_name = f"{col}_te_class_{c}"
        X_te[col_te_name] = np.nan
        X_test_te[col_te_name] = 0.0

        for train_idx, val_idx in skf.split(X, y):
            tr_df = pd.DataFrame(
                {col: X.iloc[train_idx][col], "target": y_binary[train_idx]}
            )
            stats = tr_df.groupby(col)["target"].agg(["count", "mean"])
            smooth_te = (stats["count"] * stats["mean"] + smoothing * prior) / (
                stats["count"] + smoothing
            )
            X_te.loc[val_idx, col_te_name] = (
                X.iloc[val_idx][col].map(smooth_te).fillna(prior)
            )

        # Full train target encoding for test set
        full_df = pd.DataFrame({col: X[col], "target": y_binary})
        stats_full = full_df.groupby(col)["target"].agg(["count", "mean"])
        smooth_te_full = (
            stats_full["count"] * stats_full["mean"] + smoothing * prior
        ) / (stats_full["count"] + smoothing)
        X_test_te[col_te_name] = X_test[col].map(smooth_te_full).fillna(prior)

# 4. Multi-Model Training
oof_lgb = np.zeros((n_train, num_classes))
test_lgb = np.zeros((len(X_test), num_classes))

oof_xgb = np.zeros((n_train, num_classes))
test_xgb = np.zeros((len(X_test), num_classes))

oof_cat = np.zeros((n_train, num_classes))
test_cat = np.zeros((len(X_test), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.04,
    "num_leaves": 45,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 900,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.04,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.75,
    "random_state": 42,
    "tree_method": "hist",
    "n_estimators": 900,
    "n_jobs": -1,
}

cat_params = {
    "loss_function": "MultiClass",
    "iterations": 900,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": 42,
    "verbose": False,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_te, y)):
    X_tr, y_tr = X_te.iloc[train_idx], y[train_idx]
    X_va, y_va = X_te.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va)
    test_lgb += lgb_model.predict_proba(X_test_te) / n_splits

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va)
    test_xgb += xgb_model.predict_proba(X_test_te) / n_splits

    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False
    )
    oof_cat[val_idx] = cat_model.predict_proba(X_va)
    test_cat += cat_model.predict_proba(X_test_te) / n_splits


# 5. Optimize Blending Weights
def loss_func(weights):
    w = np.exp(weights) / np.sum(np.exp(weights))
    blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat
    preds = np.argmax(blend, axis=1)
    return -accuracy_score(y, preds)


res = minimize(loss_func, [0.0, 0.0, 0.0], method="Nelder-Mead")
optimal_w = np.exp(res.x) / np.sum(np.exp(res.x))

oof_blend = optimal_w[0] * oof_lgb + optimal_w[1] * oof_xgb + optimal_w[2] * oof_cat
test_blend = optimal_w[0] * test_lgb + optimal_w[1] * test_xgb + optimal_w[2] * test_cat

oof_preds = np.argmax(oof_blend, axis=1)
cv_accuracy = accuracy_score(y, oof_preds)
print(f"Optimal Ensemble Weights: {optimal_w}")
print(f"Validation Accuracy: {cv_accuracy:.5f}")

# 6. Export Submission
os.makedirs("./working", exist_ok=True)
test_class_preds = np.argmax(test_blend, axis=1)
test_labels = label_encoder.inverse_transform(test_class_preds)

submission = pd.DataFrame({"id": np.arange(len(test_labels)), "SIZE": test_labels})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
