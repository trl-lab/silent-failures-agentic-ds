import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "WATERDEPTH"
y = train_df[target_col].values
X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

features = [c for c in X_train_raw.columns if c in X_test_raw.columns]


def feature_engineering(train_data, test_data):
    df_all = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    n_train = len(train_data)

    lat_col = next((c for c in df_all.columns if "lat" in c.lower()), None)
    lon_col = next(
        (c for c in df_all.columns if "lon" in c.lower() or "long" in c.lower()),
        None,
    )

    if lat_col and lon_col:
        lat = df_all[lat_col].fillna(df_all[lat_col].median()).values
        lon = df_all[lon_col].fillna(df_all[lon_col].median()).values

        for angle in [15, 30, 45, 60, 75]:
            rad = np.radians(angle)
            df_all[f"rot_{angle}_x"] = lat * np.cos(rad) - lon * np.sin(rad)
            df_all[f"rot_{angle}_y"] = lat * np.sin(rad) + lon * np.cos(rad)

        df_all["dist_origin"] = np.sqrt(lat**2 + lon**2)
        df_all["lat_lon_ratio"] = lat / (lon + 1e-6)

    cat_cols = []
    num_cols = []

    for c in list(df_all.columns):
        if df_all[c].dtype == "object" or df_all[c].dtype.name == "category":
            cat_cols.append(c)
        elif np.issubdtype(df_all[c].dtype, np.datetime64):
            df_all[f"{c}_year"] = df_all[c].dt.year
            df_all[f"{c}_month"] = df_all[c].dt.month
            df_all[f"{c}_day"] = df_all[c].dt.day
            df_all.drop(columns=[c], inplace=True)
        else:
            num_cols.append(c)

    for c in cat_cols:
        freq = df_all[c].value_counts(normalize=True).to_dict()
        df_all[f"{c}_freq"] = df_all[c].map(freq).fillna(0)

        le = LabelEncoder()
        df_all[c] = le.fit_transform(df_all[c].astype(str).fillna("missing"))

    return (
        df_all.iloc[:n_train].copy(),
        df_all.iloc[n_train:].copy(),
        cat_cols,
        lat_col,
        lon_col,
    )


X_train, X_test, cat_cols, lat_col, lon_col = feature_engineering(
    X_train_raw[features], X_test_raw[features]
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))

if lat_col and lon_col:
    train_coords = X_train[[lat_col, lon_col]].fillna(0).values
    test_coords = X_test[[lat_col, lon_col]].fillna(0).values
else:
    train_coords, test_coords = None, None

ks = [3, 7, 15, 30]
max_k = max(ks)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[trn_idx].copy(), y[trn_idx]
    X_va, y_va = X_train.iloc[val_idx].copy(), y[val_idx]
    X_te = X_test.copy()

    if train_coords is not None:
        knn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
        knn.fit(train_coords[trn_idx])

        # Leave-one-out for train fold (exclude self)
        tr_distances, tr_indices = knn.kneighbors(
            train_coords[trn_idx], n_neighbors=max_k + 1
        )
        tr_distances = tr_distances[:, 1:]
        tr_indices = tr_indices[:, 1:]

        # Query for val and test
        va_distances, va_indices = knn.kneighbors(
            train_coords[val_idx], n_neighbors=max_k
        )
        te_distances, te_indices = knn.kneighbors(test_coords, n_neighbors=max_k)

        X_tr["dist_to_nn_1"] = tr_distances[:, 0]
        X_va["dist_to_nn_1"] = va_distances[:, 0]
        X_te["dist_to_nn_1"] = te_distances[:, 0]

        for k in ks:
            idx_tr_k = tr_indices[:, :k]
            dist_tr_k = tr_distances[:, :k]
            y_tr_k = y_tr[idx_tr_k]
            w_tr_k = 1.0 / (dist_tr_k + 1e-5)
            X_tr[f"knn_{k}_target_mean"] = np.mean(y_tr_k, axis=1)
            X_tr[f"knn_{k}_target_wmean"] = np.sum(y_tr_k * w_tr_k, axis=1) / np.sum(
                w_tr_k, axis=1
            )
            X_tr[f"knn_{k}_target_std"] = np.std(y_tr_k, axis=1)

            idx_va_k = va_indices[:, :k]
            dist_va_k = va_distances[:, :k]
            y_va_k = y_tr[idx_va_k]
            w_va_k = 1.0 / (dist_va_k + 1e-5)
            X_va[f"knn_{k}_target_mean"] = np.mean(y_va_k, axis=1)
            X_va[f"knn_{k}_target_wmean"] = np.sum(y_va_k * w_va_k, axis=1) / np.sum(
                w_va_k, axis=1
            )
            X_va[f"knn_{k}_target_std"] = np.std(y_va_k, axis=1)

            idx_te_k = te_indices[:, :k]
            dist_te_k = te_distances[:, :k]
            y_te_k = y_tr[idx_te_k]
            w_te_k = 1.0 / (dist_te_k + 1e-5)
            X_te[f"knn_{k}_target_mean"] = np.mean(y_te_k, axis=1)
            X_te[f"knn_{k}_target_wmean"] = np.sum(y_te_k * w_te_k, axis=1) / np.sum(
                w_te_k, axis=1
            )
            X_te[f"knn_{k}_target_std"] = np.std(y_te_k, axis=1)

    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_va[c] = X_va[c].astype("category")
        X_te[c] = X_te[c].astype("category")

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_te) / kf.n_splits

    X_tr_num = X_tr.copy()
    X_va_num = X_va.copy()
    X_te_num = X_te.copy()
    for c in cat_cols:
        X_tr_num[c] = X_tr_num[c].astype(int)
        X_va_num[c] = X_va_num[c].astype(int)
        X_te_num[c] = X_te_num[c].astype(int)

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va_num)
    test_xgb += xgb_model.predict(X_te_num) / kf.n_splits

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=False,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_tr_num, y_tr, eval_set=(X_va_num, y_va))
    oof_cat[val_idx] = cat_model.predict(X_va_num)
    test_cat += cat_model.predict(X_te_num) / kf.n_splits

oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat])
test_matrix = np.column_stack([test_lgb, test_xgb, test_cat])


def loss_func(weights):
    weights = weights / np.sum(weights)
    pred = oof_matrix @ weights
    return np.sqrt(mean_squared_error(y, pred))


res = minimize(
    loss_func,
    x0=[1 / 3, 1 / 3, 1 / 3],
    bounds=[(0, 1)] * 3,
    method="SLSQP",
)
best_weights = res.x / np.sum(res.x)

oof_blend = oof_matrix @ best_weights
test_blend = test_matrix @ best_weights

oof_blend = np.clip(oof_blend, 0, None)
test_blend = np.clip(test_blend, 0, None)

cv_rmse = np.sqrt(mean_squared_error(y, oof_blend))
print(f"Validation RMSE: {cv_rmse:.4f}")

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_blend)), "WATERDEPTH": test_blend})
sub.to_csv("./working/submission.csv", index=False)
