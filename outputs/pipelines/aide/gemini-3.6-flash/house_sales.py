import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

y = train_df["price"].values
y_log = np.log1p(y)

train_len = len(train_df)
df = pd.concat(
    [train_df.drop(columns=["price"], errors="ignore"), test_df], ignore_index=True
)

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# 2. Temporal Features
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_day"] = df["date"].dt.day
    df["sale_dayofweek"] = df["date"].dt.dayofweek
    df["sale_quarter"] = df["date"].dt.quarter
    df.drop(columns=["date"], inplace=True)

sale_year = df["sale_year"] if "sale_year" in df.columns else 2015
if "yr_built" in df.columns:
    df["house_age"] = sale_year - df["yr_built"]
    if "yr_renovated" in df.columns:
        df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
        df["renovated_age"] = np.where(
            df["yr_renovated"] > 0, sale_year - df["yr_renovated"], df["house_age"]
        )

# 3. Spatial Features & Distances
if "lat" in df.columns and "long" in df.columns:

    def haversine(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
        )
        return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))

    hubs = {
        "dist_seattle": (47.6062, -122.3321),
        "dist_bellevue": (47.6101, -122.2015),
        "dist_redmond": (47.6740, -122.1215),
        "dist_medina": (47.6167, -122.2333),
        "dist_seatac": (47.4502, -122.3088),
    }
    for name, (h_lat, h_lng) in hubs.items():
        df[name] = haversine(df["lat"].values, df["long"].values, h_lat, h_lng)

    for angle in [15, 30, 45, 60, 75]:
        rad = np.radians(angle)
        df[f"rot_{angle}_x"] = df["lat"] * np.cos(rad) - df["long"] * np.sin(rad)
        df[f"rot_{angle}_y"] = df["lat"] * np.sin(rad) + df["long"] * np.cos(rad)

    coords = df[["lat", "long"]].values
    kmeans = KMeans(n_clusters=25, random_state=42, n_init=10)
    df["geo_cluster"] = kmeans.fit_predict(coords)

# 4. Structural & Domain Features
if "sqft_living" in df.columns and "sqft_lot" in df.columns:
    df["sqft_living_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1.0)
if "sqft_living" in df.columns and "sqft_living15" in df.columns:
    df["sqft_living15_ratio"] = df["sqft_living"] / (df["sqft_living15"] + 1.0)
if "sqft_lot" in df.columns and "sqft_lot15" in df.columns:
    df["sqft_lot15_ratio"] = df["sqft_lot"] / (df["sqft_lot15"] + 1.0)
if "sqft_above" in df.columns and "sqft_living" in df.columns:
    df["sqft_above_ratio"] = df["sqft_above"] / (df["sqft_living"] + 1.0)
if "sqft_basement" in df.columns and "sqft_living" in df.columns:
    df["sqft_basement_ratio"] = df["sqft_basement"] / (df["sqft_living"] + 1.0)
if "bedrooms" in df.columns and "bathrooms" in df.columns:
    df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 0.01)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    if "sqft_living" in df.columns:
        df["sqft_per_bedroom"] = df["sqft_living"] / (df["bedrooms"] + 1.0)
        df["sqft_per_room"] = df["sqft_living"] / (df["total_rooms"] + 1.0)
if "grade" in df.columns and "condition" in df.columns:
    df["grade_condition_prod"] = df["grade"] * df["condition"]

df = df.fillna(df.median(numeric_only=True))

for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].astype("category").cat.codes

X_all = df.iloc[:train_len].copy()
X_test = df.iloc[train_len:].copy()

# 5. Cross-Validation & Optimized Ensemble Modeling
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_price = np.zeros(train_len)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_log)):
    X_tr, y_tr = X_all.iloc[train_idx].copy(), y_log[train_idx]
    X_va, y_va = X_all.iloc[val_idx].copy(), y_log[val_idx]
    X_te = X_test.copy()

    pps_tr = y_tr - np.log(X_tr["sqft_living"].values + 1.0)

    # Out-of-fold target encoding and zipcode-level baseline aggregations
    if "zipcode" in X_tr.columns:
        smoothing = 10
        global_mean = y_tr.mean()
        stats = (
            pd.DataFrame({"zipcode": X_tr["zipcode"], "target": y_tr})
            .groupby("zipcode")["target"]
            .agg(["count", "mean"])
        )
        smooth_map = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        ).to_dict()

        X_tr["zipcode_te"] = X_tr["zipcode"].map(smooth_map).fillna(global_mean)
        X_va["zipcode_te"] = X_va["zipcode"].map(smooth_map).fillna(global_mean)
        X_te["zipcode_te"] = X_te["zipcode"].map(smooth_map).fillna(global_mean)

        zip_stats = (
            pd.DataFrame(
                {
                    "zipcode": X_tr["zipcode"],
                    "sqft": X_tr["sqft_living"],
                    "grade": X_tr["grade"],
                    "pps": pps_tr,
                }
            )
            .groupby("zipcode")
            .agg({"sqft": "mean", "grade": "mean", "pps": "mean"})
        )

        global_sqft = X_tr["sqft_living"].mean()
        global_grade = X_tr["grade"].mean()
        global_pps = pps_tr.mean()

        for dataset in [X_tr, X_va, X_te]:
            dataset["zip_mean_sqft"] = (
                dataset["zipcode"].map(zip_stats["sqft"]).fillna(global_sqft)
            )
            dataset["zip_mean_grade"] = (
                dataset["zipcode"].map(zip_stats["grade"]).fillna(global_grade)
            )
            dataset["zip_mean_pps"] = (
                dataset["zipcode"].map(zip_stats["pps"]).fillna(global_pps)
            )

            dataset["zip_sqft_ratio"] = dataset["sqft_living"] / (
                dataset["zip_mean_sqft"] + 1.0
            )
            dataset["zip_grade_diff"] = dataset["grade"] - dataset["zip_mean_grade"]
            dataset["zip_implied_log_price"] = (
                np.log(dataset["sqft_living"] + 1.0) + dataset["zip_mean_pps"]
            )

    # Out-of-fold Spatial K-Nearest Neighbors Target & Structural Comparative Features
    if "lat" in X_tr.columns and "long" in X_tr.columns:
        coords_tr = np.radians(X_tr[["lat", "long"]].values)
        coords_va = np.radians(X_va[["lat", "long"]].values)
        coords_te = np.radians(X_te[["lat", "long"]].values)

        nn = NearestNeighbors(n_neighbors=51, metric="haversine", n_jobs=-1)
        nn.fit(coords_tr)

        dist_tr, idx_tr = nn.kneighbors(coords_tr, n_neighbors=51)
        dist_tr_ex, idx_tr_ex = dist_tr[:, 1:], idx_tr[:, 1:]

        dist_va, idx_va = nn.kneighbors(coords_va, n_neighbors=50)
        dist_te, idx_te = nn.kneighbors(coords_te, n_neighbors=50)

        sqft_tr = X_tr["sqft_living"].values
        grade_tr = X_tr["grade"].values

        for k in [5, 15, 50]:
            vals_tr = y_tr[idx_tr_ex[:, :k]]
            X_tr[f"knn_{k}_mean"] = vals_tr.mean(axis=1)
            X_tr[f"knn_{k}_std"] = vals_tr.std(axis=1)

            pps_vals_tr = pps_tr[idx_tr_ex[:, :k]]
            X_tr[f"knn_{k}_pps_mean"] = pps_vals_tr.mean(axis=1)

            vals_va = y_tr[idx_va[:, :k]]
            X_va[f"knn_{k}_mean"] = vals_va.mean(axis=1)
            X_va[f"knn_{k}_std"] = vals_va.std(axis=1)

            pps_vals_va = pps_tr[idx_va[:, :k]]
            X_va[f"knn_{k}_pps_mean"] = pps_vals_va.mean(axis=1)

            vals_te = y_tr[idx_te[:, :k]]
            X_te[f"knn_{k}_mean"] = vals_te.mean(axis=1)
            X_te[f"knn_{k}_std"] = vals_te.std(axis=1)

            pps_vals_te = pps_tr[idx_te[:, :k]]
            X_te[f"knn_{k}_pps_mean"] = pps_vals_te.mean(axis=1)

        # Distance-weighted neighbor statistics
        w_tr = 1.0 / (dist_tr_ex[:, :15] * 6371.0 + 0.05)
        X_tr["knn_15_weighted_mean"] = (y_tr[idx_tr_ex[:, :15]] * w_tr).sum(
            axis=1
        ) / w_tr.sum(axis=1)
        X_tr["knn_15_pps_wmean"] = (pps_tr[idx_tr_ex[:, :15]] * w_tr).sum(
            axis=1
        ) / w_tr.sum(axis=1)

        w_va = 1.0 / (dist_va[:, :15] * 6371.0 + 0.05)
        X_va["knn_15_weighted_mean"] = (y_tr[idx_va[:, :15]] * w_va).sum(
            axis=1
        ) / w_va.sum(axis=1)
        X_va["knn_15_pps_wmean"] = (pps_tr[idx_va[:, :15]] * w_va).sum(
            axis=1
        ) / w_va.sum(axis=1)

        w_te = 1.0 / (dist_te[:, :15] * 6371.0 + 0.05)
        X_te["knn_15_weighted_mean"] = (y_tr[idx_te[:, :15]] * w_te).sum(
            axis=1
        ) / w_te.sum(axis=1)
        X_te["knn_15_pps_wmean"] = (pps_tr[idx_te[:, :15]] * w_te).sum(
            axis=1
        ) / w_te.sum(axis=1)

        # Comparative structural neighbor baselines and implied valuations
        sqft_tr_k15 = sqft_tr[idx_tr_ex[:, :15]].mean(axis=1)
        X_tr["knn_15_sqft_ratio"] = X_tr["sqft_living"] / (sqft_tr_k15 + 1.0)
        grade_tr_k15 = grade_tr[idx_tr_ex[:, :15]].mean(axis=1)
        X_tr["knn_15_grade_diff"] = X_tr["grade"] - grade_tr_k15
        X_tr["knn_implied_log_price"] = (
            np.log(X_tr["sqft_living"] + 1.0) + X_tr["knn_15_pps_wmean"]
        )

        sqft_va_k15 = sqft_tr[idx_va[:, :15]].mean(axis=1)
        X_va["knn_15_sqft_ratio"] = X_va["sqft_living"] / (sqft_va_k15 + 1.0)
        grade_va_k15 = grade_tr[idx_va[:, :15]].mean(axis=1)
        X_va["knn_15_grade_diff"] = X_va["grade"] - grade_va_k15
        X_va["knn_implied_log_price"] = (
            np.log(X_va["sqft_living"] + 1.0) + X_va["knn_15_pps_wmean"]
        )

        sqft_te_k15 = sqft_tr[idx_te[:, :15]].mean(axis=1)
        X_te["knn_15_sqft_ratio"] = X_te["sqft_living"] / (sqft_te_k15 + 1.0)
        grade_te_k15 = grade_tr[idx_te[:, :15]].mean(axis=1)
        X_te["knn_15_grade_diff"] = X_te["grade"] - grade_te_k15
        X_te["knn_implied_log_price"] = (
            np.log(X_te["sqft_living"] + 1.0) + X_te["knn_15_pps_wmean"]
        )

    # Model 1: LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Model 2: XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Model 3: CatBoost
    cb_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    val_lgb_price = np.expm1(lgb_model.predict(X_va))
    val_xgb_price = np.expm1(xgb_model.predict(X_va))
    val_cb_price = np.expm1(cb_model.predict(X_va))

    test_lgb_price = np.expm1(lgb_model.predict(X_te))
    test_xgb_price = np.expm1(xgb_model.predict(X_te))
    test_cb_price = np.expm1(cb_model.predict(X_te))

    # Perform constrained SLSQP optimization in price space for current fold
    preds_va = np.vstack([val_lgb_price, val_xgb_price, val_cb_price])
    preds_te = np.vstack([test_lgb_price, test_xgb_price, test_cb_price])
    y_va_true = y[val_idx]

    def blend_objective(weights):
        blend = np.sum(weights[:, None] * preds_va, axis=0)
        return np.sqrt(mean_squared_error(y_va_true, blend))

    init_weights = np.array([0.4, 0.3, 0.3])
    bounds = [(0, 1)] * 3
    constraints = {"type": "eq", "fun": lambda w: 1.0 - np.sum(w)}

    res = minimize(
        blend_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    opt_weights = res.x

    oof_price[val_idx] = np.sum(opt_weights[:, None] * preds_va, axis=0)
    test_preds += np.sum(opt_weights[:, None] * preds_te, axis=0) / kf.n_splits

# 6. Evaluation & Submission Generation
final_rmse = np.sqrt(mean_squared_error(y, oof_price))

os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_df)), "price": test_preds})
sub.to_csv("./working/submission.csv", index=False)

print(f"OOF RMSE: {final_rmse:.4f}")
