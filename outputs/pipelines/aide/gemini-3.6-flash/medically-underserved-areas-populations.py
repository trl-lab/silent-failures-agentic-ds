import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "IMU Score"
y = train_df[target_col].values
X = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Filter common columns
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]


# 2. Preprocessing & Feature Engineering
def preprocess(df_tr, df_te):
    tr = df_tr.copy()
    te = df_te.copy()

    # Row-wise missing value counts
    tr["null_count"] = tr.isnull().sum(axis=1)
    te["null_count"] = te.isnull().sum(axis=1)

    cat_cols = []
    num_cols = ["null_count"]
    for c in tr.columns:
        if c == "null_count":
            continue
        if tr[c].dtype == "object" or tr[c].dtype.name == "category":
            cat_cols.append(c)
        elif pd.api.types.is_datetime64_any_dtype(tr[c]):
            for d in [tr, te]:
                d[f"{c}_year"] = d[c].dt.year
                d[f"{c}_month"] = d[c].dt.month
                d[f"{c}_day"] = d[c].dt.day
                d.drop(columns=[c], inplace=True)
            num_cols.extend([f"{c}_year", f"{c}_month", f"{c}_day"])
        else:
            num_cols.append(c)

    for c in cat_cols:
        freq = tr[c].value_counts(normalize=True).to_dict()
        tr[f"{c}_freq"] = tr[c].map(freq).fillna(0)
        te[f"{c}_freq"] = te[c].map(freq).fillna(0)

        tr[c] = tr[c].astype("category").cat.codes
        te[c] = te[c].astype("category").cat.codes
        num_cols.append(f"{c}_freq")

    group_candidates = [
        c for c in cat_cols if tr[c].nunique() > 1 and tr[c].nunique() < 200
    ]
    num_candidates = [c for c in num_cols if "freq" not in c and c != "null_count"][:5]

    for g in group_candidates[:3]:
        for n in num_candidates:
            mean_map = tr.groupby(g)[n].mean().to_dict()
            std_map = tr.groupby(g)[n].std().to_dict()

            tr[f"{g}_{n}_mean"] = tr[g].map(mean_map).fillna(0)
            te[f"{g}_{n}_mean"] = te[g].map(mean_map).fillna(0)
            tr[f"{g}_{n}_std"] = tr[g].map(std_map).fillna(0)
            te[f"{g}_{n}_std"] = te[g].map(std_map).fillna(0)

            # Group relative z-score feature
            tr[f"{g}_{n}_zscore"] = (tr[n].fillna(0) - tr[f"{g}_{n}_mean"]) / (
                tr[f"{g}_{n}_std"] + 1e-5
            )
            te[f"{g}_{n}_zscore"] = (te[n].fillna(0) - te[f"{g}_{n}_mean"]) / (
                te[f"{g}_{n}_std"] + 1e-5
            )

    return tr, te


X_prep, X_test_prep = preprocess(X, X_test)

clean_cols = [
    str(c).replace("[", "_").replace("]", "_").replace("<", "_").replace(":", "_")
    for c in X_prep.columns
]
X_prep.columns = clean_cols
X_test_prep.columns = clean_cols

# 3. 5-Fold Cross Validation Pipeline
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_prep, y)):
    X_tr, y_tr = X_prep.iloc[train_idx], y[train_idx]
    X_va, y_va = X_prep.iloc[val_idx], y[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_lgb += lgb_model.predict(X_test_prep) / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_xgb += xgb_model.predict(X_test_prep) / kf.n_splits

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    oof_cat[val_idx] = cat_model.predict(X_va)
    test_cat += cat_model.predict(X_test_prep) / kf.n_splits

# 4. Ensemble Weight Optimization with Higher Grid Resolution
best_w = (1 / 3, 1 / 3, 1 / 3)
best_score = float("inf")

steps = 51
for w1 in np.linspace(0, 1, steps):
    for w2 in np.linspace(0, 1 - w1, steps):
        w3 = 1.0 - w1 - w2
        if w3 < -1e-5:
            continue
        oof_blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        score = root_mean_squared_error(y, oof_blend)
        if score < best_score:
            best_score = score
            best_w = (w1, w2, w3)

w1, w2, w3 = best_w
oof_final = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
test_final = w1 * test_lgb + w2 * test_xgb + w3 * test_cat

oof_final = np.clip(oof_final, 0, 100)
test_final = np.clip(test_final, 0, 100)

cv_rmse = root_mean_squared_error(y, oof_final)
print(f"5-Fold Cross Validation Blended RMSE: {cv_rmse:.5f}")

# 5. Export Submission
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_final)), "IMU Score": test_final})
sub.to_csv("./working/submission.csv", index=False)
