import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

SEED = 42
rng = np.random.default_rng(SEED)
os.makedirs("./submission", exist_ok=True)

# ---------------------------------------------------------------- load
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if "id" not in test.columns:
    test["id"] = np.arange(len(test), dtype=int)

train = train[train["price"] > 0].reset_index(drop=True)

# ---------------------------------------------------------------- split first
# Stratified 90/10 split by log-price quantiles.
log_price = np.log1p(train["price"])
bins = pd.qcut(log_price, q=20, labels=False, duplicates="drop")
val_mask = np.zeros(len(train), dtype=bool)
for b in np.unique(bins.values):
    rows = np.where(bins.values == b)[0]
    take = max(1, int(round(len(rows) * 0.10)))
    val_mask[rng.choice(rows, size=take, replace=False)] = True

trn_raw = train[~val_mask].copy()
val_raw = train[val_mask].copy()
te_raw = test.copy()

# ---------------------------------------------------------------- fit cleaning stats on training split only
tmp = trn_raw.copy()
for c in ("x", "y", "z"):
    tmp[c] = np.where(tmp[c] == 0, np.nan, tmp[c])
tmp["depth"] = tmp["depth"].replace(0, np.nan)
tmp["table"] = tmp["table"].replace(0, np.nan)
tmp["carat"] = tmp["carat"].replace(0, np.nan)

med = tmp[["x", "y", "z"]].median()
lo = tmp[["x", "y", "z"]].quantile(0.005)
hi = tmp[["x", "y", "z"]].quantile(0.995)
depth_med = float(tmp["depth"].median())
table_med = float(tmp["table"].median())
carat_med = float(tmp["carat"].median())


def preliminary(d, med):
    d = d.copy()
    z_was_zero = (d["z"] == 0).to_numpy()

    for c in ("x", "y", "z"):
        d[c] = np.where(d[c] == 0, np.nan, d[c])

    d[["x", "y", "z"]] = d[["x", "y", "z"]].fillna(med)

    if z_was_zero.any():
        d.loc[z_was_zero, "z"] = (
            d.loc[z_was_zero, "depth"]
            * (d.loc[z_was_zero, "x"] + d.loc[z_was_zero, "y"])
            / 2.0
        )

    d["z"] = d["z"].fillna(med["z"])
    return d


def clean(d):
    d = preliminary(d, med)

    for c in ("x", "y", "z"):
        d[c] = d[c].clip(lo[c], hi[c])

    d["depth"] = d["depth"].replace(0, np.nan).fillna(depth_med).clip(30.0, 90.0)
    d["table"] = d["table"].replace(0, np.nan).fillna(table_med).clip(30.0, 100.0)
    d["carat"] = d["carat"].fillna(carat_med).clip(0.1, 10.0)

    return d


trn = clean(trn_raw)
val = clean(val_raw)
te = clean(te_raw)

# ---------------------------------------------------------------- feature engineering
COLOR_ORD = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
CLARITY_ORD = {
    "I1": 0,
    "SI2": 1,
    "SI1": 2,
    "VS2": 3,
    "VS1": 4,
    "VVS2": 5,
    "VVS1": 6,
    "IF": 7,
}
CUT_ORD = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
CUT_LEVELS = ["Ideal", "Premium", "Very Good", "Good", "Fair"]


def engineer(d):
    out = d.copy()

    out["carat_log"] = np.log1p(out["carat"])
    out["carat_sqrt"] = np.sqrt(out["carat"])
    out["carat2"] = out["carat"] ** 2
    out["carat3"] = out["carat"] ** 3

    for c in ("x", "y", "z"):
        out[f"{c}_log"] = np.log1p(out[c])

    out["mean_xy"] = (out["x"] + out["y"]) / 2.0
    out["volume"] = out["x"] * out["y"] * out["z"]
    out["volume_log"] = np.log1p(out["volume"])
    out["surface"] = 2.0 * (
        out["x"] * out["y"] + out["x"] * out["z"] + out["y"] * out["z"]
    )

    out["xy_ratio"] = out["x"] / np.maximum(out["y"], 1e-6)
    out["xz_ratio"] = out["x"] / np.maximum(out["z"], 1e-6)
    out["yz_ratio"] = out["y"] / np.maximum(out["z"], 1e-6)

    out["depth_x_carat"] = out["depth"] * out["carat"]
    out["table_x_carat"] = out["table"] * out["carat"]

    out["color_ord"] = out["color"].astype(str).map(COLOR_ORD).fillna(-1).astype(np.int8)
    out["clarity_ord"] = out["clarity"].astype(str).map(CLARITY_ORD).fillna(-1).astype(np.int8)
    out["cut_ord"] = out["cut"].astype(str).map(CUT_ORD).fillna(-1).astype(np.int8)

    for name in CUT_LEVELS:
        out[f"cut_{name.replace(' ', '')}"] = (out["cut"] == name).astype(np.int8)

    if "price" in out.columns:
        out["log_price"] = np.log1p(out["price"])

    return out.reset_index(drop=True)


trn = engineer(trn)
val = engineer(val)
te = engineer(te)

FEATURES = [
    "carat",
    "x",
    "y",
    "z",
    "depth",
    "table",
    "carat_log",
    "carat_sqrt",
    "carat2",
    "carat3",
    "x_log",
    "y_log",
    "z_log",
    "mean_xy",
    "volume",
    "volume_log",
    "surface",
    "xy_ratio",
    "xz_ratio",
    "yz_ratio",
    "depth_x_carat",
    "table_x_carat",
    "color_ord",
    "clarity_ord",
    "cut_ord",
    "cut_Ideal",
    "cut_Premium",
    "cut_VeryGood",
    "cut_Good",
    "cut_Fair",
]

assert all(c in trn.columns for c in FEATURES)
assert not trn[FEATURES].isna().any().any()
assert not val[FEATURES].isna().any().any()
assert not te[FEATURES].isna().any().any()

print(
    "[setup] split=stratified-logprice-q20 10% | "
    "metric=RMSE(raw price) | postprocess=max(pred,0)"
)

# ---------------------------------------------------------------- validation model
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 255,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbose": -1,
}

dtrain = lgb.Dataset(trn[FEATURES], label=trn["price"].values)
dvalid = lgb.Dataset(val[FEATURES], label=val["price"].values, reference=dtrain)

model = lgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    valid_sets=[dvalid],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
)

dtrain_log = lgb.Dataset(trn[FEATURES], label=trn["log_price"].values)
dvalid_log = lgb.Dataset(val[FEATURES], label=val["log_price"].values, reference=dtrain_log)

model_log = lgb.train(
    params,
    dtrain_log,
    num_boost_round=5000,
    valid_sets=[dvalid_log],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
)

val_pred = model.predict(val[FEATURES])
val_pred = np.maximum(val_pred, 0.0)

val_pred_log = np.expm1(model_log.predict(val[FEATURES]))
val_pred_log = np.maximum(val_pred_log, 0.0)

best_w = 1.0
score = np.inf
for w in [round(i * 0.05, 2) for i in range(21)]:
    blended = w * val_pred + (1.0 - w) * val_pred_log
    blended = np.maximum(blended, 0.0)
    cand_score = root_mean_squared_error(val["price"].values, blended)
    if cand_score < score:
        score = cand_score
        best_w = float(w)

# ---------------------------------------------------------------- final model on all training rows
best_iter = getattr(model, "best_iteration", None)
if best_iter is None or best_iter <= 0:
    best_iter = 5000
else:
    best_iter = int(best_iter)

best_iter_log = getattr(model_log, "best_iteration", None)
if best_iter_log is None or best_iter_log <= 0:
    best_iter_log = 5000
else:
    best_iter_log = int(best_iter_log)

tr_all = pd.concat([trn, val], axis=0, ignore_index=True)
dtrain_full = lgb.Dataset(tr_all[FEATURES], label=tr_all["price"].values)
dtrain_full_log = lgb.Dataset(tr_all[FEATURES], label=tr_all["log_price"].values)

final_model = lgb.train(params, dtrain_full, num_boost_round=best_iter)
final_model_log = lgb.train(params, dtrain_full_log, num_boost_round=best_iter_log)

test_pred = final_model.predict(te[FEATURES])
test_pred = np.maximum(test_pred, 0.0)

test_pred_log = np.expm1(final_model_log.predict(te[FEATURES]))
test_pred_log = np.maximum(test_pred_log, 0.0)

test_pred = best_w * test_pred + (1.0 - best_w) * test_pred_log
test_pred = np.maximum(test_pred, 0.0)

pd.DataFrame(
    {
        "id": te["id"].astype(int),
        "price": test_pred,
    }
).to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")
