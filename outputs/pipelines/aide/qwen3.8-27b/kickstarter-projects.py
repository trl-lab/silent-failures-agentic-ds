import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

os.makedirs("./working", exist_ok=True)

TARGET = "status"
SEED = 42
np.random.seed(SEED)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

y_raw = train[TARGET].copy()
train = train.drop(columns=[TARGET])

valid_y = y_raw.notna()
y_raw = y_raw[valid_y]
train = train.loc[valid_y]

# Encode target as binary.
if y_raw.dtype == object or y_raw.dtype == "category":
    vals = set(y_raw.dropna().astype(str).unique())
    if {"successful", "failed"}.issubset(vals):
        y = (y_raw.astype(str) == "successful").astype(int)
    else:
        _, y = pd.factorize(y_raw.astype(str), sort=True)
else:
    y = y_raw.astype(int)

# Safety for unexpected multi-class labels: keep the two most frequent classes.
if y.nunique() > 2:
    top2 = y.value_counts().head(2).index.tolist()
    keep = y.isin(top2)
    y = y[keep]
    train = train.loc[keep]
    mapping = {c: i for i, c in enumerate(sorted(top2))}
    y = y.map(mapping).astype(int)

y = y.reset_index(drop=True)
train = train.reset_index(drop=True)

# Choose common features and remove obvious identifiers / leaked target-like columns.
common = [c for c in train.columns if c in test.columns]
drop_cols = set()

for c in common:
    if c.lower() in {"id", "index", "status", "target", "label", "outcome", "result"}:
        drop_cols.add(c)

for c in common:
    if c in drop_cols:
        continue
    s = train[c]
    if s.notna().sum() == 0:
        drop_cols.add(c)
        continue

    # Drop columns that almost exactly match the raw target.
    try:
        match = (s.astype(str) == y_raw.astype(str)).sum() / max(1, y_raw.notna().sum())
        if match > 0.95 and s.nunique() <= 2:
            drop_cols.add(c)
    except Exception:
        pass

    # Drop numeric columns that are essentially the target.
    try:
        num_s = pd.to_numeric(s, errors="coerce")
        if num_s.notna().sum() > 0.9 * len(y):
            corr = num_s.corr(y)
            if pd.notna(corr) and abs(corr) > 0.999:
                drop_cols.add(c)
    except Exception:
        pass

common = [c for c in common if c not in drop_cols]
Xtr = train[common].copy()
Xte = test[common].copy()

# Improved preprocessing: use native LightGBM categoricals for eligible columns.
CAT_THRESHOLD = 1000
cat_cols = []

for col in list(Xtr.columns):
    if col not in Xte.columns:
        continue

    s_tr = Xtr[col]
    s_te = Xte[col]

    if s_tr.notna().sum() == 0:
        Xtr = Xtr.drop(columns=[col])
        Xte = Xte.drop(columns=[col])
        continue

    # Boolean to numeric.
    if pd.api.types.is_bool_dtype(s_tr):
        Xtr[col] = s_tr.astype(int)
        Xte[col] = s_te.astype(int)
        continue

    # Keep native numeric columns.
    if pd.api.types.is_numeric_dtype(s_tr):
        Xtr[col] = pd.to_numeric(Xtr[col], errors="coerce")
        Xte[col] = pd.to_numeric(Xte[col], errors="coerce")
        Xtr[col] = Xtr[col].replace([np.inf, -np.inf], np.nan)
        Xte[col] = Xte[col].replace([np.inf, -np.inf], np.nan)
        continue

    # Native datetime columns.
    if pd.api.types.is_datetime64_any_dtype(s_tr):
        dt_tr = s_tr
        dt_te = s_te
    else:
        # Try numeric strings first.
        num_tr = pd.to_numeric(s_tr, errors="coerce")
        if num_tr.notna().mean() > 0.9:
            Xtr[col] = num_tr
            Xte[col] = pd.to_numeric(s_te, errors="coerce")
            Xtr[col] = Xtr[col].replace([np.inf, -np.inf], np.nan)
            Xte[col] = Xte[col].replace([np.inf, -np.inf], np.nan)
            continue

        # Try datetime strings.
        sample = s_tr.dropna().head(50)
        dt_ok = False
        if len(sample) >= 5:
            dt_sample = pd.to_datetime(sample, errors="coerce")
            if dt_sample.notna().mean() > 0.8:
                dt_ok = True

        if dt_ok:
            dt_tr = pd.to_datetime(s_tr, errors="coerce")
            dt_te = pd.to_datetime(s_te, errors="coerce")
        else:
            dt_tr = None
            dt_te = None

    if dt_tr is not None and dt_te is not None:
        Xtr[f"{col}_ts"] = dt_tr.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
        Xte[f"{col}_ts"] = dt_te.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
        for attr in ["year", "month", "day", "hour", "dayofweek"]:
            Xtr[f"{col}_{attr}"] = getattr(dt_tr.dt, attr).astype(float)
            Xte[f"{col}_{attr}"] = getattr(dt_te.dt, attr).astype(float)
        Xtr = Xtr.drop(columns=[col])
        Xte = Xte.drop(columns=[col])
        continue

    # Categorical / object columns: use native categorical dtype when cardinality is safe.
    nuniq = s_tr.dropna().nunique()
    if nuniq <= 1 or nuniq > CAT_THRESHOLD:
        Xtr = Xtr.drop(columns=[col])
        Xte = Xte.drop(columns=[col])
        continue

    tr_str = s_tr.astype("string")
    te_str = s_te.astype("string")
    cats = sorted({str(v) for v in tr_str.dropna().unique().tolist()})
    cat_dtype = pd.CategoricalDtype(categories=cats, ordered=False)

    Xtr[col] = tr_str.astype(cat_dtype)
    Xte[col] = te_str.astype(cat_dtype)
    cat_cols.append(col)

# Ensure train/test have exactly the same final feature set.
common_final = [c for c in Xtr.columns if c in Xte.columns]
Xtr = Xtr[common_final]
Xte = Xte[common_final]

# Ensure all remaining non-categorical columns are numeric.
for col in Xtr.columns:
    if pd.api.types.is_categorical_dtype(Xtr[col]):
        continue
    Xtr[col] = pd.to_numeric(Xtr[col], errors="coerce")
    Xte[col] = pd.to_numeric(Xte[col], errors="coerce")
    Xtr[col] = Xtr[col].replace([np.inf, -np.inf], np.nan)
    Xte[col] = Xte[col].replace([np.inf, -np.inf], np.nan)

Xtr = Xtr.loc[:, Xtr.notna().any(axis=0)]
Xte = Xte.loc[:, Xte.notna().any(axis=0)]
Xtr = Xtr.loc[:, Xtr.nunique(dropna=True) > 1]
Xte = Xte.loc[:, [c for c in Xte.columns if c in Xtr.columns]]
Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
Xte = Xte.loc[:, ~Xte.columns.duplicated()]

# Re-align after filtering and refresh categorical feature list.
common_final = [c for c in Xtr.columns if c in Xte.columns]
Xtr = Xtr[common_final]
Xte = Xte[common_final]
cat_cols = [c for c in Xtr.columns if pd.api.types.is_categorical_dtype(Xtr[c])]

if Xtr.shape[1] == 0:
    Xtr["constant_feature"] = 0.0
    Xte["constant_feature"] = 0.0


# Add a few safe, interpretable interactions if the relevant columns exist.
def safe_ratio(num, den):
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


if (
    "goal" in Xtr.columns
    and "usd_pledged" in Xtr.columns
    and pd.api.types.is_numeric_dtype(Xtr["goal"])
    and pd.api.types.is_numeric_dtype(Xtr["usd_pledged"])
):
    Xtr["pledged_goal_ratio"] = safe_ratio(Xtr["usd_pledged"], Xtr["goal"])
    Xte["pledged_goal_ratio"] = safe_ratio(Xte["usd_pledged"], Xte["goal"])

if (
    "created_at_ts" in Xtr.columns
    and "deadline_ts" in Xtr.columns
    and pd.api.types.is_numeric_dtype(Xtr["created_at_ts"])
    and pd.api.types.is_numeric_dtype(Xtr["deadline_ts"])
):
    Xtr["campaign_duration_days"] = (
        Xtr["deadline_ts"] - Xtr["created_at_ts"]
    ) / 86400.0
    Xte["campaign_duration_days"] = (
        Xte["deadline_ts"] - Xte["created_at_ts"]
    ) / 86400.0

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbose": -1,
}

num_boost_round = 1000
early_stop = 50

# 5-fold stratified CV for an unbiased ROC-AUC estimate.
oof = np.zeros(len(Xtr))
models = []
best_iters = []

if y.nunique() >= 2 and len(y) >= 10:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y), start=1):
        dtrain = lgb.Dataset(
            Xtr.iloc[tr_idx],
            label=y.iloc[tr_idx],
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        dvalid = lgb.Dataset(
            Xtr.iloc[va_idx],
            label=y.iloc[va_idx],
            reference=dtrain,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        models.append(model)

        bi = getattr(model, "best_iteration", 0) or 0
        if bi > 0:
            best_iters.append(bi)

        oof[va_idx] = model.predict(Xtr.iloc[va_idx], num_iteration=max(1, bi))
        print(f"Fold {fold}: best_iteration={max(1, bi)}")

    cv_auc = roc_auc_score(y, oof)
else:
    # Fallback if the data is too small for 5-fold.
    from sklearn.model_selection import train_test_split

    tr_idx, va_idx = train_test_split(
        np.arange(len(Xtr)), test_size=0.2, random_state=SEED, stratify=y
    )
    dtrain = lgb.Dataset(
        Xtr.iloc[tr_idx],
        label=y.iloc[tr_idx],
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        Xtr.iloc[va_idx],
        label=y.iloc[va_idx],
        reference=dtrain,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    bi = getattr(model, "best_iteration", 0) or 0
    best_iters.append(max(1, bi))
    oof[va_idx] = model.predict(Xtr.iloc[va_idx], num_iteration=max(1, bi))
    cv_auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])

print(f"CV ROC-AUC: {cv_auc:.6f}")

# Refit on the full training set for the final submission.
final_rounds = int(np.mean(best_iters)) if best_iters else 300
final_rounds = max(50, min(final_rounds, num_boost_round))

final_model = lgb.train(
    params,
    lgb.Dataset(
        Xtr,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False,
    ),
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(Xte, num_iteration=final_rounds)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "status": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
