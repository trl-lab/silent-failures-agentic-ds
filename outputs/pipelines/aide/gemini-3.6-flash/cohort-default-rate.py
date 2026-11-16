import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    import lightgbm as lgb

train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Average or Greater than 30"

# Parse target variable cleanly
y_raw = train_df[target_col]
if y_raw.dtype == object or y_raw.dtype.name == "category":
    y_str = y_raw.astype(str).str.strip().str.lower()
    unique_vals = y_str.unique()
    val_map = {}
    for v in unique_vals:
        if v in [
            "1",
            "true",
            "yes",
            "y",
            "greater",
            "greater than 30",
            "above",
            "high",
            "1.0",
        ]:
            val_map[v] = 1
        elif v in ["0", "false", "no", "n", "less", "below", "low", "0.0"]:
            val_map[v] = 0
    if len(val_map) == len(unique_vals):
        y = y_str.map(val_map).astype(int)
    else:
        y = (y_str == unique_vals[0]).astype(int)
else:
    y = y_raw.astype(int)

train_features = train_df.drop(columns=[target_col], errors="ignore")

n_train = len(train_features)
df = pd.concat([train_features, test_df], ignore_index=True)

id_cols = [
    c
    for c in df.columns
    if c.lower() in ["id", "index", "row_id", "opeid", "school_id"]
]
features = [c for c in df.columns if c not in id_cols]

num_cols = []
cat_cols = []
for c in features:
    if pd.api.types.is_numeric_dtype(df[c]):
        num_cols.append(c)
    else:
        cat_cols.append(c)

for c in cat_cols:
    df[c] = df[c].astype(str).str.strip().fillna("missing")

# Frequency Encoding
for c in cat_cols:
    freq = df[c].value_counts(normalize=True).to_dict()
    df[f"{c}_freq"] = df[c].map(freq).fillna(0).astype(np.float32)

# Pairwise Categorical Interactions
if len(cat_cols) >= 2:
    for i in range(min(len(cat_cols), 3)):
        for j in range(i + 1, min(len(cat_cols), 3)):
            c1, c2 = cat_cols[i], cat_cols[j]
            inter_col = f"{c1}_{c2}_inter"
            inter_series = df[c1].astype(str) + "_" + df[c2].astype(str)
            freq = inter_series.value_counts(normalize=True).to_dict()
            df[f"{inter_col}_freq"] = (
                inter_series.map(freq).fillna(0).astype(np.float32)
            )

# Group Aggregation Features
if len(cat_cols) > 0 and len(num_cols) > 0:
    grp_cats = [c for c in cat_cols if 2 <= df[c].nunique() <= 500][:3]
    top_nums = num_cols[:5]
    for g_col in grp_cats:
        for n_col in top_nums:
            df[f"{g_col}_{n_col}_mean"] = (
                df[g_col]
                .map(df.groupby(g_col)[n_col].transform("mean"))
                .astype(np.float32)
            )
            df[f"{g_col}_{n_col}_std"] = (
                df[g_col]
                .map(df.groupby(g_col)[n_col].transform("std"))
                .fillna(0)
                .astype(np.float32)
            )

# Missing Value Indicators & Log Transformations
for c in num_cols:
    if df[c].isnull().sum() > 0:
        df[f"{c}_isna"] = df[c].isnull().astype(np.float32)
    med = df[c].median()
    df[c] = df[c].fillna(med)
    if (df[c] >= 0).all() and df[c].skew() > 1.5:
        df[f"{c}_log"] = np.log1p(df[c]).astype(np.float32)

X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

# Out-of-fold Target Encoding
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_te_train = pd.DataFrame(index=X_train.index)
te_test = pd.DataFrame(0.0, index=X_test.index, columns=[f"{c}_te" for c in cat_cols])

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y)):
    for c in cat_cols:
        col_te = f"{c}_te"
        te_map = y.iloc[trn_idx].groupby(X_train.iloc[trn_idx][c]).mean()
        global_mean = y.iloc[trn_idx].mean()
        counts = X_train.iloc[trn_idx][c].value_counts()
        smoothed_map = (counts * te_map + 10 * global_mean) / (counts + 10)
        oof_te_train.loc[val_idx, col_te] = (
            X_train.iloc[val_idx][c].map(smoothed_map).fillna(global_mean)
        )
        te_test[col_te] += X_test[c].map(smoothed_map).fillna(global_mean) / 5.0

for c in cat_cols:
    col_te = f"{c}_te"
    X_train[col_te] = oof_te_train[col_te].astype(np.float32)
    X_test[col_te] = te_test[col_te].astype(np.float32)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))

if HAS_CATBOOST:
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[trn_idx], y.iloc[trn_idx]
        X_va, y_va = X_train.iloc[val_idx], y.iloc[val_idx]

        X_tr_cb = X_tr.copy()
        X_va_cb = X_va.copy()
        X_te_cb = X_test.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(str)
            X_va_cb[c] = X_va_cb[c].astype(str)
            X_te_cb[c] = X_te_cb[c].astype(str)

        model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            eval_metric="AUC",
            random_seed=42 + fold,
            verbose=False,
            cat_features=cat_cols if len(cat_cols) > 0 else None,
        )
        model.fit(
            X_tr_cb,
            y_tr,
            eval_set=(X_va_cb, y_va),
            early_stopping_rounds=50,
            verbose=False,
        )

        oof_preds[val_idx] = model.predict_proba(X_va_cb)[:, 1]
        test_preds += model.predict_proba(X_te_cb)[:, 1] / skf.n_splits
else:
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[trn_idx], y.iloc[trn_idx]
        X_va, y_va = X_train.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            random_state=42 + fold,
            objective="binary",
            metric="auc",
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

cv_score = roc_auc_score(y, oof_preds)
print(f"5-Fold Cross-Validation ROC AUC Score: {cv_score:.5f}")

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "Average or Greater than 30": test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
