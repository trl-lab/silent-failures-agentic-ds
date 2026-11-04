import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

TARGET = "class"
SEED = 42
VAL_FRACTION = 0.2
TARGET_SMOOTHING = 20.0


def clean_target(y: pd.Series) -> pd.Series:
    y = y.copy()
    if y.dtype == object or y.dtype.name == "string":
        y = y.astype(str).str.strip().str.lower()
        mapping = {
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "true": 1,
            "false": 0,
            "churned": 1,
            "not churned": 0,
            "y": 1,
            "n": 0,
            "t": 1,
            "f": 0,
            "churn": 1,
            "no churn": 0,
        }
        y = y.map(mapping)
        if y.isna().any():
            y = y.fillna(pd.to_numeric(y, errors="coerce"))
    else:
        y = pd.to_numeric(y, errors="coerce")
    return y


train_raw = pd.read_parquet("./input/train.parquet")
test_raw = pd.read_parquet("./input/test.parquet")
train_raw = train_raw.reset_index(drop=True)
test_raw = test_raw.reset_index(drop=True)

if TARGET not in train_raw.columns:
    raise ValueError(f"Target column '{TARGET}' not found in train data.")

y_train_raw = clean_target(train_raw[TARGET])
valid_target = y_train_raw.isin([0, 1])
if not valid_target.all():
    train_raw = train_raw.loc[valid_target].copy()
    y_train_raw = y_train_raw.loc[valid_target]
y_train_raw = y_train_raw.astype(int)

if TARGET in test_raw.columns:
    test_raw = test_raw.drop(columns=[TARGET])

test_id = pd.Series(np.arange(len(test_raw)), index=test_raw.index)
feature_cols = [c for c in train_raw.columns if c != TARGET and c.lower() != "id"]

for col in feature_cols:
    if col not in test_raw.columns:
        test_raw[col] = np.nan

if y_train_raw.nunique() > 1:
    train_idx, val_idx = train_test_split(
        train_raw.index,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=y_train_raw,
    )
else:
    train_idx, val_idx = train_test_split(
        train_raw.index,
        test_size=VAL_FRACTION,
        random_state=SEED,
    )

X_train_raw = train_raw.loc[train_idx, feature_cols].copy()
y_train = y_train_raw.loc[train_idx].astype(int)
X_val_raw = train_raw.loc[val_idx, feature_cols].copy()
y_val = y_train_raw.loc[val_idx].astype(int)
X_test_raw = test_raw[feature_cols].copy()

explicit_categoricals = {
    "state",
    "area code",
    "area_code",
    "international plan",
    "international_plan",
    "intl plan",
    "intl_plan",
    "voice mail plan",
    "voice_mail_plan",
    "vmail plan",
    "vmail_plan",
}

numeric_cols = []
categorical_cols = []

for col in feature_cols:
    if (
        col in explicit_categoricals
        or X_train_raw[col].dtype == object
        or X_train_raw[col].dtype.name == "string"
        or pd.api.types.is_categorical_dtype(X_train_raw[col])
    ):
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)

for col in numeric_cols:
    X_train_raw[col] = pd.to_numeric(X_train_raw[col], errors="coerce")

for col in categorical_cols:
    X_train_raw[col] = (
        X_train_raw[col].astype("string").fillna("missing").str.strip().str.lower()
    )

numeric_medians = (
    X_train_raw[numeric_cols].median() if len(numeric_cols) else pd.Series(dtype=float)
)
numeric_q01 = (
    X_train_raw[numeric_cols].quantile(0.01)
    if len(numeric_cols)
    else pd.Series(dtype=float)
)
numeric_q99 = (
    X_train_raw[numeric_cols].quantile(0.99)
    if len(numeric_cols)
    else pd.Series(dtype=float)
)


def clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)

        med = numeric_medians.get(col, 0.0)
        try:
            med = float(med)
        except (TypeError, ValueError):
            med = 0.0
        if not np.isfinite(med):
            med = 0.0
        x = x.fillna(med)

        lo = numeric_q01.get(col, np.nan)
        try:
            lo = float(lo)
        except (TypeError, ValueError):
            lo = np.nan

        hi = numeric_q99.get(col, np.nan)
        try:
            hi = float(hi)
        except (TypeError, ValueError):
            hi = np.nan

        if not np.isfinite(lo):
            lo = float(x.min()) if len(x) else 0.0
        if not np.isfinite(hi):
            hi = float(x.max()) if len(x) else 0.0
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = 0.0
            hi = 0.0
        if lo > hi:
            lo, hi = hi, lo

        if lo == hi:
            x = x.fillna(lo)
        else:
            x = x.clip(lo, hi)

        out[col] = x.astype(float)
    return out


num_train = clean_numeric_frame(X_train_raw)
scaler = StandardScaler().fit(num_train) if len(numeric_cols) else None

prior = float(y_train.mean()) if len(y_train) else 0.0
cat_stats = {}

for col in categorical_cols:
    counts = X_train_raw[col].value_counts(dropna=False)
    sums = y_train.groupby(X_train_raw[col]).sum()
    smoothed = (sums + prior * TARGET_SMOOTHING) / (counts + TARGET_SMOOTHING)
    freq = counts / counts.sum()
    cat_stats[col] = {
        "enc": smoothed,
        "freq": freq,
        "prior": prior,
        "categories": [str(c) for c in counts.index],
    }


def clean_categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in categorical_cols:
        out[col] = df[col].astype("string").fillna("missing").str.strip().str.lower()
    return out


def find_numeric_column(substrings, exclude=()):
    for col in numeric_cols:
        low = col.lower()
        if all(s in low for s in substrings) and not any(e in low for e in exclude):
            return col
    return None


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    num = clean_numeric_frame(raw)
    cat = clean_categorical_frame(raw)

    if len(num.columns) > 0:
        out[num.columns] = num

    if scaler is not None and len(numeric_cols):
        scaled = (num - scaler.mean_) / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        out[[f"sc_{c}" for c in numeric_cols]] = scaled.values

    for col in numeric_cols:
        out[f"log_{col}"] = np.log1p(num[col].clip(lower=0.0))

    for col in categorical_cols:
        x = cat[col]
        stats = cat_stats[col]
        out[f"tenc_{col}"] = x.map(stats["enc"]).fillna(stats["prior"]).astype(float)
        out[f"freq_{col}"] = x.map(stats["freq"]).fillna(0.0).astype(float)

        if len(stats["categories"]) <= 10:
            for i, category in enumerate(stats["categories"]):
                safe_cat = str(category).replace(" ", "_")
                out[f"oh_{col}_{safe_cat}_{i}"] = (x == category).astype(float)

    for col in categorical_cols:
        low = col.lower()
        if "international" in low or "intl" in low:
            out["international_plan"] = (
                cat[col].isin(["yes", "1", "true", "y", "t"]).astype(float)
            )
        if "voice mail" in low or "vmail" in low or "voice_mail" in low:
            out["voice_mail_plan"] = (
                cat[col].isin(["yes", "1", "true", "y", "t"]).astype(float)
            )

    minute_cols = [c for c in numeric_cols if "minute" in c.lower()]
    call_cols = [
        c for c in numeric_cols if "call" in c.lower() and "customer" not in c.lower()
    ]
    charge_cols = [c for c in numeric_cols if "charge" in c.lower()]

    out["total_minutes"] = num[minute_cols].sum(axis=1) if minute_cols else 0.0
    out["total_calls"] = num[call_cols].sum(axis=1) if call_cols else 0.0
    out["total_charge"] = num[charge_cols].sum(axis=1) if charge_cols else 0.0

    for period in ["day", "eve", "night", "intl"]:
        m_col = find_numeric_column(["minute", period])
        c_col = find_numeric_column(["call", period])
        ch_col = find_numeric_column(["charge", period])

        if m_col:
            out[f"{period}_minutes"] = num[m_col]
            out[f"{period}_minute_share"] = num[m_col] / out["total_minutes"].replace(
                0, np.nan
            )
            if c_col:
                out[f"{period}_avg_duration"] = num[m_col] / num[c_col].replace(
                    0, np.nan
                )
        if c_col:
            out[f"{period}_calls"] = num[c_col]
            out[f"{period}_call_share"] = num[c_col] / out["total_calls"].replace(
                0, np.nan
            )
        if ch_col:
            out[f"{period}_charge"] = num[ch_col]
            out[f"{period}_charge_share"] = num[ch_col] / out["total_charge"].replace(
                0, np.nan
            )
            if m_col:
                out[f"{period}_charge_per_minute"] = num[ch_col] / num[m_col].replace(
                    0, np.nan
                )

    cs_col = find_numeric_column(["customer", "call"]) or find_numeric_column(
        ["service", "call"]
    )
    if cs_col:
        out["customer_service_calls"] = num[cs_col]
        out["log_customer_service_calls"] = np.log1p(num[cs_col].clip(lower=0.0))
        out["customer_service_calls_sq"] = num[cs_col] ** 2
        if "international_plan" in out.columns:
            out["service_x_intl"] = num[cs_col] * out["international_plan"]
        if "total_minutes" in out.columns:
            out["service_x_minutes"] = num[cs_col] * out["total_minutes"]
        if "intl_minutes" in out.columns:
            out["service_x_intl_minutes"] = num[cs_col] * out["intl_minutes"]

    age_col = find_numeric_column(["account", "length"]) or find_numeric_column(["age"])
    if age_col:
        out["account_length"] = num[age_col]
        out["log_account_length"] = np.log1p(num[age_col].clip(lower=0.0))
        if "international_plan" in out.columns:
            out["age_x_intl"] = num[age_col] * out["international_plan"]

    vm_col = find_numeric_column(["vmail", "message"]) or find_numeric_column(
        ["voice mail", "message"]
    )
    if vm_col:
        out["vmail_messages"] = num[vm_col]
        out["log_vmail_messages"] = np.log1p(num[vm_col].clip(lower=0.0))
        if "voice_mail_plan" in out.columns:
            out["vmail_x_plan"] = num[vm_col] * out["voice_mail_plan"]

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return out


X_train = build_features(X_train_raw)
X_val = build_features(X_val_raw)
X_test = build_features(X_test_raw)

feature_names = list(X_train.columns)
X_val = X_val.reindex(columns=feature_names, fill_value=0.0).astype(float)
X_test = X_test.reindex(columns=feature_names, fill_value=0.0).astype(float)

y_train_for_weight = y_train
pos = int((y_train_for_weight == 1).sum())
neg = int((y_train_for_weight == 0).sum())
if pos > 0 and neg > 0:
    scale_pos_weight = float(neg / pos) ** 0.5
else:
    scale_pos_weight = 1.0

monotone_map = {}
for col in feature_names:
    low = str(col).lower()

    if (
        (
            "customer_service_calls" in low
            or "customer call" in low
            or ("customer" in low and "call" in low and "service" in low)
        )
        and "sq" not in low
        and "log" not in low
        and "x_" not in low
    ):
        monotone_map[col] = 1

    if low in {"international_plan", "intl_plan", "international plan", "intl plan"}:
        monotone_map[col] = 1

    if low in {"voice_mail_plan", "vmail_plan", "voice mail plan", "vmail plan"}:
        monotone_map[col] = -1

monotone_constraints = [int(monotone_map.get(col, 0)) for col in feature_names]

model_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 3000,
    "min_child_samples": 20,
    "min_split_gain": 0.01,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "scale_pos_weight": float(scale_pos_weight),
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

if len(feature_names) > 0:
    model_params["monotone_constraints"] = monotone_constraints

model = lgb.LGBMClassifier(**model_params)

if y_val.nunique() > 1:
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
else:
    model.fit(X_train, y_train)

best_iter = getattr(model, "best_iteration_", None)
if best_iter is None or best_iter <= 0:
    best_iter = int(model_params["n_estimators"])
else:
    best_iter = int(best_iter)

val_pred = model.predict_proba(X_val)[:, 1]
val_pred = np.nan_to_num(val_pred, nan=0.5, posinf=1.0, neginf=0.0)
val_pred = np.clip(val_pred, 0.0, 1.0)

if y_val.nunique() < 2 or np.unique(val_pred).size < 2:
    score = 0.5
else:
    score = float(roc_auc_score(y_val, val_pred))

final_params = dict(model_params)
final_params["n_estimators"] = best_iter
if "monotone_constraints" in final_params:
    final_params["monotone_constraints"] = list(final_params["monotone_constraints"])

final_model = lgb.LGBMClassifier(**final_params)
X_all = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_all = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
final_model.fit(X_all, y_all)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.nan_to_num(test_pred, nan=0.5, posinf=1.0, neginf=0.0)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "class": test_pred,
    }
)
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")
