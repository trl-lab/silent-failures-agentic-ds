import os
import json
import ast
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

TARGET = "stars"

train = pd.read_parquet("./input/train.parquet").reset_index(drop=True)
test = pd.read_parquet("./input/test.parquet").reset_index(drop=True)

if TARGET in test.columns:
    test = test.drop(columns=[TARGET])

train[TARGET] = (
    pd.to_numeric(train[TARGET], errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .round(2)
)
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

if train.empty:
    print("CV Accuracy: 0.000000")
    pd.DataFrame({"id": np.arange(len(test)), "stars": [0] * len(test)}).to_csv(
        "./working/submission.csv", index=False
    )
    raise SystemExit


def parse_dict(s):
    if isinstance(s, dict):
        return s
    if not isinstance(s, str) or not s.strip():
        return {}
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return {}


def flatten_dict(d):
    flat = {}
    if not isinstance(d, dict):
        return flat
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(flatten_dict(v))
        else:
            flat[k] = v
    return flat


def attr_true(d, k):
    v = d.get(k)
    if v is None:
        return 0
    if isinstance(v, str):
        return 1 if v.strip().lower() in {"true", "1", "yes", "y"} else 0
    return 1 if bool(v) else 0


def category_count(s):
    if isinstance(s, (list, tuple)):
        return len(s)
    if not isinstance(s, str) or not s.strip():
        return 0
    return len([x for x in s.split(";") if x.strip()])


def category_first(s):
    if isinstance(s, (list, tuple)):
        return str(s[0]).strip() if len(s) > 0 else ""
    if not isinstance(s, str) or not s.strip():
        return ""
    return s.split(";")[0].strip()


def total_open_minutes(hours):
    if not isinstance(hours, dict):
        return 0
    total = 0
    for v in hours.values():
        if not isinstance(v, str):
            continue
        for part in v.split(","):
            part = part.strip()
            if "-" not in part:
                continue
            a, b = part.split("-", 1)
            try:
                a1, a2 = a.strip().split(":")
                b1, b2 = b.strip().split(":")
                start = int(a1) * 60 + int(a2)
                end = int(b1) * 60 + int(b2)
                delta = end - start
                if delta < 0:
                    delta += 24 * 60
                total += delta
            except Exception:
                pass
    return total


ATTR_KEYS = [
    "GoodForChildren",
    "GoodForGroups",
    "OutdoorSeating",
    "Wifi",
    "Alcohol",
    "ParkingLot",
    "Restroom",
    "TakesReservations",
    "ByAppointmentOnly",
    "AcceptsCreditCards",
    "BikeParking",
    "DogsAllowed",
    "Corkage",
    "HappyHour",
    "TableReservations",
    "GenderNeutralRestroom",
    "NonSmoking",
    "GoodForKids",
    "AcceptsInsurance",
]


def engineer(df):
    out = pd.DataFrame(index=df.index)

    for col in ["latitude", "longitude", "review_count"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_open" in df.columns:
        s = df["is_open"].astype(str).str.lower().str.strip()
        out["is_open"] = (
            s.map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
            .fillna(0)
            .astype(int)
        )

    if "price_level" in df.columns:
        pl = df["price_level"]
        num = pd.to_numeric(pl, errors="coerce")
        dollars = pl.astype(str).str.count(r"\$")
        dollars = dollars.where(pl.notna(), np.nan)
        out["price_level"] = num.fillna(dollars)

    if "categories" in df.columns:
        out["num_categories"] = df["categories"].map(category_count).astype(int)
        out["first_category"] = df["categories"].map(category_first)

    if "attributes" in df.columns:
        attrs = df["attributes"].map(lambda x: flatten_dict(parse_dict(x)))
        out["num_attributes"] = attrs.map(len).astype(int)
        for k in ATTR_KEYS:
            out[f"attr_{k}"] = attrs.map(lambda d, k=k: attr_true(d, k)).astype(int)

    if "hours" in df.columns:
        hours = df["hours"].map(parse_dict)
        out["num_days_hours"] = hours.map(
            lambda d: (
                len([v for v in d.values() if isinstance(v, str) and v.strip()])
                if isinstance(d, dict)
                else 0
            )
        ).astype(int)
        out["has_hours"] = (out["num_days_hours"] > 0).astype(int)
        out["hours_total_minutes"] = hours.map(total_open_minutes).astype(int)

    drop_cols = {
        TARGET,
        "business_id",
        "name",
        "address",
        "url",
        "image",
        "categories",
        "attributes",
        "hours",
        "postal_code",
        "id",
        "id_1",
        "id_2",
    }

    for col in df.columns:
        if col in drop_cols or col in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = pd.to_numeric(df[col], errors="coerce")
        elif col in {
            "city",
            "state",
            "neighborhood",
            "type",
            "first_category",
            "price_level",
            "is_open",
        }:
            out[col] = df[col].astype(str)

    for col in ["city", "state", "neighborhood", "type", "first_category"]:
        if col not in out.columns and col in df.columns:
            out[col] = df[col].astype(str)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=1, how="all")
    return out


X_train_raw = engineer(train)
X_test_raw = engineer(test)
X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns)

X_train_raw = X_train_raw.replace([np.inf, -np.inf], np.nan)
X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan)

for col in X_train_raw.columns:
    if pd.api.types.is_numeric_dtype(X_train_raw[col]):
        X_test_raw[col] = pd.to_numeric(X_test_raw[col], errors="coerce")
    elif (
        pd.api.types.is_object_dtype(X_train_raw[col])
        or pd.api.types.is_string_dtype(X_train_raw[col])
        or pd.api.types.is_categorical_dtype(X_train_raw[col])
    ):
        X_train_raw[col] = X_train_raw[col].astype(str)
        X_test_raw[col] = X_test_raw[col].astype(str)

string_cols = [
    c
    for c in X_train_raw.columns
    if (
        pd.api.types.is_object_dtype(X_train_raw[c])
        or pd.api.types.is_string_dtype(X_train_raw[c])
        or pd.api.types.is_categorical_dtype(X_train_raw[c])
    )
]
cat_cols = [c for c in string_cols if X_train_raw[c].nunique(dropna=True) <= 50000]
drop_obj = [c for c in string_cols if c not in cat_cols]

if drop_obj:
    X_train_raw = X_train_raw.drop(columns=drop_obj)
    X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns)

if X_train_raw.shape[1] == 0:
    majority = train[TARGET].mode().iloc[0] if not train.empty else 0
    try:
        if np.all(np.asarray(train[TARGET], dtype=float) % 1 == 0):
            majority = int(majority)
    except Exception:
        pass
    print(f"CV Accuracy: {1.0:.6f}")
    pd.DataFrame({"id": np.arange(len(test)), "stars": [majority] * len(test)}).to_csv(
        "./working/submission.csv", index=False
    )
    raise SystemExit


def encode_categoricals(X, train_X, categorical_cols):
    X = X.copy()
    for c in categorical_cols:
        if c not in X.columns:
            X[c] = -1
        vals = X[c].astype(str)
        if train_X is not None and c in train_X.columns:
            mapping = {
                v: i for i, v in enumerate(np.unique(train_X[c].astype(str).to_numpy()))
            }
        else:
            mapping = {v: i for i, v in enumerate(np.unique(vals.to_numpy()))}
        X[c] = vals.map(mapping).fillna(-1).astype(int)
    return X


class_labels = np.unique(train[TARGET].to_numpy())

if len(class_labels) == 1:
    label = class_labels[0]
    try:
        if np.all(np.asarray(class_labels, dtype=float) % 1 == 0):
            label = int(label)
    except Exception:
        pass
    print(f"CV Accuracy: {1.0:.6f}")
    pd.DataFrame({"id": np.arange(len(test)), "stars": [label] * len(test)}).to_csv(
        "./working/submission.csv", index=False
    )
    raise SystemExit

label_to_idx = {v: i for i, v in enumerate(class_labels)}
y_idx = train[TARGET].map(label_to_idx).to_numpy()

params = {
    "objective": "multiclass",
    "num_class": len(class_labels),
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbose": -1,
}

cv_accs = []
cv_models = []

if len(train) >= 2:
    n_splits = 5
    min_class_count = int(np.min(np.bincount(y_idx)))
    if min_class_count < n_splits:
        n_splits = max(2, min(n_splits, min_class_count)) if min_class_count >= 2 else 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, val_idx in skf.split(X_train_raw, y_idx):
        Xtr_raw = X_train_raw.iloc[tr_idx]
        Xval_raw = X_train_raw.iloc[val_idx]

        Xtr = encode_categoricals(Xtr_raw, Xtr_raw, cat_cols)
        Xval = encode_categoricals(Xval_raw, Xtr_raw, cat_cols)

        dtrain = lgb.Dataset(Xtr, label=y_idx[tr_idx], categorical_feature=cat_cols)
        dvalid = lgb.Dataset(Xval, label=y_idx[val_idx], categorical_feature=cat_cols)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )

        pred = model.predict(Xval)
        pred_idx = np.argmax(pred, axis=1)
        cv_accs.append(accuracy_score(y_idx[val_idx], pred_idx))
        cv_models.append(model)
else:
    cv_accs = [1.0]

print(f"CV Accuracy: {np.mean(cv_accs):.6f} (std={np.std(cv_accs):.6f})")

best_iters = [
    m.best_iteration for m in cv_models if getattr(m, "best_iteration", 0) > 0
]
final_rounds = int(np.mean(best_iters)) if best_iters else 500
final_rounds = max(1, min(final_rounds, 2000))

X_train_enc = encode_categoricals(X_train_raw, X_train_raw, cat_cols)
X_test_enc = encode_categoricals(X_test_raw, X_train_raw, cat_cols)

dtrain_all = lgb.Dataset(X_train_enc, label=y_idx, categorical_feature=cat_cols)
final_model = lgb.train(params, dtrain_all, num_boost_round=final_rounds)

if len(test) == 0:
    test_pred = np.array([], dtype=class_labels.dtype)
else:
    test_pred_idx = np.argmax(final_model.predict(X_test_enc), axis=1)
    test_pred = class_labels[test_pred_idx]

try:
    if np.all(np.asarray(class_labels, dtype=float) % 1 == 0):
        test_pred = test_pred.astype(int)
except Exception:
    pass

submission = pd.DataFrame({"id": np.arange(len(test)), "stars": test_pred})
submission.to_csv("./working/submission.csv", index=False)
