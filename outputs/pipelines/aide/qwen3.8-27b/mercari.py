import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

os.makedirs("./working", exist_ok=True)


def get_series(df: pd.DataFrame, col: str, default="") -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    name = get_series(df, "name", "").fillna("").astype(str)
    desc = get_series(df, "itemdesc", "").fillna("").astype(str)

    feats = pd.DataFrame(index=df.index)
    feats["name_len"] = name.str.len()
    feats["desc_len"] = desc.str.len()
    feats["name_words"] = name.str.split().str.len().fillna(0)
    feats["desc_words"] = desc.str.split().str.len().fillna(0)
    feats["desc_name_len_ratio"] = feats["desc_len"] / (feats["name_len"] + 1)
    feats["has_desc"] = (feats["desc_len"] > 0).astype(int)
    return feats


def add_category_parts(df: pd.DataFrame) -> pd.DataFrame:
    cat = get_series(df, "category_name", "").fillna("").astype(str)
    parts = cat.str.split(r"\s*>\s*", expand=True, regex=True)

    for i in range(3):
        col = f"cat{i + 1}"
        if i in parts.columns:
            parts[col] = parts[i]
        else:
            parts[col] = "Unknown"

    parts = parts.fillna("Unknown")
    return parts[["cat1", "cat2", "cat3"]]


def prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["shipping"] = pd.to_numeric(get_series(df, "shipping", np.nan), errors="coerce")
    out["item_condition"] = pd.to_numeric(
        get_series(df, "item_condition", np.nan), errors="coerce"
    )

    out = pd.concat([out, add_text_features(df), add_category_parts(df)], axis=1)

    out["brand"] = (
        get_series(df, "brand", "__missing__").fillna("__missing__").astype(str)
    )
    out["class_name"] = (
        get_series(df, "class_name", "__missing__").fillna("__missing__").astype(str)
    )

    return out


CAT_COLS = ["brand", "class_name", "cat1", "cat2", "cat3"]


def to_categorical(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    out = df.copy()
    for col in CAT_COLS:
        out[col] = pd.Categorical(out[col], categories=categories[col])
    return out


def make_frequency_maps(base_df: pd.DataFrame) -> dict:
    return {col: base_df[col].value_counts(normalize=True) for col in CAT_COLS}


def add_frequency_features(
    cat_df: pd.DataFrame,
    base_df: pd.DataFrame,
    frequency_maps: dict,
) -> pd.DataFrame:
    out = cat_df.copy()
    for col in CAT_COLS:
        out[f"freq_{col}"] = (
            base_df[col].map(frequency_maps[col]).fillna(0.0).astype(float)
        )
    return out


train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train = train.dropna(subset=["price"]).reset_index(drop=True)
y = train["price"].astype(float)

base_train = prepare_base_features(train)
base_test = prepare_base_features(test)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": -1,
    "verbosity": -1,
    "seed": 42,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_pred = np.zeros(len(base_train), dtype=float)
best_iters = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(base_train), start=1):
    tr_base = base_train.iloc[tr_idx]
    val_base = base_train.iloc[val_idx]

    fold_categories = {col: pd.unique(tr_base[col]) for col in CAT_COLS}

    tr_cat = to_categorical(tr_base, fold_categories)
    val_cat = to_categorical(val_base, fold_categories)

    fold_freq_maps = make_frequency_maps(tr_base)

    tr_feat = add_frequency_features(tr_cat, tr_base, fold_freq_maps)
    val_feat = add_frequency_features(val_cat, val_base, fold_freq_maps)

    dtrain = lgb.Dataset(
        tr_feat,
        label=y.iloc[tr_idx],
        categorical_feature=CAT_COLS,
    )
    dvalid = lgb.Dataset(
        val_feat,
        label=y.iloc[val_idx],
        categorical_feature=CAT_COLS,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )

    best_iter = max(1, int(model.best_iteration))
    best_iters.append(best_iter)

    val_pred[val_idx] = model.predict(val_feat, num_iteration=best_iter)

    fold_rmse = root_mean_squared_error(y.iloc[val_idx], val_pred[val_idx])
    print(f"Fold {fold} RMSE: {fold_rmse:.4f}")

cv_rmse = root_mean_squared_error(y, val_pred)
print(f"CV RMSE: {cv_rmse:.4f}")

final_rounds = max(1, int(np.mean(best_iters))) if best_iters else 1

full_categories = {col: pd.unique(base_train[col]) for col in CAT_COLS}

X_train_full_cat = to_categorical(base_train, full_categories)
X_test_cat = to_categorical(base_test, full_categories)

full_freq_maps = make_frequency_maps(base_train)

X_train_full = add_frequency_features(X_train_full_cat, base_train, full_freq_maps)
X_test = add_frequency_features(X_test_cat, base_test, full_freq_maps)

dtrain_full = lgb.Dataset(
    X_train_full,
    label=y,
    categorical_feature=CAT_COLS,
)

final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_pred = final_model.predict(X_test, num_iteration=final_rounds)
test_pred = np.clip(
    np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None
)

submission = pd.DataFrame(
    {
        "id": np.arange(len(X_test), dtype=int),
        "price": test_pred,
    }
)

submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
