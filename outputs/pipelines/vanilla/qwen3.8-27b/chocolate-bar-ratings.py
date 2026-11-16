import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/chocolate-bar-ratings/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/chocolate-bar-ratings/data/test.parquet"
TARGET = "Rating"


def tokenize_text(value):
    if pd.isna(value):
        return []
    s = str(value).lower()
    return [t.strip() for t in re.split(r"[,;|]+", s) if t.strip()]


def extract_percent(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return np.nan
    return np.nan


def ingredient_count(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan

    m = re.match(r"^\s*(\d+)", s)
    if m:
        return float(int(m.group(1)))

    if "-" in s:
        s = s.split("-", 1)[1]

    parts = [p.strip() for p in s.split(",") if p.strip()]
    return float(len(parts)) if parts else np.nan


def count_items(value):
    return len(tokenize_text(value))


def add_basic_features(df):
    df = df.copy()

    if "Cocoa Percent" in df.columns:
        df["cocoa_percent_num"] = df["Cocoa Percent"].apply(extract_percent)

    if "Ingredients" in df.columns:
        df["ingredient_count"] = df["Ingredients"].apply(ingredient_count)

    if "Most Memorable Characteristics" in df.columns:
        df["n_characteristics"] = df["Most Memorable Characteristics"].apply(count_items)

    if "Review Date" in df.columns:
        df["review_date_num"] = pd.to_numeric(df["Review Date"], errors="coerce")

    return df


def build_text_vocab(df, col, max_terms=200, min_count=2):
    if col not in df.columns:
        return []

    counts = Counter()
    for value in df[col].dropna():
        for token in tokenize_text(value):
            counts[token] += 1

    terms = [term for term, count in counts.items() if count >= min_count]
    if not terms:
        terms = list(counts.keys())

    terms.sort(key=lambda term: (-counts[term], term))
    return terms[:max_terms]


def add_text_features(df, col, vocab):
    if col not in df.columns or not vocab:
        return df.copy()

    df = df.copy()
    token_sets = [set(tokenize_text(value)) for value in df[col]]

    for i, term in enumerate(vocab):
        df[f"char_{i}"] = [1 if term in token_set else 0 for token_set in token_sets]

    return df


def align_columns(train_df, test_df):
    common = [c for c in train_df.columns if c in test_df.columns]
    train_df = train_df[common].copy()
    test_df = test_df[common].copy()

    if train_df.shape[1] == 0:
        return train_df, test_df

    train_df = train_df.dropna(axis=1, how="all")
    test_df = test_df.reindex(columns=train_df.columns)

    const = [c for c in train_df.columns if train_df[c].nunique(dropna=True) <= 1]
    if const:
        train_df = train_df.drop(columns=const)
        test_df = test_df.drop(columns=const)

    return train_df, test_df


def make_split(X, y):
    if len(X) < 20 or y.nunique() <= 1:
        return X, X.iloc[0:0], y, y.iloc[0:0]

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    if y_tr.nunique() <= 1 or y_val.nunique() <= 1:
        return X, X.iloc[0:0], y, y.iloc[0:0]

    return X_tr, X_val, y_tr, y_val


def get_cat_cols(df):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def encode_cats(train_df, test_df, cat_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for c in cat_cols:
        if c not in train_df.columns:
            continue

        train_series = train_df[c].fillna("missing").astype(str)
        cats = pd.unique(train_series)

        train_df[c] = pd.Categorical(train_series, categories=cats)

        if c in test_df.columns:
            test_series = test_df[c].fillna("missing").astype(str)
            test_df[c] = pd.Categorical(test_series, categories=cats)
        else:
            test_df[c] = pd.Categorical(
                pd.Series(["missing"] * len(test_df), index=test_df.index),
                categories=cats,
            )

    return train_df, test_df


def aligned_proba(proba, model_classes, all_classes, all_idx):
    out = np.zeros((proba.shape[0], len(all_classes)), dtype=np.float64)

    for i, cls in enumerate(model_classes):
        idx = all_idx.get(cls)
        if idx is not None:
            out[:, idx] = proba[:, i]

    if not np.any(out):
        out[:, 0] = 1.0

    return out


def make_catboost(cat_cols, iterations=1000):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        cat_features=cat_cols if cat_cols else None,
    )


def fit_catboost(model, X_tr, y_tr, X_val, y_val):
    if len(X_val) > 0:
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0,
        )
    else:
        model.fit(X_tr, y_tr, verbose=0)
    return model


def make_lgbm(cat_cols, n_estimators=1000):
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        categorical_feature=cat_cols if cat_cols else [],
    )


def fit_lgbm(model, X_tr, y_tr, X_val, y_val):
    if len(X_val) > 0:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            early_stopping_rounds=50,
            verbose=False,
        )
    else:
        model.fit(X_tr, y_tr, verbose=False)
    return model


def get_best_iterations(model, default):
    best = getattr(model, "best_iteration_", None)
    if best is not None:
        try:
            best = int(best)
            if best >= 0:
                return max(1, best + 1)
        except Exception:
            pass
    return default


def save_predictions(n, preds):
    preds = np.asarray(preds)
    if len(preds) != n:
        preds = np.resize(preds, n)
    pd.DataFrame({"id": np.arange(n), "Rating": preds}).to_csv(
        "predictions.csv", index=False
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        save_predictions(len(test_df), np.array(["0"] * len(test_df)))
        return

    train_df = train_df.dropna(subset=[TARGET])
    if len(train_df) == 0:
        save_predictions(len(test_df), np.array(["0"] * len(test_df)))
        return

    y = train_df[TARGET].astype(str).str.strip()
    mask = y.notna() & (y != "")
    y = y[mask]
    X_raw = train_df.loc[mask].drop(columns=[TARGET]).copy()

    if len(y) == 0:
        save_predictions(len(test_df), np.array(["0"] * len(test_df)))
        return

    X_test_raw = test_df.copy()
    if TARGET in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[TARGET])

    X_raw, X_test_raw = align_columns(X_raw, X_test_raw)

    if X_raw.shape[1] == 0:
        mode_val = y.mode().iloc[0] if len(y) > 0 else "0"
        save_predictions(len(X_test_raw), np.array([mode_val] * len(X_test_raw)))
        return

    if y.nunique() <= 1:
        save_predictions(len(X_test_raw), np.array([y.iloc[0]] * len(X_test_raw)))
        return

    X_tr_raw, X_val_raw, y_tr, y_val = make_split(X_raw, y)

    X_tr = add_basic_features(X_tr_raw)
    X_val = add_basic_features(X_val_raw)

    vocab_tr = build_text_vocab(X_tr, "Most Memorable Characteristics", max_terms=200)
    X_tr = add_text_features(X_tr, "Most Memorable Characteristics", vocab_tr)
    X_val = add_text_features(X_val, "Most Memorable Characteristics", vocab_tr)
    X_val = X_val.reindex(columns=X_tr.columns)

    cat_cols_val = get_cat_cols(X_tr)
    all_classes = np.unique(y)
    all_idx = {c: i for i, c in enumerate(all_classes)}

    cb = None
    cb_val_proba = None
    try:
        cb = make_catboost(cat_cols_val)
        cb = fit_catboost(cb, X_tr, y_tr, X_val, y_val)
        cb_val_proba = cb.predict_proba(X_val) if len(X_val) > 0 else None
    except Exception:
        cb = None

    lgbm = None
    lgbm_val_proba = None
    try:
        X_tr_enc, X_val_enc = encode_cats(X_tr, X_val, cat_cols_val)
        lgbm = make_lgbm(cat_cols_val)
        lgbm = fit_lgbm(lgbm, X_tr_enc, y_tr, X_val_enc, y_val)
        lgbm_val_proba = lgbm.predict_proba(X_val_enc) if len(X_val_enc) > 0 else None
    except Exception:
        lgbm = None

    use_lgbm = False
    if cb_val_proba is not None and len(X_val) > 0:
        cb_val_aligned = aligned_proba(cb_val_proba, cb.classes_, all_classes, all_idx)
        cb_val_acc = accuracy_score(y_val, all_classes[np.argmax(cb_val_aligned, axis=1)])

        if lgbm_val_proba is not None:
            lgbm_val_aligned = aligned_proba(
                lgbm_val_proba, lgbm.classes_, all_classes, all_idx
            )
            ens_val_aligned = (cb_val_aligned + lgbm_val_aligned) / 2.0
            ens_val_acc = accuracy_score(
                y_val, all_classes[np.argmax(ens_val_aligned, axis=1)]
            )
            use_lgbm = ens_val_acc >= cb_val_acc
    elif lgbm_val_proba is not None:
        use_lgbm = True
    else:
        use_lgbm = lgbm is not None

    X_full = add_basic_features(X_raw)
    X_test = add_basic_features(X_test_raw)

    vocab_full = build_text_vocab(X_full, "Most Memorable Characteristics", max_terms=200)
    X_full = add_text_features(X_full, "Most Memorable Characteristics", vocab_full)
    X_test = add_text_features(X_test, "Most Memorable Characteristics", vocab_full)
    X_test = X_test.reindex(columns=X_full.columns)

    if len(X_test) == 0:
        save_predictions(0, np.array([]))
        return

    cat_cols_full = get_cat_cols(X_full)

    cb_full = None
    cb_test_proba = None
    if cb is not None:
        try:
            cb_iters = get_best_iterations(cb, 1000)
            cb_full = make_catboost(cat_cols_full, iterations=cb_iters)
            cb_full.fit(X_full, y, verbose=0)
            cb_test_proba = cb_full.predict_proba(X_test)
        except Exception:
            cb_full = None

    lgbm_full = None
    lgbm_test_proba = None
    if lgbm is not None and (use_lgbm or cb_full is None):
        try:
            lgbm_iters = get_best_iterations(lgbm, 1000)
            X_full_enc, X_test_enc = encode_cats(X_full, X_test, cat_cols_full)
            lgbm_full = make_lgbm(cat_cols_full, n_estimators=lgbm_iters)
            lgbm_full.fit(X_full_enc, y, verbose=False)
            lgbm_test_proba = lgbm_full.predict_proba(X_test_enc)
        except Exception:
            lgbm_full = None

    if cb_full is None and lgbm_full is None:
        mode_val = y.mode().iloc[0] if len(y) > 0 else "0"
        save_predictions(len(X_test), np.array([mode_val] * len(X_test)))
        return

    probas = []
    if cb_full is not None:
        probas.append(
            aligned_proba(cb_test_proba, cb_full.classes_, all_classes, all_idx)
        )

    if lgbm_full is not None:
        probas.append(
            aligned_proba(lgbm_test_proba, lgbm_full.classes_, all_classes, all_idx)
        )

    final_proba = probas[0]
    for p in probas[1:]:
        final_proba = final_proba + p
    final_proba = final_proba / len(probas)

    preds = all_classes[np.argmax(final_proba, axis=1)]
    save_predictions(len(X_test), preds)


if __name__ == "__main__":
    main()