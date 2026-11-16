import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/child-mind-institute-problematic-internet-use/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/child-mind-institute-problematic-internet-use/data/test.parquet"
TARGET = "sii"
RANDOM_STATE = 42


def safe_qwk(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    if len(np.unique(y_true)) < 2:
        return 1.0
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def normalize_categorical(series):
    s = series.astype(object)
    s = s.where(s.notna(), "missing")
    s = s.apply(lambda x: str(x).strip() if pd.notna(x) else "missing")
    s = s.replace({"None": "missing", "nan": "missing", "NaN": "missing", "": "missing"})
    return s


def is_categorical(series):
    return series.dtype == object or series.dtype.name in {"category", "string"}


def clean_frame(df, feature_cols, cat_cols, num_cols):
    out = df[feature_cols].copy()

    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    for col in cat_cols:
        out[col] = normalize_categorical(out[col])

    return out


def predict_with_thresholds(probs, a, b):
    pred = np.zeros(probs.shape[0], dtype=int)
    pred[(probs[:, 1] + probs[:, 2]) > a] = 1
    pred[probs[:, 2] > b] = 2
    return pred


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.drop(columns=["id"], errors="ignore")
    test_df = test_df.drop(columns=["id"], errors="ignore")

    y = train_df[TARGET].copy()
    train_df = train_df.drop(columns=[TARGET])

    valid = y.notna()
    y = y[valid].astype(int)
    train_df = train_df.loc[valid]

    feature_cols = [c for c in train_df.columns if c in test_df.columns]

    kept_cols = []
    for col in feature_cols:
        s = train_df[col]
        if is_categorical(s):
            vals = normalize_categorical(s)
            if vals.nunique(dropna=True) > 1:
                kept_cols.append(col)
        else:
            vals = pd.to_numeric(s, errors="coerce")
            if vals.notna().any() and vals.nunique(dropna=True) > 1:
                kept_cols.append(col)

    feature_cols = kept_cols

    if not feature_cols:
        fallback = int(y.mode().iloc[0]) if not y.empty else 0
        pd.DataFrame(
            {
                "id": np.arange(len(test_df)),
                "sii": np.full(len(test_df), fallback, dtype=int),
            }
        ).to_csv("predictions.csv", index=False)
        return

    cat_cols = [c for c in feature_cols if is_categorical(train_df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train = clean_frame(train_df, feature_cols, cat_cols, num_cols)
    X_test = clean_frame(test_df, feature_cols, cat_cols, num_cols)

    for col in cat_cols:
        X_train[col] = X_train[col].astype(object)
        X_test[col] = X_test[col].astype(object)

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    except Exception:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y, test_size=0.2, random_state=RANDOM_STATE
        )

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        cat_features=cat_cols,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=0,
    )

    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None or best_iter <= 0:
        best_iter = 1000

    val_probs = model.predict_proba(X_val)
    class_order = np.argsort(model.classes_)
    val_probs = val_probs[:, class_order]
    classes = np.sort(model.classes_)
    class_to_rank = {c: i for i, c in enumerate(classes)}
    y_val_rank = np.array([class_to_rank[c] for c in y_val])

    argmax_pred = np.argmax(val_probs, axis=1)
    best_score = safe_qwk(y_val_rank, argmax_pred)
    use_thresholds = False
    best_a = 0.5
    best_b = 0.5

    if val_probs.shape[1] == 3:
        grid = np.linspace(0.0, 1.0, 21)
        for a in grid:
            for b in grid:
                if b < a:
                    continue
                pred = predict_with_thresholds(val_probs, a, b)
                score = safe_qwk(y_val_rank, pred)
                if score > best_score:
                    best_score = score
                    best_a = a
                    best_b = b
                    use_thresholds = True

    final_iters = int(max(100, best_iter))
    final_model = CatBoostClassifier(
        iterations=final_iters,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        cat_features=cat_cols,
    )
    final_model.fit(X_train, y, verbose=0)

    test_probs = final_model.predict_proba(X_test)
    test_probs = test_probs[:, class_order]

    if use_thresholds and test_probs.shape[1] == 3:
        test_pred_rank = predict_with_thresholds(test_probs, best_a, best_b)
    else:
        test_pred_rank = np.argmax(test_probs, axis=1)

    rank_to_class = {i: c for i, c in enumerate(classes)}
    test_pred = np.array([rank_to_class[i] for i in test_pred_rank], dtype=int)

    pd.DataFrame(
        {
            "id": np.arange(len(test_df)),
            "sii": test_pred,
        }
    ).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()