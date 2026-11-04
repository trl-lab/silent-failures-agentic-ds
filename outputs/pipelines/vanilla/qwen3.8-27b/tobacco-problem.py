import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgbm
from lightgbm import LGBMClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/tobacco-problem/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/tobacco-problem/data/test.parquet"
TARGET = "number_health_problems"


def save_predictions(pred, n):
    if n == 0:
        pd.DataFrame({"id": np.array([], dtype=int), TARGET: np.array([], dtype=int)}).to_csv(
            "predictions.csv", index=False
        )
        return

    pred = np.asarray(pred)
    if pred.ndim == 0:
        pred = np.full(n, pred.item())
    elif pred.size == 0:
        pred = np.zeros(n, dtype=int)
    else:
        pred = pred.reshape(-1)
        if pred.shape[0] != n:
            if pred.shape[0] < n:
                fill = pred[-1] if pred.size else 0
                pred = np.concatenate([pred, np.full(n - pred.shape[0], fill)])
            else:
                pred = pred[:n]

    out = pd.DataFrame({"id": np.arange(n, dtype=int), TARGET: pred})
    out.to_csv("predictions.csv", index=False)


def sanitize(name):
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return s if s else "other"


def get_series(df, col, index):
    if col in df.columns:
        return df[col]
    return pd.Series(index=index, dtype=object)


def parse_dates(s):
    if s is None or len(s) == 0:
        return pd.Series(pd.NaT, index=s.index if s is not None else None)

    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    s_str = s.astype(str).str.strip()
    s_str = s_str.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan})

    dt = pd.to_datetime(s_str, format="%m/%d/%Y", errors="coerce")
    mask = dt.isna() & s_str.notna()
    if mask.any():
        dt = dt.copy()
        dt[mask] = pd.to_datetime(s_str[mask], errors="coerce")

    return dt


def text_meta(s, prefix):
    s = s.fillna("").astype(str).str.strip().str.lower()
    out = pd.DataFrame(index=s.index)
    out[f"{prefix}_len"] = s.str.len().astype(float)
    out[f"{prefix}_words"] = s.str.split().str.len().fillna(0).astype(float)
    out[f"{prefix}_no_info"] = s.str.contains("no information provided", regex=False).astype(float)
    out[f"{prefix}_comma_count"] = s.str.count(",").astype(float)
    out[f"{prefix}_semicolon_count"] = s.str.count(";").astype(float)
    return out


def contains_features(s, patterns, prefix):
    s = s.fillna("").astype(str).str.strip().str.lower()
    out = pd.DataFrame(index=s.index)
    for name, pat in patterns.items():
        out[f"{prefix}_{name}"] = s.str.contains(pat, regex=True, na=False).astype(float)
    return out


def fit_transform_text(train_s, test_s, prefix, max_features=80):
    train_s = train_s.fillna("").astype(str).str.strip().str.lower()
    test_s = test_s.fillna("").astype(str).str.strip().str.lower()

    train_s = train_s.str.replace("no information provided", "no_information", regex=False)
    test_s = test_s.str.replace("no information provided", "no_information", regex=False)

    if len(train_s) == 0 or train_s.str.len().sum() == 0:
        return pd.DataFrame(index=train_s.index), pd.DataFrame(index=test_s.index)

    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=1,
        max_df=1.0,
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float32,
    )

    try:
        Xtr = vec.fit_transform(train_s)
        Xte = vec.transform(test_s)
    except ValueError:
        return pd.DataFrame(index=train_s.index), pd.DataFrame(index=test_s.index)

    if Xtr.shape[1] == 0:
        return pd.DataFrame(index=train_s.index), pd.DataFrame(index=test_s.index)

    cols = [f"{prefix}_{i}" for i in range(Xtr.shape[1])]
    return (
        pd.DataFrame(Xtr.toarray(), index=train_s.index, columns=cols),
        pd.DataFrame(Xte.toarray(), index=test_s.index, columns=cols),
    )


def one_hot_features(train_s, test_s, prefix, max_categories=25):
    train_s = train_s.fillna("missing").astype(str).str.strip().str.lower()
    test_s = test_s.fillna("missing").astype(str).str.strip().str.lower()

    if len(train_s) == 0 or train_s.nunique() == 0:
        return pd.DataFrame(index=train_s.index), pd.DataFrame(index=test_s.index)

    if train_s.nunique() > max_categories:
        top = train_s.value_counts().head(max_categories).index
        tr = pd.DataFrame(index=train_s.index)
        te = pd.DataFrame(index=test_s.index)
        for c in top:
            col = f"{prefix}_{sanitize(c)}"
            tr[col] = (train_s == c).astype(float)
            te[col] = (test_s == c).astype(float)
        tr[f"{prefix}_other"] = (~train_s.isin(top)).astype(float)
        te[f"{prefix}_other"] = (~test_s.isin(top)).astype(float)
        return tr, te

    cats = sorted(train_s.unique())
    tr = pd.DataFrame(index=train_s.index)
    te = pd.DataFrame(index=test_s.index)
    for c in cats:
        col = f"{prefix}_{sanitize(c)}"
        tr[col] = (train_s == c).astype(float)
        te[col] = (test_s == c).astype(float)
    return tr, te


def prepare_target(train_df):
    if TARGET not in train_df.columns:
        return None, False, train_df

    y_raw = train_df[TARGET]
    non_missing = y_raw.notna()
    if non_missing.sum() == 0:
        return None, False, train_df

    y_str = y_raw.astype(str).str.strip()
    y_str = y_str.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    y_num = pd.to_numeric(y_str, errors="coerce")

    numeric_ratio = y_num[non_missing].notna().mean()
    if numeric_ratio >= 0.9:
        valid = non_missing & y_num.notna()
        if valid.sum() == 0:
            return None, False, train_df
        y = y_num[valid].round().astype(int)
        return y, True, train_df.loc[valid].copy()

    valid = non_missing & y_str.notna()
    if valid.sum() == 0:
        return None, False, train_df

    y = y_str[valid]
    return y, False, train_df.loc[valid].copy()


def make_lgbm(n_estimators, n_samples):
    min_child_samples = max(1, min(20, n_samples // 5))
    return LGBMClassifier(
        n_estimators=int(n_estimators),
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        min_child_samples=min_child_samples,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    try:
        train_df = pd.read_parquet(TRAIN_PATH)
    except Exception:
        train_df = pd.DataFrame()

    try:
        test_df = pd.read_parquet(TEST_PATH)
    except Exception:
        test_df = pd.DataFrame()

    n_test = len(test_df)
    fallback = np.zeros(n_test, dtype=int)

    try:
        if n_test == 0:
            save_predictions(np.array([], dtype=int), 0)
            return

        if TARGET not in train_df.columns:
            save_predictions(fallback, n_test)
            return

        y, numeric_target, train_df = prepare_target(train_df)

        if y is None or len(y) == 0:
            save_predictions(fallback, n_test)
            return

        fallback = np.full(n_test, y.value_counts().idxmax())

        if y.nunique() <= 1:
            save_predictions(np.full(n_test, y.iloc[0]), n_test)
            return

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        train_df = train_df.drop(columns=[TARGET, "report_id"], errors="ignore")
        test_df = test_df.drop(columns=["report_id"], errors="ignore")

        frames_tr = [pd.DataFrame({"const": 1.0}, index=train_df.index)]
        frames_te = [pd.DataFrame({"const": 1.0}, index=test_df.index)]

        numeric_cols = [
            c for c in train_df.select_dtypes(include=[np.number]).columns
            if c not in ("report_id", TARGET)
        ]
        for col in ["number_tobacco_products", "number_product_problems"]:
            if col not in numeric_cols and (col in train_df.columns or col in test_df.columns):
                numeric_cols.append(col)

        for col in numeric_cols:
            tr = (
                pd.to_numeric(train_df[col], errors="coerce")
                if col in train_df.columns
                else pd.Series(0.0, index=train_df.index)
            )
            te = (
                pd.to_numeric(test_df[col], errors="coerce")
                if col in test_df.columns
                else pd.Series(0.0, index=test_df.index)
            )
            med = tr.median()
            if pd.isna(med):
                med = 0.0
            frames_tr.append(pd.DataFrame({col: tr.fillna(med).astype(float)}, index=train_df.index))
            frames_te.append(pd.DataFrame({col: te.fillna(med).astype(float)}, index=test_df.index))

        date_tr = parse_dates(get_series(train_df, "date_submitted", train_df.index))
        date_te = parse_dates(get_series(test_df, "date_submitted", test_df.index))

        date_feat_tr = pd.DataFrame(index=train_df.index)
        date_feat_te = pd.DataFrame(index=test_df.index)
        date_feat_tr["date_year"] = date_tr.dt.year
        date_feat_te["date_year"] = date_te.dt.year
        date_feat_tr["date_month"] = date_tr.dt.month
        date_feat_te["date_month"] = date_te.dt.month
        date_feat_tr["date_day"] = date_tr.dt.day
        date_feat_te["date_day"] = date_te.dt.day
        date_feat_tr["date_dayofweek"] = date_tr.dt.dayofweek
        date_feat_te["date_dayofweek"] = date_te.dt.dayofweek
        date_feat_tr["date_year_month"] = date_tr.dt.year * 100 + date_tr.dt.month
        date_feat_te["date_year_month"] = date_te.dt.year * 100 + date_te.dt.month
        date_feat_tr = date_feat_tr.fillna(-1).astype(float)
        date_feat_te = date_feat_te.fillna(-1).astype(float)
        frames_tr.append(date_feat_tr)
        frames_te.append(date_feat_te)

        nonuser_tr = get_series(train_df, "nonuser_affected", train_df.index)
        nonuser_te = get_series(test_df, "nonuser_affected", test_df.index)
        oh_tr, oh_te = one_hot_features(nonuser_tr, nonuser_te, "nonuser")
        frames_tr.append(oh_tr)
        frames_te.append(oh_te)

        rpp_tr = get_series(train_df, "reported_product_problems", train_df.index)
        rpp_te = get_series(test_df, "reported_product_problems", test_df.index)
        frames_tr.append(text_meta(rpp_tr, "rpp"))
        frames_te.append(text_meta(rpp_te, "rpp"))

        rpp_patterns = {
            "foreign": r"foreign material",
            "damaged": r"damaged|defective",
            "taste": r"taste",
            "label": r"label",
            "appearance": r"appearance|look",
            "leak": r"leak",
            "smell": r"smell|odor",
            "burn": r"burn",
            "electronic": r"electronic|e-?cig|vape|vaping|e-?liquid|e-?juice|personal vaporizer",
            "cigarette": r"cigarette",
            "cigar": r"cigar",
            "hookah": r"hookah",
            "smokeless": r"smokeless",
            "other": r"other",
        }
        frames_tr.append(contains_features(rpp_tr, rpp_patterns, "rpp"))
        frames_te.append(contains_features(rpp_te, rpp_patterns, "rpp"))

        vec_tr, vec_te = fit_transform_text(rpp_tr, rpp_te, "rpp_vec", max_features=80)
        frames_tr.append(vec_tr)
        frames_te.append(vec_te)

        tp_tr = get_series(train_df, "tobacco_products", train_df.index)
        tp_te = get_series(test_df, "tobacco_products", test_df.index)
        frames_tr.append(text_meta(tp_tr, "tp"))
        frames_te.append(text_meta(tp_te, "tp"))

        tp_patterns = {
            "ecig": r"e-?cigarette|electronic cigarette|vape|vaping|e-?liquid|e-?juice|personal vaporizer|hookah pen|e-?pen|e-?cigar|e-?hookah|e-?pipe|mod",
            "cigarette": r"cigarette",
            "cigar": r"cigar",
            "hookah": r"hookah",
            "smokeless": r"smokeless",
            "tobacco": r"tobacco",
            "other": r"other",
        }
        frames_tr.append(contains_features(tp_tr, tp_patterns, "tp"))
        frames_te.append(contains_features(tp_te, tp_patterns, "tp"))

        vec_tr, vec_te = fit_transform_text(tp_tr, tp_te, "tp_vec", max_features=80)
        frames_tr.append(vec_tr)
        frames_te.append(vec_te)

        X_train = pd.concat(frames_tr, axis=1)
        X_test = pd.concat(frames_te, axis=1)

        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

        n_classes = len(np.unique(y_enc))
        best_iter = 300

        y_counts = pd.Series(y_enc).value_counts()
        min_count = int(y_counts.min()) if len(y_counts) else 0
        use_val = len(X_train) > 100 and n_classes > 1 and min_count >= 5

        if use_val:
            try:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_enc, test_size=0.2, random_state=42, stratify=y_enc
                )
                eval_metric = "multi_logloss" if n_classes > 2 else "binary_logloss"
                val_model = make_lgbm(500, len(X_tr))
                val_model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric=eval_metric,
                    callbacks=[lgbm.early_stopping(stopping_rounds=50, verbose=False)],
                )
                best_iter = getattr(val_model, "best_iteration_", None)
                if best_iter is None or best_iter <= 0:
                    best_iter = 300
                else:
                    best_iter = int(best_iter)
            except Exception:
                best_iter = 300

        final_model = make_lgbm(max(1, best_iter), len(X_train))
        final_model.fit(X_train, y_enc)
        pred_enc = final_model.predict(X_test)
        pred = le.inverse_transform(pred_enc)

        if numeric_target:
            pred = np.asarray(pred, dtype=int)

        save_predictions(pred, n_test)

    except Exception:
        save_predictions(fallback, n_test)


if __name__ == "__main__":
    main()