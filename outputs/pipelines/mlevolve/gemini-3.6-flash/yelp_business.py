import ast
from collections import Counter
import json
import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def safe_parse_dict(val):
    """Safely parse nested attribute/hours string into Python dictionary."""
    if pd.isna(val) or val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            val_clean = (
                val.replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
                .replace("None", "null")
            )
            res = json.loads(val_clean)
            if isinstance(res, dict):
                return res
        except Exception:
            pass
        try:
            res = ast.literal_eval(val)
            if isinstance(res, dict):
                return res
        except Exception:
            pass
    return {}


def extract_dynamic_attributes(train_attr_series, test_attr_series):
    """Dynamically parse all nested JSON attribute key-value pairs into features."""
    train_parsed = [safe_parse_dict(v) for v in train_attr_series]
    test_parsed = [safe_parse_dict(v) for v in test_attr_series]

    def flatten_attr(d):
        flat = {}
        if not isinstance(d, dict):
            return flat
        for k, v in d.items():
            if isinstance(v, str) and (v.startswith("{") or v.startswith("u'{") or v.startswith('b"{')):
                sub_d = safe_parse_dict(v)
                if isinstance(sub_d, dict) and len(sub_d) > 0:
                    for sub_k, sub_v in sub_d.items():
                        flat[f"{k}_{sub_k}"] = str(sub_v)
                    continue
            flat[k] = str(v)
        return flat

    train_flat = [flatten_attr(d) for d in train_parsed]
    test_flat = [flatten_attr(d) for d in test_parsed]

    all_keys = Counter()
    for d in train_flat:
        for k in d.keys():
            all_keys[k] += 1

    valid_keys = [k for k, count in all_keys.items() if count >= 10]

    train_attr_df = pd.DataFrame(index=train_attr_series.index)
    test_attr_df = pd.DataFrame(index=test_attr_series.index)

    for k in valid_keys:
        s_tr = pd.Series([d.get(k, None) for d in train_flat], index=train_attr_series.index)
        s_te = pd.Series([d.get(k, None) for d in test_flat], index=test_attr_series.index)

        s_tr_clean = s_tr.astype(str).str.lower().str.strip(" '\"")
        s_te_clean = s_te.astype(str).str.lower().str.strip(" '\"")

        unique_vals = set(s_tr_clean.unique()) | set(s_te_clean.unique())

        if unique_vals.issubset({'true', 'false', 'none', 'nan'}):
            train_attr_df[f"attr_{k}"] = (s_tr_clean == 'true').astype(int)
            test_attr_df[f"attr_{k}"] = (s_te_clean == 'true').astype(int)
        else:
            le = LabelEncoder()
            combined = pd.concat([s_tr_clean, s_te_clean])
            le.fit(combined)
            train_attr_df[f"attr_{k}"] = le.transform(s_tr_clean)
            test_attr_df[f"attr_{k}"] = le.transform(s_te_clean)

    train_attr_df["attr_count"] = [len(d) for d in train_flat]
    test_attr_df["attr_count"] = [len(d) for d in test_flat]

    return train_attr_df, test_attr_df


def parse_hours_to_df(series):
    """Extract operational hours indicators."""
    extracted = []
    for val in series:
        d = safe_parse_dict(val)
        days_open = len(d)
        extracted.append({"days_open": days_open, "is_open_7days": int(days_open == 7)})
    return pd.DataFrame(extracted)


def main():
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    print("Loading raw parquet datasets...")
    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    target_col = "stars"
    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    else:
        test_ids = np.arange(len(test_df))

    # Target Encoding
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(train_df[target_col])
    label_to_star = {i: cls for i, cls in enumerate(target_encoder.classes_)}
    num_classes = len(label_to_star)

    # Initialize Feature DataFrames
    X_train_feats = pd.DataFrame(index=train_df.index)
    X_test_feats = pd.DataFrame(index=test_df.index)

    # 1. Numerical & Binary Features
    if "review_count" in train_df.columns:
        X_train_feats["review_count"] = train_df["review_count"].fillna(0).values
        X_test_feats["review_count"] = test_df["review_count"].fillna(0).values
        X_train_feats["log_review_count"] = np.log1p(X_train_feats["review_count"])
        X_test_feats["log_review_count"] = np.log1p(X_test_feats["review_count"])

    if "is_open" in train_df.columns:
        X_train_feats["is_open"] = train_df["is_open"].fillna(0).astype(int).values
        X_test_feats["is_open"] = test_df["is_open"].fillna(0).astype(int).values

    # 2. Spatial Features
    if "latitude" in train_df.columns and "longitude" in train_df.columns:
        for df_src, df_dst in [(train_df, X_train_feats), (test_df, X_test_feats)]:
            lat = df_src["latitude"].fillna(0).values
            lon = df_src["longitude"].fillna(0).values
            df_dst["latitude"] = lat
            df_dst["longitude"] = lon
            df_dst["lat_lon_sum"] = lat + lon
            df_dst["lat_lon_diff"] = lat - lon
            df_dst["lat_round1"] = np.round(lat, 1)
            df_dst["lon_round1"] = np.round(lon, 1)

        mean_lat = train_df["latitude"].mean()
        mean_lon = train_df["longitude"].mean()
        X_train_feats["dist_to_center"] = np.sqrt(
            (train_df["latitude"] - mean_lat) ** 2
            + (train_df["longitude"] - mean_lon) ** 2
        ).fillna(0)
        X_test_feats["dist_to_center"] = np.sqrt(
            (test_df["latitude"] - mean_lat) ** 2
            + (test_df["longitude"] - mean_lon) ** 2
        ).fillna(0)

    # 3. Attributes Parsing
    if "attributes" in train_df.columns:
        train_attr_df, test_attr_df = extract_dynamic_attributes(train_df["attributes"], test_df["attributes"])
        for col in train_attr_df.columns:
            X_train_feats[col] = train_attr_df[col].values
            X_test_feats[col] = test_attr_df[col].values

    # 4. Hours Parsing
    if "hours" in train_df.columns:
        train_hrs_df = parse_hours_to_df(train_df["hours"])
        test_hrs_df = parse_hours_to_df(test_df["hours"])
        for col in train_hrs_df.columns:
            X_train_feats[f"hrs_{col}"] = train_hrs_df[col].values
            X_test_feats[f"hrs_{col}"] = test_hrs_df[col].values

    # 5. Text & Category Features
    if "categories" in train_df.columns:
        train_cats = train_df["categories"].fillna("").astype(str)
        test_cats = test_df["categories"].fillna("").astype(str)

        X_train_feats["cat_count"] = train_cats.apply(
            lambda x: len(x.split(",")) if x else 0
        )
        X_test_feats["cat_count"] = test_cats.apply(
            lambda x: len(x.split(",")) if x else 0
        )

        X_train_feats["cat_str_len"] = train_cats.apply(len)
        X_test_feats["cat_str_len"] = test_cats.apply(len)

        # Direct multi-hot binary indicator columns for top business categories
        cat_counter = Counter()
        for cat_str in train_cats:
            if cat_str:
                for c in cat_str.split(","):
                    c_clean = c.strip()
                    if c_clean:
                        cat_counter[c_clean] += 1

        top_categories = [c for c, count in cat_counter.most_common(120)]
        for cat_item in top_categories:
            clean_cat = re.sub(r"\W+", "_", cat_item)
            col_name = f"cat_has_{clean_cat}"
            X_train_feats[col_name] = train_cats.apply(
                lambda x: 1 if cat_item in [item.strip() for item in x.split(",")] else 0
            ).values
            X_test_feats[col_name] = test_cats.apply(
                lambda x: 1 if cat_item in [item.strip() for item in x.split(",")] else 0
            ).values

        tfidf_cat = TfidfVectorizer(
            max_features=250, stop_words="english", token_pattern=r"(?u)\b\w+\b"
        )
        cat_train_vec = tfidf_cat.fit_transform(train_cats)
        cat_test_vec = tfidf_cat.transform(test_cats)

        svd_cat = TruncatedSVD(n_components=15, random_state=42)
        cat_train_svd = svd_cat.fit_transform(cat_train_vec)
        cat_test_svd = svd_cat.transform(cat_test_vec)

        for i in range(15):
            X_train_feats[f"cat_svd_{i}"] = cat_train_svd[:, i]
            X_test_feats[f"cat_svd_{i}"] = cat_test_svd[:, i]

    if "name" in train_df.columns:
        train_names = train_df["name"].fillna("").astype(str)
        test_names = test_df["name"].fillna("").astype(str)

        X_train_feats["name_len"] = train_names.apply(len)
        X_test_feats["name_len"] = test_names.apply(len)
        X_train_feats["name_words"] = train_names.apply(lambda x: len(x.split()))
        X_test_feats["name_words"] = test_names.apply(lambda x: len(x.split()))

        tfidf_name = TfidfVectorizer(
            max_features=100, stop_words="english", token_pattern=r"(?u)\b\w+\b"
        )
        name_train_vec = tfidf_name.fit_transform(train_names)
        name_test_vec = tfidf_name.transform(test_names)

        svd_name = TruncatedSVD(n_components=8, random_state=42)
        name_train_svd = svd_name.fit_transform(name_train_vec)
        name_test_svd = svd_name.transform(name_test_vec)

        for i in range(8):
            X_train_feats[f"name_svd_{i}"] = name_train_svd[:, i]
            X_test_feats[f"name_svd_{i}"] = name_test_svd[:, i]

    # 6. Frequency Encodings & Group Statistics
    cat_cols = [
        c
        for c in ["city", "state", "postal_code"]
        if c in train_df.columns and c in test_df.columns
    ]
    for col in cat_cols:
        freq = train_df[col].value_counts().to_dict()
        X_train_feats[f"{col}_freq"] = train_df[col].map(freq).fillna(0).astype(float)
        X_test_feats[f"{col}_freq"] = test_df[col].map(freq).fillna(0).astype(float)

        if "review_count" in train_df.columns:
            group_stat = train_df.groupby(col)["review_count"].mean().to_dict()
            X_train_feats[f"{col}_mean_reviews"] = (
                train_df[col].map(group_stat).fillna(0).astype(float)
            )
            X_test_feats[f"{col}_mean_reviews"] = (
                test_df[col].map(group_stat).fillna(0).astype(float)
            )

    X_train_feats = X_train_feats.fillna(0)
    X_test_feats = X_test_feats.fillna(0)

    # 7. Model Training & Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_train_feats, y_train))

    n_samples_train = len(y_train)
    n_samples_test = len(X_test_feats)

    oof_probs = np.zeros((n_samples_train, num_classes), dtype=np.float64)
    test_probs = np.zeros((n_samples_test, num_classes), dtype=np.float64)

    raw_target = train_df[target_col].astype(float).values
    global_target_mean = float(np.mean(raw_target))

    lgb_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 45,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": 0.04,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    num_folds = len(folds)
    for fold, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_train_feats.iloc[train_idx].copy(), y_train[train_idx]
        X_va, y_va = X_train_feats.iloc[val_idx].copy(), y_train[val_idx]
        X_te = X_test_feats.copy()

        # Out-of-fold target encoding for spatial entities (city, state, postal_code)
        for c_col in cat_cols:
            tr_df_fold = train_df.iloc[train_idx]
            group_means = tr_df_fold[target_col].astype(float).groupby(tr_df_fold[c_col]).mean().to_dict()

            te_col_name = f"{c_col}_target_enc"
            X_tr[te_col_name] = train_df.iloc[train_idx][c_col].map(group_means).fillna(global_target_mean).values
            X_va[te_col_name] = train_df.iloc[val_idx][c_col].map(group_means).fillna(global_target_mean).values
            X_te[te_col_name] = test_df[c_col].map(group_means).fillna(global_target_mean).values

        # LightGBM Model
        lgb_model = lgb.LGBMClassifier(
            **lgb_params,
            n_estimators=1000,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        lgb_val_probs = lgb_model.predict_proba(X_va)
        lgb_test_probs = lgb_model.predict_proba(X_te)

        # XGBoost Model
        xgb_model = xgb.XGBClassifier(
            **xgb_params,
            n_estimators=1000,
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        xgb_val_probs = xgb_model.predict_proba(X_va)
        xgb_test_probs = xgb_model.predict_proba(X_te)

        # Ensemble prediction probabilities (average)
        val_pred_probs = 0.5 * lgb_val_probs + 0.5 * xgb_val_probs
        oof_probs[val_idx] = val_pred_probs

        test_pred_probs = 0.5 * lgb_test_probs + 0.5 * xgb_test_probs
        test_probs += test_pred_probs / num_folds

        fold_preds = np.argmax(val_pred_probs, axis=1)
        fold_acc = accuracy_score(y_va, fold_preds)
        print(f"Fold {fold + 1}/{num_folds} Accuracy: {fold_acc:.5f}")

    oof_preds = np.argmax(oof_probs, axis=1)
    overall_accuracy = accuracy_score(y_train, oof_preds)

    test_preds_class = np.argmax(test_probs, axis=1)
    test_preds_stars = [label_to_star[cls_idx] for cls_idx in test_preds_class]

    submission_df = pd.DataFrame({"id": test_ids, "stars": test_preds_stars})
    sub_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {overall_accuracy}")


if __name__ == "__main__":
    main()
