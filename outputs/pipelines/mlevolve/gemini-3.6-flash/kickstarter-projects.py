import json
import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def main():
    # ----------------------------------------------------
    # Step 1: Data Processing & Feature Engineering
    # ----------------------------------------------------
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Identify target column
    target_col = "status"
    if target_col in train_df.columns:
        if (
            train_df[target_col].dtype == object
            or train_df[target_col].dtype == "category"
        ):
            y_raw = train_df[target_col].astype(str).str.lower()
            train_df["target"] = y_raw.str.contains("success|1|true|successful").astype(
                int
            )
        else:
            train_df["target"] = (train_df[target_col] == 1).astype(int)
    else:
        raise ValueError(f"Target column '{target_col}' not found in train data.")

    # Preserve row identifier if present
    id_col = None
    for col in ["id", "project_id", "index"]:
        if col in train_df.columns and col in test_df.columns:
            id_col = col
            break

    if id_col is None:
        train_df["id"] = np.arange(len(train_df))
        test_df["id"] = np.arange(len(test_df))
        id_col = "id"

    # Detect column types
    datetime_cols = []
    text_cols = []
    num_cols = []
    cat_cols = []

    for col in train_df.columns:
        if col in ["target", target_col, id_col]:
            continue
        # Post-campaign outcome leaks that might exist in train but not test or shouldn't be used
        if col in [
            "pledged",
            "backers",
            "usd_pledged_real",
            "usd_pledged",
            "state_changed_at",
        ]:
            if (
                col not in test_df.columns
                or test_df[col].isnull().all()
                or (test_df[col] == 0).all()
            ):
                continue

        col_type = train_df[col].dtype
        col_lower = col.lower()

        if (
            "date" in col_lower
            or "time" in col_lower
            or "launched" in col_lower
            or "deadline" in col_lower
            or "created" in col_lower
        ):
            datetime_cols.append(col)
        elif col_lower in ["name", "blurb", "description", "title"]:
            text_cols.append(col)
        elif pd.api.types.is_numeric_dtype(col_type):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    # Perform Train / Validation Split FIRST to enforce strict isolation
    train_split, val_split = train_test_split(
        train_df, test_size=0.20, random_state=42, stratify=train_df["target"]
    )

    train_split = train_split.copy().reset_index(drop=True)
    val_split = val_split.copy().reset_index(drop=True)
    test_split = test_df.copy().reset_index(drop=True)

    # Extract temporal features
    def extract_datetime_features(df):
        df_feat = pd.DataFrame(index=df.index)
        parsed_dates = {}
        for col in datetime_cols:
            parsed = pd.to_datetime(
                df[col],
                errors="coerce",
                unit="s" if pd.api.types.is_numeric_dtype(df[col]) else None,
            )
            if parsed.isnull().sum() > 0.5 * len(df):
                parsed = pd.to_datetime(df[col], errors="coerce")
            parsed_dates[col] = parsed

            df_feat[f"{col}_year"] = parsed.dt.year.fillna(-1).astype(int)
            df_feat[f"{col}_month"] = parsed.dt.month.fillna(-1).astype(int)
            df_feat[f"{col}_day"] = parsed.dt.day.fillna(-1).astype(int)
            df_feat[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(-1).astype(int)
            df_feat[f"{col}_hour"] = parsed.dt.hour.fillna(-1).astype(int)

            # Cyclical encoding
            df_feat[f"{col}_month_sin"] = np.sin(
                2 * np.pi * parsed.dt.month / 12.0
            ).fillna(0)
            df_feat[f"{col}_month_cos"] = np.cos(
                2 * np.pi * parsed.dt.month / 12.0
            ).fillna(0)
            df_feat[f"{col}_dow_sin"] = np.sin(
                2 * np.pi * parsed.dt.dayofweek / 7.0
            ).fillna(0)
            df_feat[f"{col}_dow_cos"] = np.cos(
                2 * np.pi * parsed.dt.dayofweek / 7.0
            ).fillna(0)

        # Duration logic
        launched_col = next((c for c in datetime_cols if "launch" in c.lower()), None)
        deadline_col = next((c for c in datetime_cols if "dead" in c.lower()), None)
        created_col = next((c for c in datetime_cols if "create" in c.lower()), None)

        if (
            launched_col
            and deadline_col
            and launched_col in parsed_dates
            and deadline_col in parsed_dates
        ):
            duration = (
                parsed_dates[deadline_col] - parsed_dates[launched_col]
            ).dt.total_seconds() / 86400.0
            df_feat["campaign_duration_days"] = duration.fillna(
                duration.median() if len(duration) > 0 else 30.0
            )

        if (
            launched_col
            and created_col
            and launched_col in parsed_dates
            and created_col in parsed_dates
        ):
            prep_time = (
                parsed_dates[launched_col] - parsed_dates[created_col]
            ).dt.total_seconds() / 86400.0
            df_feat["prep_duration_days"] = prep_time.fillna(
                prep_time.median() if len(prep_time) > 0 else 0.0
            )

        return df_feat

    train_dt = extract_datetime_features(train_split)
    val_dt = extract_datetime_features(val_split)
    test_dt = extract_datetime_features(test_split)

    # Extract goal & numerical features
    goal_col = next(
        (c for c in num_cols if "goal" in c.lower()),
        num_cols[0] if len(num_cols) > 0 else None,
    )

    def extract_numerical_features(df, dt_feat):
        df_feat = pd.DataFrame(index=df.index)
        for col in num_cols:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df_feat[col] = vals
            df_feat[f"{col}_log"] = np.log1p(np.maximum(0, vals))

        if goal_col and goal_col in df_feat.columns:
            dur = dt_feat.get("campaign_duration_days", pd.Series(30.0, index=df.index))
            dur_clamped = np.maximum(1.0, dur)
            df_feat["goal_per_day"] = df_feat[f"{goal_col}_log"] / dur_clamped

        return df_feat

    train_num = extract_numerical_features(train_split, train_dt)
    val_num = extract_numerical_features(val_split, val_dt)
    test_num = extract_numerical_features(test_split, test_dt)

    # Category statistics & normalized goal ratios (computed ON TRAIN ONLY)
    main_cat_col = next(
        (c for c in cat_cols if "main" in c.lower() or "cat" in c.lower()),
        cat_cols[0] if len(cat_cols) > 0 else None,
    )

    if goal_col and main_cat_col:
        cat_stats = (
            train_split.groupby(main_cat_col)[goal_col]
            .agg(cat_goal_mean="mean", cat_goal_median="median", cat_goal_std="std")
            .reset_index()
        )

        global_mean = train_split[goal_col].mean()
        global_median = train_split[goal_col].median()

        for df_raw, df_num in zip(
            [train_split, val_split, test_split], [train_num, val_num, test_num]
        ):
            merged = pd.merge(
                df_raw[[main_cat_col]], cat_stats, on=main_cat_col, how="left"
            )
            mean_val = merged["cat_goal_mean"].fillna(global_mean).values
            median_val = merged["cat_goal_median"].fillna(global_median).values

            df_num["goal_to_cat_mean_ratio"] = df_num[goal_col] / (mean_val + 1e-5)
            df_num["goal_to_cat_median_ratio"] = df_num[goal_col] / (median_val + 1e-5)

    # Text features (length, punctuation, TF-IDF, Ridge OOF, TruncatedSVD 64)
    def extract_text_style(df):
        df_feat = pd.DataFrame(index=df.index)
        combined_text = pd.Series("", index=df.index)
        for col in text_cols:
            txt = df[col].astype(str).fillna("")
            combined_text = combined_text + " " + txt
            df_feat[f"{col}_len"] = txt.str.len()
            df_feat[f"{col}_words"] = txt.str.split().str.len()
            df_feat[f"{col}_upper_ratio"] = (
                txt.str.findall(r"[A-Z]").str.len() / (df_feat[f"{col}_len"] + 1)
            ).fillna(0)
            df_feat[f"{col}_excl_count"] = txt.str.count("!")
            df_feat[f"{col}_quest_count"] = txt.str.count(r"\?")
        df_feat["total_text_len"] = combined_text.str.len()
        df_feat["total_text_words"] = combined_text.str.split().str.len()
        return df_feat, combined_text.str.strip()

    train_txt_style, train_text_corpus = extract_text_style(train_split)
    val_txt_style, val_text_corpus = extract_text_style(val_split)
    test_txt_style, test_text_corpus = extract_text_style(test_split)

    if main_cat_col:
        train_split["_total_text_len"] = train_txt_style["total_text_len"]
        cat_text_stats = (
            train_split.groupby(main_cat_col)["_total_text_len"]
            .median()
            .reset_index(name="cat_text_len_median")
        )
        global_text_len_median = train_txt_style["total_text_len"].median()

        for df_raw, df_txt in zip(
            [train_split, val_split, test_split],
            [train_txt_style, val_txt_style, test_txt_style],
        ):
            merged = pd.merge(
                df_raw[[main_cat_col]], cat_text_stats, on=main_cat_col, how="left"
            )
            med_len = merged["cat_text_len_median"].fillna(global_text_len_median).values
            df_txt["text_len_to_cat_median_ratio"] = df_txt["total_text_len"] / (
                med_len + 1e-5
            )

        train_split.drop(columns=["_total_text_len"], inplace=True)

    train_meta_df = pd.DataFrame(index=train_split.index)
    val_meta_df = pd.DataFrame(index=val_split.index)
    test_meta_df = pd.DataFrame(index=test_split.index)

    if len(text_cols) > 0 and train_text_corpus.str.len().sum() > 0:
        word_tfidf = TfidfVectorizer(
            max_features=25000, stop_words="english", ngram_range=(1, 2), min_df=2
        )
        char_tfidf = TfidfVectorizer(
            max_features=25000, analyzer="char_wb", ngram_range=(3, 5), min_df=3
        )

        tr_word = word_tfidf.fit_transform(train_text_corpus)
        va_word = word_tfidf.transform(val_text_corpus)
        te_word = word_tfidf.transform(test_text_corpus)

        tr_char = char_tfidf.fit_transform(train_text_corpus)
        va_char = char_tfidf.transform(val_text_corpus)
        te_char = char_tfidf.transform(test_text_corpus)

        X_tr_tfidf = hstack([tr_word, tr_char]).tocsr()
        X_va_tfidf = hstack([va_word, va_char]).tocsr()
        X_te_tfidf = hstack([te_word, te_char]).tocsr()

        svd_n = 64
        svd = TruncatedSVD(n_components=svd_n, random_state=42)
        train_svd = svd.fit_transform(X_tr_tfidf)
        val_svd = svd.transform(X_va_tfidf)
        test_svd = svd.transform(X_te_tfidf)

        svd_cols = [f"tfidf_svd_{i}" for i in range(svd_n)]
        train_svd_df = pd.DataFrame(
            train_svd, columns=svd_cols, index=train_split.index
        )
        val_svd_df = pd.DataFrame(val_svd, columns=svd_cols, index=val_split.index)
        test_svd_df = pd.DataFrame(test_svd, columns=svd_cols, index=test_split.index)

        # OOF text predictions via Ridge & LogisticRegression
        skf_text = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_ridge = np.zeros(len(train_split))
        oof_logit = np.zeros(len(train_split))

        for tr_idx, val_idx in skf_text.split(train_split, train_split["target"]):
            ridge_fold = Ridge(alpha=1.0, random_state=42)
            ridge_fold.fit(X_tr_tfidf[tr_idx], train_split["target"].iloc[tr_idx])
            oof_ridge[val_idx] = ridge_fold.predict(X_tr_tfidf[val_idx])

            logit_fold = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            logit_fold.fit(X_tr_tfidf[tr_idx], train_split["target"].iloc[tr_idx])
            oof_logit[val_idx] = logit_fold.predict_proba(X_tr_tfidf[val_idx])[:, 1]

        ridge_full = Ridge(alpha=1.0, random_state=42)
        ridge_full.fit(X_tr_tfidf, train_split["target"])
        val_ridge = ridge_full.predict(X_va_tfidf)
        test_ridge = ridge_full.predict(X_te_tfidf)

        logit_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        logit_full.fit(X_tr_tfidf, train_split["target"])
        val_logit = logit_full.predict_proba(X_va_tfidf)[:, 1]
        test_logit = logit_full.predict_proba(X_te_tfidf)[:, 1]

        train_meta_df["text_oof_ridge"] = oof_ridge
        train_meta_df["text_oof_logit"] = oof_logit
        val_meta_df["text_oof_ridge"] = val_ridge
        val_meta_df["text_oof_logit"] = val_logit
        test_meta_df["text_oof_ridge"] = test_ridge
        test_meta_df["text_oof_logit"] = test_logit
    else:
        train_svd_df = pd.DataFrame(index=train_split.index)
        val_svd_df = pd.DataFrame(index=val_split.index)
        test_svd_df = pd.DataFrame(index=test_split.index)

    # Categorical Encodings (Frequency & Label Encoding fitted ON TRAIN ONLY)
    train_cat_enc = pd.DataFrame(index=train_split.index)
    val_cat_enc = pd.DataFrame(index=val_split.index)
    test_cat_enc = pd.DataFrame(index=test_split.index)

    for col in cat_cols:
        freq_map = train_split[col].value_counts(normalize=True).to_dict()
        train_cat_enc[f"{col}_freq"] = (
            train_split[col].map(freq_map).fillna(0).astype(float)
        )
        val_cat_enc[f"{col}_freq"] = (
            val_split[col].map(freq_map).fillna(0).astype(float)
        )
        test_cat_enc[f"{col}_freq"] = (
            test_split[col].map(freq_map).fillna(0).astype(float)
        )

        le = LabelEncoder()
        train_str = train_split[col].astype(str).fillna("missing")
        val_str = val_split[col].astype(str).fillna("missing")
        test_str = test_split[col].astype(str).fillna("missing")

        le.fit(pd.concat([train_str, pd.Series(["unseen_label"])]))

        val_str_mapped = val_str.map(
            lambda s: s if s in le.classes_ else "unseen_label"
        )
        test_str_mapped = test_str.map(
            lambda s: s if s in le.classes_ else "unseen_label"
        )

        train_cat_enc[f"{col}_le"] = le.transform(train_str)
        val_cat_enc[f"{col}_le"] = le.transform(val_str_mapped)
        test_cat_enc[f"{col}_le"] = le.transform(test_str_mapped)

    # Out-of-fold Target Encoding for categoricals (TRAIN ONLY)
    if len(cat_cols) > 0:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        global_target_mean = train_split["target"].mean()

        for col in cat_cols:
            train_cat_enc[f"{col}_te"] = np.nan
            for tr_idx, val_idx in skf.split(train_split, train_split["target"]):
                fold_tr = train_split.iloc[tr_idx]
                te_map = fold_tr.groupby(col)["target"].mean().to_dict()
                train_cat_enc.iloc[
                    val_idx, train_cat_enc.columns.get_loc(f"{col}_te")
                ] = train_split.iloc[val_idx][col].map(te_map)

            train_cat_enc[f"{col}_te"] = train_cat_enc[f"{col}_te"].fillna(
                global_target_mean
            )

            full_te_map = train_split.groupby(col)["target"].mean().to_dict()
            val_cat_enc[f"{col}_te"] = (
                val_split[col].map(full_te_map).fillna(global_target_mean)
            )
            test_cat_enc[f"{col}_te"] = (
                test_split[col].map(full_te_map).fillna(global_target_mean)
            )

    # Concatenate feature groups
    X_train = pd.concat(
        [train_dt, train_num, train_txt_style, train_svd_df, train_meta_df, train_cat_enc],
        axis=1,
    )
    X_val = pd.concat(
        [val_dt, val_num, val_txt_style, val_svd_df, train_meta_df if False else val_meta_df, val_cat_enc],
        axis=1,
    )
    X_test = pd.concat(
        [test_dt, test_num, test_txt_style, test_svd_df, test_meta_df, test_cat_enc],
        axis=1,
    )

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Fill missing values with column medians computed on train
    for col in X_train.columns:
        if X_train[col].isnull().any():
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_val[col] = X_val[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)

    y_train = train_split["target"].values
    y_val = val_split["target"].values

    # ----------------------------------------------------
    # Step 2: Model Configuration
    # ----------------------------------------------------
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    cb_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1,
    }

    # ----------------------------------------------------
    # Step 3: Model Training, Evaluation & Ensembling
    # ----------------------------------------------------
    # 1. LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    val_preds_lgb = lgb_model.predict_proba(X_val)[:, 1]
    test_preds_lgb = lgb_model.predict_proba(X_test)[:, 1]

    # 2. XGBoost
    xgb_fit_params = xgb_params.copy()
    xgb_fit_params["early_stopping_rounds"] = 50
    xgb_model = xgb.XGBClassifier(**xgb_fit_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_preds_xgb = xgb_model.predict_proba(X_val)[:, 1]
    test_preds_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # 3. CatBoost
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    val_preds_cb = cb_model.predict_proba(X_val)[:, 1]
    test_preds_cb = cb_model.predict_proba(X_test)[:, 1]

    # Search optimal blending weights on validation predictions
    best_auc = 0.0
    best_weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)

    for w1 in np.linspace(0, 1, 21):
        for w2 in np.linspace(0, 1 - w1, 21):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-5:
                continue
            w3 = max(0.0, w3)
            p_val = w1 * val_preds_lgb + w2 * val_preds_xgb + w3 * val_preds_cb
            score = roc_auc_score(y_val, p_val)
            if score > best_auc:
                best_auc = score
                best_weights = (w1, w2, w3)

    w1, w2, w3 = best_weights
    val_preds_ensemble = w1 * val_preds_lgb + w2 * val_preds_xgb + w3 * val_preds_cb
    test_preds_ensemble = w1 * test_preds_lgb + w2 * test_preds_xgb + w3 * test_preds_cb

    final_val_auc = roc_auc_score(y_val, val_preds_ensemble)

    # Save submission file
    os.makedirs("./submission", exist_ok=True)
    sub_df = pd.DataFrame(
        {"id": test_split[id_col].values, "status": test_preds_ensemble}
    )
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_auc}")


if __name__ == "__main__":
    main()
