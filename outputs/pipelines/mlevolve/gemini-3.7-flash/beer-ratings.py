import itertools
import os
import warnings
from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb

warnings.filterwarnings("ignore")


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Executes dynamic feature engineering, text decomposition, group aggregations,
    and leak-free out-of-fold target encoding, guaranteeing valid dtypes for LightGBM.
    """
    target_col = "review_overall"
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Assign fold split on training set prior to feature extraction
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train_df)):
        train_df.iloc[val_idx, train_df.columns.get_loc("fold")] = fold

    train_df["_dataset_split"] = "train"
    test_df["_dataset_split"] = "test"
    test_df["fold"] = -1
    if target_col not in test_df.columns:
        test_df[target_col] = np.nan

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # Identify potential text and categorical columns
    text_cols = [c for c in ["Description", "review_text", "Notes"] if c in combined_df.columns]

    # Process text columns with TF-IDF + TruncatedSVD
    for col in text_cols:
        text_series = combined_df[col].fillna("").astype(str)
        combined_df[f"{col}_len"] = text_series.str.len()
        combined_df[f"{col}_word_count"] = text_series.apply(lambda x: len(x.split()))
        combined_df[f"{col}_caps_ratio"] = text_series.apply(
            lambda x: sum(1 for char in x if char.isupper()) / (len(x) + 1.0)
        )

        train_mask = combined_df["_dataset_split"] == "train"
        tfidf = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
        svd = TruncatedSVD(n_components=12, random_state=42)

        tfidf_train = tfidf.fit_transform(text_series[train_mask])
        svd.fit(tfidf_train)
        tfidf_all = tfidf.transform(text_series)
        svd_all = svd.transform(tfidf_all)

        for i in range(12):
            combined_df[f"{col}_svd_{i}"] = svd_all[:, i]

    # Domain Aspect Rating Interactions
    aspect_cols = [c for c in ["review_taste", "review_aroma", "review_palate", "review_appearance"] if c in combined_df.columns]
    if aspect_cols:
        aspect_mat = combined_df[aspect_cols].values
        combined_df["aspect_mean"] = np.nanmean(aspect_mat, axis=1)
        combined_df["aspect_std"] = np.nanstd(aspect_mat, axis=1)
        combined_df["aspect_min"] = np.nanmin(aspect_mat, axis=1)
        combined_df["aspect_max"] = np.nanmax(aspect_mat, axis=1)
        combined_df["aspect_range"] = combined_df["aspect_max"] - combined_df["aspect_min"]

        if all(c in combined_df.columns for c in ["review_taste", "review_aroma", "review_palate", "review_appearance"]):
            combined_df["aspect_composite"] = (
                0.4 * combined_df["review_taste"]
                + 0.2 * combined_df["review_aroma"]
                + 0.2 * combined_df["review_palate"]
                + 0.2 * combined_df["review_appearance"]
            )
        else:
            combined_df["aspect_composite"] = combined_df["aspect_mean"]

        for c1, c2 in itertools.combinations(aspect_cols, 2):
            combined_df[f"{c1}_minus_{c2}"] = combined_df[c1] - combined_df[c2]
            combined_df[f"{c1}_ratio_{c2}"] = (combined_df[c1] + 1e-4) / (combined_df[c2] + 1e-4)
            combined_df[f"{c1}_mult_{c2}"] = combined_df[c1] * combined_df[c2]

    # IBU Domain Features
    if "Min IBU" in combined_df.columns and "Max IBU" in combined_df.columns:
        combined_df["ibu_range"] = combined_df["Max IBU"] - combined_df["Min IBU"]
        combined_df["ibu_ratio"] = combined_df["Max IBU"] / (combined_df["Min IBU"] + 1.0)
        combined_df["ibu_mean"] = (combined_df["Max IBU"] + combined_df["Min IBU"]) / 2.0

    # Flavor / Sensory Profile Features
    flavor_candidates = [
        "Astringency", "Body", "Alcohol", "Bitter", "Sweet", "Sour", "Salty",
        "Fruits", "Hoppy", "Spices", "Malty"
    ]
    flavor_cols = [c for c in flavor_candidates if c in combined_df.columns]
    if flavor_cols:
        flavor_mat = combined_df[flavor_cols].values
        combined_df["flavor_sum"] = np.nanmean(flavor_mat, axis=1) * len(flavor_cols)
        combined_df["flavor_mean"] = np.nanmean(flavor_mat, axis=1)
        combined_df["flavor_std"] = np.nanstd(flavor_mat, axis=1)
        combined_df["flavor_max"] = np.nanmax(flavor_mat, axis=1)
        combined_df["flavor_min"] = np.nanmin(flavor_mat, axis=1)
        combined_df["flavor_range"] = combined_df["flavor_max"] - combined_df["flavor_min"]

        if "Sweet" in combined_df.columns and "Bitter" in combined_df.columns:
            combined_df["sweet_to_bitter"] = combined_df["Sweet"] / (combined_df["Bitter"] + 1e-4)
        if "Fruits" in combined_df.columns and "Hoppy" in combined_df.columns:
            combined_df["fruits_to_hoppy"] = combined_df["Fruits"] / (combined_df["Hoppy"] + 1e-4)
        if "Malty" in combined_df.columns and "Bitter" in combined_df.columns:
            combined_df["malty_to_bitter"] = combined_df["Malty"] / (combined_df["Bitter"] + 1e-4)

    # Split train and test back to calculate aggregations strictly on training sets
    train_processed = combined_df[combined_df["_dataset_split"] == "train"].copy()
    test_processed = combined_df[combined_df["_dataset_split"] == "test"].copy()

    # Categorical Columns Identification
    all_obj_cols = [
        c for c in train_processed.columns
        if (train_processed[c].dtype == "object" or train_processed[c].dtype.name == "category")
        and c not in ["_dataset_split"] + text_cols
    ]

    # Frequency Encoding
    for col in all_obj_cols:
        freq_map = train_processed[col].value_counts(normalize=True).to_dict()
        count_map = train_processed[col].value_counts().to_dict()
        train_processed[f"{col}_freq"] = train_processed[col].map(freq_map).fillna(0)
        test_processed[f"{col}_freq"] = test_processed[col].map(freq_map).fillna(0)
        train_processed[f"{col}_count_log"] = np.log1p(train_processed[col].map(count_map).fillna(0))
        test_processed[f"{col}_count_log"] = np.log1p(test_processed[col].map(count_map).fillna(0))

    # Multi-Entity Out-of-Fold Aggregations and Relative Bias Features
    entity_candidates = ["review_profilename", "beer_beerid", "beer_name", "brewery_name", "brewery_id", "beer_style", "Style", "Brewery"]
    entities = [c for c in entity_candidates if c in train_processed.columns]
    agg_features = [target_col] + [
        c for c in ["review_taste", "review_aroma", "review_palate", "review_appearance", "aspect_composite", "ABV", "beer_abv", "flavor_mean"]
        if c in train_processed.columns
    ]

    for ent in entities:
        for feat in agg_features:
            global_feat_mean = train_processed[feat].mean()
            m_weight = 10.0

            train_processed[f"oof_{ent}_{feat}_mean"] = np.nan
            train_processed[f"oof_{ent}_{feat}_std"] = np.nan

            for fold in range(5):
                trn_mask = train_processed["fold"] != fold
                val_mask = train_processed["fold"] == fold

                trn_sub = train_processed[trn_mask]
                stats = trn_sub.groupby(ent)[feat].agg(["count", "mean", "std"])
                smoothed_mean = (stats["count"] * stats["mean"] + m_weight * global_feat_mean) / (stats["count"] + m_weight)

                mean_dict = smoothed_mean.to_dict()
                std_dict = stats["std"].fillna(0).to_dict()

                train_processed.loc[val_mask, f"oof_{ent}_{feat}_mean"] = (
                    train_processed.loc[val_mask, ent].map(mean_dict).fillna(global_feat_mean)
                )
                train_processed.loc[val_mask, f"oof_{ent}_{feat}_std"] = (
                    train_processed.loc[val_mask, ent].map(std_dict).fillna(0)
                )

            full_stats = train_processed.groupby(ent)[feat].agg(["count", "mean", "std"])
            full_smoothed = (full_stats["count"] * full_stats["mean"] + m_weight * global_feat_mean) / (full_stats["count"] + m_weight)
            test_processed[f"oof_{ent}_{feat}_mean"] = (
                test_processed[ent].map(full_smoothed.to_dict()).fillna(global_feat_mean)
            )
            test_processed[f"oof_{ent}_{feat}_std"] = (
                test_processed[ent].map(full_stats["std"].fillna(0).to_dict()).fillna(0)
            )

            if feat != target_col:
                train_processed[f"{feat}_diff_{ent}_mean"] = train_processed[feat] - train_processed[f"oof_{ent}_{feat}_mean"]
                test_processed[f"{feat}_diff_{ent}_mean"] = test_processed[feat] - test_processed[f"oof_{ent}_{feat}_mean"]

    # Explicitly unify categorical columns
    for col in all_obj_cols:
        train_processed[col] = train_processed[col].astype(str)
        test_processed[col] = test_processed[col].astype(str)
        categories = sorted(list(set(train_processed[col].unique()).union(set(test_processed[col].unique()))))
        train_processed[col] = pd.Categorical(train_processed[col], categories=categories)
        test_processed[col] = pd.Categorical(test_processed[col], categories=categories)

    # Drop meta/raw text columns
    drop_meta_cols = ["_dataset_split", "review_time"] + text_cols
    train_processed.drop(
        columns=[c for c in drop_meta_cols if c in train_processed.columns],
        inplace=True,
    )
    test_processed.drop(
        columns=[c for c in drop_meta_cols if c in test_processed.columns],
        inplace=True,
    )

    feature_cols = [
        c for c in train_processed.columns if c not in [target_col, "fold", "id"]
    ]
    categorical_features = [c for c in all_obj_cols if c in feature_cols]

    return train_processed, test_processed, feature_cols, categorical_features, target_col


def main():
    input_dir = "./input"
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # Load raw data
    train_raw = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    test_raw = pd.read_parquet(os.path.join(input_dir, "test.parquet"))

    # Feature Engineering
    train_df, test_df, feature_cols, cat_cols, target_col = build_features(
        train_raw, test_raw
    )

    n_folds = 5
    target_min = train_df[target_col].min()
    target_max = train_df[target_col].max()
    active_cat_cols = [c for c in cat_cols if c in feature_cols]

    oof_lgb = np.zeros(len(train_df))
    test_lgb = np.zeros(len(test_df))

    oof_cb = np.zeros(len(train_df))
    test_cb = np.zeros(len(test_df))

    oof_xgb = np.zeros(len(train_df))
    test_xgb = np.zeros(len(test_df))

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # 5-Fold Training Loop for Diverse Models
    for fold in range(n_folds):
        trn_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        X_train, y_train = train_df.loc[trn_idx, feature_cols], train_df.loc[trn_idx, target_col]
        X_val, y_val = train_df.loc[val_idx, feature_cols], train_df.loc[val_idx, target_col]
        X_test = test_df[feature_cols]

        # 1. LightGBM
        trn_data = lgb.Dataset(X_train, label=y_train, categorical_feature=active_cat_cols, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data, categorical_feature=active_cat_cols, free_raw_data=False)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
        model_lgb = lgb.train(params=lgb_params, train_set=trn_data, num_boost_round=1500, valid_sets=[trn_data, val_data], callbacks=callbacks)

        oof_lgb[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
        test_lgb += model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_folds

        # 2. CatBoost Regressor
        model_cb = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42 + fold,
            verbose=False,
            thread_count=-1,
        )
        model_cb.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            cat_features=active_cat_cols,
            early_stopping_rounds=50,
            verbose=False,
        )
        oof_cb[val_idx] = model_cb.predict(X_val)
        test_cb += model_cb.predict(X_test) / n_folds

        # 3. XGBoost Regressor
        model_xgb = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=50,
        )
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = model_xgb.predict(X_val)
        test_xgb += model_xgb.predict(X_test) / n_folds

    # Ensemble Optimization (SLSQP blend)
    oof_matrix = np.column_stack([oof_lgb, oof_cb, oof_xgb])
    test_matrix = np.column_stack([test_lgb, test_cb, test_xgb])
    y_true = train_df[target_col].values

    def blend_objective(weights):
        w = np.array(weights)
        w = w / (np.sum(w) + 1e-12)
        pred = np.sum(oof_matrix * w, axis=1)
        return root_mean_squared_error(y_true, pred)

    opt_res = minimize(
        blend_objective,
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        constraints={"type": "eq", "fun": lambda w: 1.0 - np.sum(w)},
        method="SLSQP",
    )
    best_weights = opt_res.x / np.sum(opt_res.x)

    oof_blended = np.sum(oof_matrix * best_weights, axis=1)
    oof_blended = np.clip(oof_blended, target_min, target_max)
    test_blended = np.sum(test_matrix * best_weights, axis=1)
    test_blended = np.clip(test_blended, target_min, target_max)

    final_rmse = root_mean_squared_error(y_true, oof_blended)

    # Save final submission
    if "id" in test_df.columns:
        sub_id = test_df["id"].values
    elif "id" in test_raw.columns:
        sub_id = test_raw["id"].values
    else:
        sub_id = np.arange(len(test_df))

    submission_df = pd.DataFrame({"id": sub_id, "review_overall": test_blended})
    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_rmse}")


if __name__ == "__main__":
    main()
