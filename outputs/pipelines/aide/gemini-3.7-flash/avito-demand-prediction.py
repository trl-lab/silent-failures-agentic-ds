import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def main():
    print("Loading datasets...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    n_train = len(train_df)
    n_test = len(test_df)
    print(f"Train samples: {n_train}, Test samples: {n_test}")

    target = train_df["deal_probability"].values.astype(np.float32)

    # Combine data for unified feature engineering
    df = pd.concat(
        [train_df.drop(columns=["deal_probability"], errors="ignore"), test_df],
        ignore_index=True,
    )
    del train_df, test_df
    gc.collect()

    print("Engineering tabular and temporal features...")
    # Date features
    df["activation_date"] = pd.to_datetime(df["activation_date"])
    df["weekday"] = df["activation_date"].dt.weekday.astype(np.int8)
    df["day"] = df["activation_date"].dt.day.astype(np.int8)

    # Price features
    df["price"] = df["price"].astype(np.float32)
    df["price_is_na"] = df["price"].isna().astype(np.int8)
    df["log_price"] = np.log1p(df["price"].fillna(0)).astype(np.float32)

    # Category price aggregation
    price_cat_mean = (
        df.groupby("category_name")["log_price"].transform("mean").astype(np.float32)
    )
    price_cat_std = (
        df.groupby("category_name")["log_price"]
        .transform("std")
        .fillna(0)
        .astype(np.float32)
    )
    df["price_to_cat_diff"] = df["log_price"] - price_cat_mean
    df["price_to_cat_zscore"] = (
        df["price_to_cat_diff"] / (price_cat_std + 1e-5)
    ).astype(np.float32)

    # Image features
    df["has_image"] = df["image"].notna().astype(np.int8)
    df["image_top_1"] = df["image_top_1"].fillna(-1).astype(np.float32)

    # Text length features
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    df["title_len"] = df["title"].str.len().astype(np.int32)
    df["title_words"] = df["title"].str.split().str.len().astype(np.int16)
    df["desc_len"] = df["description"].str.len().astype(np.int32)
    df["desc_words"] = df["description"].str.split().str.len().astype(np.int16)

    # User activity features
    df["user_item_count"] = (
        df.groupby("user_id")["item_id"].transform("count").astype(np.int32)
    )

    # Label encode categorical columns
    cat_cols = [
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "param_1",
        "param_2",
        "param_3",
        "user_type",
    ]
    for col in cat_cols:
        df[col] = df[col].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col]).astype(np.int32)

    print("Extracting TF-IDF and out-of-fold Ridge features...")
    # Text modeling with Ridge regression
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Title TF-IDF + Ridge
    tfidf_title = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
    title_tfidf_train = tfidf_title.fit_transform(df["title"].iloc[:n_train])
    title_tfidf_test = tfidf_title.transform(df["title"].iloc[n_train:])

    oof_ridge_title = np.zeros(n_train, dtype=np.float32)
    test_ridge_title = np.zeros(n_test, dtype=np.float32)

    for train_idx, val_idx in kf.split(title_tfidf_train):
        ridge = Ridge(alpha=10.0, solver="sag", random_state=42)
        ridge.fit(title_tfidf_train[train_idx], target[train_idx])
        oof_ridge_title[val_idx] = ridge.predict(title_tfidf_train[val_idx])
        test_ridge_title += ridge.predict(title_tfidf_test) / kf.n_splits

    df["ridge_pred_title"] = np.concatenate([oof_ridge_title, test_ridge_title]).astype(
        np.float32
    )

    del (
        tfidf_title,
        title_tfidf_train,
        title_tfidf_test,
        oof_ridge_title,
        test_ridge_title,
    )
    gc.collect()

    # Description TF-IDF + Ridge
    tfidf_desc = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5)
    desc_tfidf_train = tfidf_desc.fit_transform(df["description"].iloc[:n_train])
    desc_tfidf_test = tfidf_desc.transform(df["description"].iloc[n_train:])

    oof_ridge_desc = np.zeros(n_train, dtype=np.float32)
    test_ridge_desc = np.zeros(n_test, dtype=np.float32)

    for train_idx, val_idx in kf.split(desc_tfidf_train):
        ridge = Ridge(alpha=15.0, solver="sag", random_state=42)
        ridge.fit(desc_tfidf_train[train_idx], target[train_idx])
        oof_ridge_desc[val_idx] = ridge.predict(desc_tfidf_train[val_idx])
        test_ridge_desc += ridge.predict(desc_tfidf_test) / kf.n_splits

    df["ridge_pred_desc"] = np.concatenate([oof_ridge_desc, test_ridge_desc]).astype(
        np.float32
    )

    del tfidf_desc, desc_tfidf_train, desc_tfidf_test, oof_ridge_desc, test_ridge_desc
    gc.collect()

    features = [
        "region",
        "city",
        "parent_category_name",
        "category_name",
        "param_1",
        "param_2",
        "param_3",
        "user_type",
        "image_top_1",
        "item_seq_number",
        "weekday",
        "day",
        "price_is_na",
        "log_price",
        "price_to_cat_diff",
        "price_to_cat_zscore",
        "has_image",
        "title_len",
        "title_words",
        "desc_len",
        "desc_words",
        "user_item_count",
        "ridge_pred_title",
        "ridge_pred_desc",
    ]

    print(f"Features used ({len(features)}): {features}")

    X_train = df.iloc[:n_train][features]
    X_test = df.iloc[n_train:][features]
    y_train = target

    del df
    gc.collect()

    print("Training LightGBM model with 5-fold CV...")
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 100,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)

    for fold, (t_idx, v_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"--- Fold {fold + 1} ---")
        trn_data = lgb.Dataset(
            X_train.iloc[t_idx], label=y_train[t_idx], categorical_feature=cat_cols
        )
        val_data = lgb.Dataset(
            X_train.iloc[v_idx],
            label=y_train[v_idx],
            reference=trn_data,
            categorical_feature=cat_cols,
        )

        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1500,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
        )

        oof_preds[v_idx] = model.predict(
            X_train.iloc[v_idx], num_iteration=model.best_iteration
        )
        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        )

    oof_preds_clipped = np.clip(oof_preds, 0.0, 1.0)
    cv_rmse = root_mean_squared_error(y_train, oof_preds_clipped)
    print(f"\n==========================================")
    print(f"Validation 5-Fold Overall RMSE: {cv_rmse:.5f}")
    print(f"==========================================")

    # Save submission
    os.makedirs("./working", exist_ok=True)
    sub = pd.DataFrame(
        {
            "id": np.arange(n_test, dtype=np.int64),
            "deal_probability": np.clip(test_preds, 0.0, 1.0),
        }
    )
    sub_path = "./working/submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Submission saved successfully to {sub_path} with {len(sub)} rows.")


if __name__ == "__main__":
    main()
