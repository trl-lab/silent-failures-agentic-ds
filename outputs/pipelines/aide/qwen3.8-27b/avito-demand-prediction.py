import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifiers / raw hashes that are not useful without extra aggregation.
    drop_cols = ["item_id", "image", "user_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Date features.
    if "activation_date" in df.columns:
        df["activation_date"] = pd.to_datetime(df["activation_date"], errors="coerce")
        df["activation_year"] = df["activation_date"].dt.year
        df["activation_month"] = df["activation_date"].dt.month
        df["activation_day"] = df["activation_date"].dt.day
        df["activation_dayofweek"] = df["activation_date"].dt.dayofweek
        df["activation_dayofyear"] = df["activation_date"].dt.dayofyear
        df = df.drop(columns=["activation_date"])

    # Numeric transformations and missing indicators.
    numeric_cols = ["price", "image_top_1", "item_seq_number"]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        df[f"{col}_isna"] = df[col].isna().astype(np.int8)
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        if col == "price":
            df[col] = np.log1p(df[col].clip(lower=0))
        elif col == "image_top_1":
            df[col] = np.log1p(df[col].clip(lower=0))
        else:
            df[col] = np.log1p(df[col].clip(lower=0))

    # Simple text features.
    if "title" in df.columns:
        title = df["title"].fillna("").astype(str)
        df["title_len"] = title.str.len().astype("int32")
        df["title_word_count"] = title.str.split().str.len().fillna(0).astype("int32")
        df["title_digit_count"] = title.str.count(r"\d").astype("int32")
        df = df.drop(columns=["title"])

    if "description" in df.columns:
        desc = df["description"].fillna("").astype(str)
        df["description_len"] = desc.str.len().astype("int32")
        df["description_word_count"] = (
            desc.str.split().str.len().fillna(0).astype("int32")
        )
        df["description_digit_count"] = desc.str.count(r"\d").astype("int32")
        df = df.drop(columns=["description"])

    if "title_len" in df.columns and "description_len" in df.columns:
        df["desc_title_len_ratio"] = (
            df["description_len"] / (df["title_len"] + 1)
        ).astype("float32")

    return df


def main() -> None:
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    y = train.pop("deal_probability")
    if "deal_probability" in test.columns:
        test = test.drop(columns=["deal_probability"])

    train = engineer_features(train)
    test = engineer_features(test)

    categorical_cols = [
        "parent_category_name",
        "category_name",
        "user_type",
        "region",
        "city",
        "param_1",
        "param_2",
        "param_3",
    ]

    # Align categorical levels between train and test using train levels only.
    # Unseen test categories become NaN, which LightGBM can handle.
    categorical_cols = [
        c for c in categorical_cols if c in train.columns and c in test.columns
    ]
    for col in categorical_cols:
        train[col] = train[col].astype("category")
        test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

    # Keep only columns present in both dataframes and not object dtype.
    feature_cols = [
        c for c in train.columns if c in test.columns and train[c].dtype != object
    ]

    X = train[feature_cols]
    X_test = test[feature_cols]

    # Choose objective based on target values.
    unique_y = set(y.dropna().unique())
    if unique_y <= {0, 1}:
        objective = "binary"
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        objective = "regression"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    params = {
        "objective": objective,
        "metric": "rmse",
        "learning_rate": 0.1,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
    }

    fold_scores = []
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_valid_fold = y.iloc[valid_idx]

        train_data = lgb.Dataset(
            X_train_fold,
            label=y_train_fold,
            categorical_feature=categorical_cols,
        )
        valid_data = lgb.Dataset(
            X_valid_fold,
            label=y_valid_fold,
            categorical_feature=categorical_cols,
            reference=train_data,
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        fold_rmse = model.best_score["valid"][0][1]
        fold_scores.append(fold_rmse)
        best_iterations.append(model.best_iteration)
        print(f"Fold {fold} RMSE: {fold_rmse:.6f}")

    cv_rmse = float(np.mean(fold_scores))
    print(f"CV RMSE: {cv_rmse:.6f}")

    final_rounds = max(1, int(np.mean(best_iterations)))
    full_data = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_cols,
    )

    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_pred = final_model.predict(X_test)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=np.int64),
            "deal_probability": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
