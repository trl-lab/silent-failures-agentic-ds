import ast
import os
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


def parse_json_safely(val):
    if pd.isna(val) or val is None or val == "":
        return []
    if isinstance(val, (list, dict)):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


def extract_names(val):
    parsed = parse_json_safely(val)
    if isinstance(parsed, list):
        return [
            item["name"] for item in parsed if isinstance(item, dict) and "name" in item
        ]
    elif isinstance(parsed, dict) and "name" in parsed:
        return [parsed["name"]]
    return []


def extract_crew_role(val, job_name="Director"):
    parsed = parse_json_safely(val)
    if isinstance(parsed, list):
        return [
            item["name"]
            for item in parsed
            if isinstance(item, dict) and item.get("job") == job_name
        ]
    return []


def fix_release_date(date_str):
    if pd.isna(date_str) or not str(date_str).strip():
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        parts = str(date_str).split("/")
        if len(parts) == 3:
            m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
            if y <= 20:
                y += 2000
            elif y < 100:
                y += 1900
            dt = pd.Timestamp(year=y, month=m, day=d)
            return dt.year, dt.month, dt.day, dt.dayofweek, dt.quarter
        else:
            dt = pd.to_datetime(date_str)
            return dt.year, dt.month, dt.day, dt.dayofweek, dt.quarter
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def engineer_features(df):
    feat = pd.DataFrame(index=df.index)
    feat["id"] = df["id"]

    # Numeric base features
    feat["budget"] = df["budget"].fillna(0)
    feat["log_budget"] = np.log1p(feat["budget"])
    feat["budget_is_zero"] = (feat["budget"] == 0).astype(int)

    feat["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    feat["log_popularity"] = np.log1p(np.maximum(0, feat["popularity"]))

    feat["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(
        df["runtime"].median()
    )
    feat["runtime_is_zero"] = (feat["runtime"] == 0).astype(int)

    feat["budget_to_popularity"] = feat["budget"] / (feat["popularity"] + 1.0)
    feat["budget_to_runtime"] = feat["budget"] / (feat["runtime"] + 1.0)

    # Text length & presence features
    feat["has_homepage"] = df["homepage"].notna().astype(int)
    feat["has_tagline"] = df["tagline"].notna().astype(int)
    feat["has_overview"] = df["overview"].notna().astype(int)
    feat["has_belongs_to_collection"] = df["belongs_to_collection"].notna().astype(int)
    feat["has_poster"] = df["poster_path"].notna().astype(int)

    feat["title_len"] = df["original_title"].astype(str).str.len()
    feat["overview_len"] = df["overview"].astype(str).str.len()
    feat["tagline_len"] = df["tagline"].astype(str).str.len()

    # Temporal features
    date_info = [fix_release_date(d) for d in df["release_date"]]
    feat["release_year"] = [x[0] for x in date_info]
    feat["release_month"] = [x[1] for x in date_info]
    feat["release_day"] = [x[2] for x in date_info]
    feat["release_dayofweek"] = [x[3] for x in date_info]
    feat["release_quarter"] = [x[4] for x in date_info]

    # Impute missing date values with median
    feat["release_year"] = feat["release_year"].fillna(feat["release_year"].median())
    feat["release_month"] = feat["release_month"].fillna(feat["release_month"].median())
    feat["release_day"] = feat["release_day"].fillna(feat["release_day"].median())
    feat["release_dayofweek"] = feat["release_dayofweek"].fillna(
        feat["release_dayofweek"].median()
    )
    feat["release_quarter"] = feat["release_quarter"].fillna(
        feat["release_quarter"].median()
    )

    # JSON list counts
    feat["num_genres"] = df["genres"].apply(lambda x: len(parse_json_safely(x)))
    feat["num_production_companies"] = df["production_companies"].apply(
        lambda x: len(parse_json_safely(x))
    )
    feat["num_production_countries"] = df["production_countries"].apply(
        lambda x: len(parse_json_safely(x))
    )
    feat["num_spoken_languages"] = df["spoken_languages"].apply(
        lambda x: len(parse_json_safely(x))
    )
    feat["num_cast"] = df["cast"].apply(lambda x: len(parse_json_safely(x)))
    feat["num_crew"] = df["crew"].apply(lambda x: len(parse_json_safely(x)))
    feat["num_keywords"] = df["Keywords"].apply(lambda x: len(parse_json_safely(x)))

    # Language feature
    feat["is_en"] = (df["original_language"] == "en").astype(int)

    return feat


def add_categorical_indicators(train_df, test_df, train_raw, test_raw):
    # Frequent genres
    all_genres = [
        g for sublist in train_raw["genres"].apply(extract_names) for g in sublist
    ]
    top_genres = pd.Series(all_genres).value_counts().head(20).index.tolist()
    for g in top_genres:
        col_name = f'genre_{re.sub(r"[^a-zA-Z0-9]", "_", g)}'
        train_df[col_name] = train_raw["genres"].apply(
            lambda x: int(g in extract_names(x))
        )
        test_df[col_name] = test_raw["genres"].apply(
            lambda x: int(g in extract_names(x))
        )

    # Frequent production companies
    all_companies = [
        c
        for sublist in train_raw["production_companies"].apply(extract_names)
        for c in sublist
    ]
    top_companies = pd.Series(all_companies).value_counts().head(20).index.tolist()
    for c in top_companies:
        col_name = f'company_{re.sub(r"[^a-zA-Z0-9]", "_", c[:15])}'
        train_df[col_name] = train_raw["production_companies"].apply(
            lambda x: int(c in extract_names(x))
        )
        test_df[col_name] = test_raw["production_companies"].apply(
            lambda x: int(c in extract_names(x))
        )

    # Frequent directors
    all_directors = [
        d
        for sublist in train_raw["crew"].apply(
            lambda x: extract_crew_role(x, "Director")
        )
        for d in sublist
    ]
    top_directors = pd.Series(all_directors).value_counts().head(20).index.tolist()
    for d in top_directors:
        col_name = f'director_{re.sub(r"[^a-zA-Z0-9]", "_", d[:15])}'
        train_df[col_name] = train_raw["crew"].apply(
            lambda x: int(d in extract_crew_role(x, "Director"))
        )
        test_df[col_name] = test_raw["crew"].apply(
            lambda x: int(d in extract_crew_role(x, "Director"))
        )

    return train_df, test_df


def main():
    print("Loading data...")
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    print(f"Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")

    X_train = engineer_features(train_raw)
    X_test = engineer_features(test_raw)

    X_train, X_test = add_categorical_indicators(X_train, X_test, train_raw, test_raw)

    feature_cols = [c for c in X_train.columns if c != "id"]
    y_train = np.log1p(np.maximum(0, train_raw["revenue"].values))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(train_raw))
    test_predictions = np.zeros(len(test_raw))

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train.iloc[train_idx][feature_cols], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx][feature_cols], y_train[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        val_pred = model.predict(X_va)
        oof_predictions[val_idx] = val_pred
        test_predictions += model.predict(X_test[feature_cols]) / kf.n_splits

    cv_rmsle = root_mean_squared_error(y_train, oof_predictions)
    print(f"5-Fold CV RMSLE: {cv_rmsle:.5f}")

    # Save predictions
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {"id": test_raw["id"], "revenue": np.expm1(np.maximum(0, test_predictions))}
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission saved successfully to ./working/submission.csv")


if __name__ == "__main__":
    main()
