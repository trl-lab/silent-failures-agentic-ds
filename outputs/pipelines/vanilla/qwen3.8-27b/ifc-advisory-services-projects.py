import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-advisory-services-projects/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/ifc-advisory-services-projects/data/test.parquet"
TARGET = "Estimated Total Budget ($)"
RANDOM_STATE = 42

CAT_COLS = [
    "Country",
    "IFC Country Code",
    "IFC Region",
    "Business Line",
    "Status",
    "WB Country Code",
]

DATE_COLS = [
    "Disclosure Date",
    "IFC Approval Date",
    "Projected Start Date",
]


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    parsed = {}
    for col in DATE_COLS:
        if col in df.columns:
            parsed[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            parsed[col] = pd.Series(pd.NaT, index=df.index)

    for col, dt in parsed.items():
        base = col.replace(" Date", "")
        df[f"{base}_year"] = dt.dt.year
        df[f"{base}_month"] = dt.dt.month
        df[f"{base}_day"] = dt.dt.day

    df["days_approval_to_start"] = (
        parsed["Projected Start Date"] - parsed["IFC Approval Date"]
    ).dt.days
    df["days_disclosure_to_approval"] = (
        parsed["Disclosure Date"] - parsed["IFC Approval Date"]
    ).dt.days
    df["days_disclosure_to_start"] = (
        parsed["Disclosure Date"] - parsed["Projected Start Date"]
    ).dt.days

    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Project Name" in df.columns:
        name = df["Project Name"].fillna("").astype(str)
    else:
        name = pd.Series("", index=df.index)

    name_lower = name.str.lower()
    df["name_len"] = name.str.len()
    df["name_word_count"] = name.str.count(r"\S+")
    df["name_has_project"] = name_lower.str.contains("project", regex=False).astype(int)
    df["name_has_esg"] = name_lower.str.contains("esg", regex=False).astype(int)

    if "Country" in df.columns:
        country = df["Country"].fillna("").astype(str).str.lower().str.strip()
    else:
        country = pd.Series("", index=df.index)

    df["name_has_country"] = [
        int(c in n) if c else 0 for c, n in zip(country, name_lower)
    ]

    if "Project URL" in df.columns:
        url = df["Project URL"].fillna("").astype(str)
    else:
        url = pd.Series("", index=df.index)

    slug = url.str.split("/").str[-1].str.lower()
    df["url_slug_len"] = slug.str.len()
    df["url_slug_word_count"] = slug.str.count(r"[\w-]+")

    if "IFC Country Code" in df.columns:
        code = df["IFC Country Code"].fillna("").astype(str).str.lower().str.strip()
    else:
        code = pd.Series("", index=df.index)

    df["url_has_country_code"] = [
        int(c in s) if c else 0 for c, s in zip(code, slug)
    ]

    return df


def get_numeric_columns(df: pd.DataFrame) -> list:
    exclude = set(CAT_COLS) | {TARGET}
    exclude.update(DATE_COLS)
    exclude.update(["Project Number", "Project Name", "Project URL"])

    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = add_date_features(train_df)
    test_df = add_date_features(test_df)

    train_df = add_text_features(train_df)
    test_df = add_text_features(test_df)

    for col in CAT_COLS:
        if col not in train_df.columns:
            train_df[col] = "missing"
        if col not in test_df.columns:
            test_df[col] = "missing"

        train_df[col] = train_df[col].fillna("missing").astype(str)
        test_df[col] = test_df[col].fillna("missing").astype(str)

        categories = sorted(train_df[col].unique())
        categories = [c for c in categories if c != "missing"]
        mapping = {c: i + 1 for i, c in enumerate(categories)}

        train_df[col] = train_df[col].map(mapping).fillna(0).astype(int)
        test_df[col] = test_df[col].map(mapping).fillna(0).astype(int)

    numeric_cols = get_numeric_columns(train_df)

    for col in numeric_cols:
        if col not in test_df.columns:
            test_df[col] = np.nan

        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

        median = train_df[col].median()
        if pd.isna(median):
            median = 0.0

        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)

    feature_cols = numeric_cols + [c for c in CAT_COLS if c in train_df.columns]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET].copy() if TARGET in train_df.columns else pd.Series(np.nan, index=train_df.index)
    X_test = test_df[feature_cols].copy()

    valid = y_train.notna()
    X_train = X_train.loc[valid].reset_index(drop=True)
    y_train = y_train.loc[valid].reset_index(drop=True)

    return X_train, y_train, X_test, feature_cols


def make_model(kind: str, cat_feature_cols: list):
    if kind == "lgbm":
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    return CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        verbose=0,
        cat_features=cat_feature_cols,
    )


def fit_model(model, kind: str, X_train: pd.DataFrame, y_train: pd.Series, cat_feature_cols: list):
    if kind == "lgbm":
        model.fit(
            X_train,
            y_train,
            categorical_feature=cat_feature_cols,
            verbose=False,
        )
    else:
        model.fit(
            X_train,
            y_train,
            verbose=0,
        )
    return model


def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def save_predictions(predictions: np.ndarray) -> None:
    output = pd.DataFrame(
        {
            "id": np.arange(len(predictions), dtype=int),
            TARGET: predictions,
        }
    )
    output.to_csv("predictions.csv", index=False)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    test_df = test_df.drop(columns=[TARGET], errors="ignore")

    X_train, y_train, X_test, feature_cols = prepare_features(train_df, test_df)

    if X_test.shape[0] == 0:
        save_predictions(np.array([], dtype=float))
        return

    y_train = pd.to_numeric(y_train, errors="coerce")
    valid = y_train.notna()
    X_train = X_train.loc[valid].reset_index(drop=True)
    y_train = y_train.loc[valid].reset_index(drop=True)

    if X_train.shape[0] == 0 or len(feature_cols) == 0:
        save_predictions(np.zeros(X_test.shape[0], dtype=float))
        return

    y_train = y_train.astype(float)
    lower_bound = min(0.0, float(y_train.min()))
    fallback_value = float(y_train.mean())

    cat_feature_cols = [
        c for c in CAT_COLS
        if c in feature_cols and X_train[c].nunique(dropna=True) > 1
    ]

    if X_train.shape[0] >= 50:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
    else:
        X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train

    variants = [
        ("lgbm", "raw"),
        ("catboost", "raw"),
    ]

    use_log = bool((y_train >= 0).all()) and y_train.nunique() > 1
    if use_log:
        variants.extend([
            ("lgbm", "log"),
            ("catboost", "log"),
        ])

    val_scores = {}
    for kind, transform in variants:
        y_tr_t = np.log1p(y_tr) if transform == "log" else y_tr
        y_val_t = np.log1p(y_val) if transform == "log" else y_val

        try:
            model = make_model(kind, cat_feature_cols)
            fit_model(model, kind, X_tr, y_tr_t, cat_feature_cols)
            pred_val = predict_model(model, X_val)

            if transform == "log":
                pred_val = np.expm1(pred_val)

            pred_val = np.clip(pred_val, lower_bound, None)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        except Exception:
            rmse = np.inf

        val_scores[(kind, transform)] = rmse

    finite_scores = {k: v for k, v in val_scores.items() if np.isfinite(v)}

    if not finite_scores:
        save_predictions(np.full(X_test.shape[0], fallback_value, dtype=float))
        return

    weights = {k: 1.0 / max(v, 1e-8) for k, v in finite_scores.items()}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = {k: 1.0 / len(weights) for k in weights}
        total_weight = sum(weights.values())

    final_predictions = np.zeros(X_test.shape[0], dtype=float)

    for (kind, transform), weight in weights.items():
        y_train_t = np.log1p(y_train) if transform == "log" else y_train

        try:
            model = make_model(kind, cat_feature_cols)
            fit_model(model, kind, X_train, y_train_t, cat_feature_cols)
            pred = predict_model(model, X_test)

            if transform == "log":
                pred = np.expm1(pred)

            pred = np.clip(pred, lower_bound, None)
        except Exception:
            pred = np.full(X_test.shape[0], fallback_value, dtype=float)

        final_predictions += (weight / total_weight) * pred

    save_predictions(final_predictions)


if __name__ == "__main__":
    main()