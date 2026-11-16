import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/meta-critic_whisky/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/meta-critic_whisky/data/test.parquet"
TARGET = "Meta Critic"
RANDOM_STATE = 42

KEYWORDS = [
    "peaty", "peat", "port", "sherry", "oloroso", "pedro", "ximenez", "px",
    "bourbon", "rye", "wine", "madeira", "marsala", "amontillado", "fino",
    "manzanilla", "islay", "cask", "barrel", "bottle", "single", "limited",
    "reserve", "vintage", "finish", "finished", "cask strength", "highland",
    "lowland", "island", "speyside", "campbeltown", "jura", "orkney",
    "ireland", "canada", "japan", "taiwan", "usa", "scotland", "england",
    "france", "spain", "italy", "germany", "australia", "india", "mexico",
    "argentina", "chile", "new zealand", "south africa", "caribbean",
    "panama", "honduras", "guatemala", "peru", "bolivia", "uruguay",
    "paraguay", "venezuela", "colombia", "ecuador", "brazil"
]

CAT_COLS = ["Class", "Super Cluster", "Cluster", "Country", "Type"]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def clean_name(value):
    if isinstance(value, str):
        return value.lower()
    return ""


def extract_age(value):
    s = clean_name(value)
    if not s:
        return np.nan

    patterns = [
        r"(\d{1,3})\s*yo\b",
        r"(\d{1,3})\s*yr\b",
        r"(\d{1,3})\s*years?\b",
        r"(\d{1,3})\s*year old\b",
        r"(\d{1,3})\s*y\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            age = int(match.group(1))
            if 1 <= age <= 80:
                return float(age)

    numbers = re.findall(r"(?<!\d)(\d{1,3})(?!\d)", s)
    ages = [int(x) for x in numbers if 1 <= int(x) <= 60]
    if ages:
        return float(max(ages))

    return np.nan


def extract_vintage(value):
    s = clean_name(value)
    if not s:
        return np.nan

    match = re.search(r"(?<!\d)(1[89]\d{2}|20\d{2}|21\d{2})(?!\d)", s)
    if match:
        return float(int(match.group(1)))

    return np.nan


def cost_score(value):
    if not isinstance(value, str):
        return 0.0
    score = value.count("$")
    if "+" in value:
        score += 1
    return float(score)


def keyword_flags(name):
    s = clean_name(name)
    return [int(kw in s) for kw in KEYWORDS]


def make_numeric_columns(df):
    df = df.copy()
    df["Whisky"] = df["Whisky"].fillna("").astype(str)

    df["cost_score"] = df["Cost"].apply(cost_score)
    df["age"] = df["Whisky"].apply(extract_age)
    df["vintage_year"] = df["Whisky"].apply(extract_vintage)

    flag_matrix = np.array([keyword_flags(name) for name in df["Whisky"]], dtype=float)
    flag_cols = [f"kw_{i}" for i in range(len(KEYWORDS))]
    for i, col in enumerate(flag_cols):
        df[col] = flag_matrix[:, i]

    numeric_cols = ["cost_score", "age", "vintage_year"] + flag_cols
    return df, numeric_cols


def compute_numeric_medians(df, numeric_cols):
    medians = {}
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        median = values.median()
        medians[col] = 0.0 if pd.isna(median) else float(median)
    return medians


def fit_text_vectorizer(df):
    names = df["Whisky"].fillna("").astype(str).tolist()
    if not any(name.strip() for name in names):
        return None
    vectorizer = CountVectorizer(
        lowercase=True,
        binary=True,
        max_features=250,
        min_df=1,
        token_pattern=r"\w+",
    )
    try:
        vectorizer.fit(names)
    except ValueError:
        return None
    return vectorizer


def transform_text(df, vectorizer):
    names = df["Whisky"].fillna("").astype(str).tolist()
    if vectorizer is None:
        return np.zeros((len(df), 0), dtype=float)
    try:
        return vectorizer.transform(names).toarray().astype(float)
    except ValueError:
        return np.zeros((len(df), 0), dtype=float)


def fit_categories(df, cat_cols):
    categories = {}
    for col in cat_cols:
        values = df[col].fillna("missing").astype(str).unique()
        categories[col] = sorted(values)
    return categories


def transform_categories(df, categories, cat_cols):
    parts = []
    for col in cat_cols:
        values = df[col].fillna("missing").astype(str).to_numpy()
        for category in categories[col]:
            parts.append((values == category).astype(float))
    if not parts:
        return np.zeros((len(df), 0), dtype=float)
    return np.column_stack(parts)


def prepare_features(df, vectorizer=None, categories=None, numeric_medians=None):
    df, numeric_cols = make_numeric_columns(df)

    if numeric_medians is None:
        numeric_medians = compute_numeric_medians(df, numeric_cols)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(numeric_medians[col])

    if categories is None:
        categories = fit_categories(df, CAT_COLS)

    x_numeric = df[numeric_cols].to_numpy(dtype=float)
    x_text = transform_text(df, vectorizer)
    x_cat = transform_categories(df, categories, CAT_COLS)

    return np.hstack([x_numeric, x_text, x_cat])


def make_model(name):
    if name == "lgbm":
        return LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    if name == "xgb":
        return XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="rmse",
        )
    if name == "cat":
        return CatBoostRegressor(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        )
    raise ValueError(f"Unknown model: {name}")


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET]).copy()
    if TARGET in test_df.columns:
        test_df = test_df.drop(columns=[TARGET]).copy()

    y_full = train_df[TARGET].astype(float).to_numpy()

    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_split = train_df.iloc[train_idx]
    val_split = train_df.iloc[val_idx]

    vectorizer = fit_text_vectorizer(train_split)
    categories = fit_categories(train_split, CAT_COLS)
    numeric_medians = None

    x_train, y_train = prepare_features(train_split, vectorizer, categories, numeric_medians), train_split[TARGET].astype(float).to_numpy()
    x_val, y_val = prepare_features(val_split, vectorizer, categories, numeric_medians), val_split[TARGET].astype(float).to_numpy()

    model_names = ["lgbm", "xgb", "cat"]
    val_predictions = []
    val_scores = []

    for name in model_names:
        model = make_model(name)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        pred = np.asarray(pred, dtype=float)
        pred = np.where(np.isfinite(pred), pred, float(np.mean(y_train)))
        val_predictions.append(pred)
        val_scores.append(rmse(y_val, pred))

    val_scores = np.asarray(val_scores, dtype=float)
    finite = np.isfinite(val_scores)
    if not finite.any():
        weights = np.ones(len(model_names), dtype=float)
    else:
        weights = np.zeros(len(model_names), dtype=float)
        weights[finite] = 1.0 / np.maximum(val_scores[finite], 1e-6)

    weights /= weights.sum()

    ensemble_val_pred = np.average(val_predictions, axis=0, weights=weights)
    ensemble_val_rmse = rmse(y_val, ensemble_val_pred)

    best_idx = int(np.argmin(val_scores))
    best_single_rmse = float(val_scores[best_idx])

    if best_single_rmse < ensemble_val_rmse:
        selected_names = [model_names[best_idx]]
        selected_weights = np.array([1.0], dtype=float)
    else:
        selected_names = model_names
        selected_weights = weights

    x_full = prepare_features(train_df, None, None, None)
    x_test = prepare_features(test_df, None, None, None)

    # Refit selected models on the full training set using full-train preprocessing.
    vectorizer_full = fit_text_vectorizer(train_df)
    categories_full = fit_categories(train_df, CAT_COLS)
    x_full = prepare_features(train_df, vectorizer_full, categories_full, None)
    x_test = prepare_features(test_df, vectorizer_full, categories_full, None)

    final_predictions = np.zeros(len(test_df), dtype=float)
    for name, weight in zip(selected_names, selected_weights):
        model = make_model(name)
        model.fit(x_full, y_full)
        pred = model.predict(x_test)
        pred = np.asarray(pred, dtype=float)
        pred = np.where(np.isfinite(pred), pred, float(np.mean(y_full)))
        final_predictions += weight * pred

    final_predictions = np.clip(final_predictions, 0.0, 10.0)

    output = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "Meta Critic": final_predictions.astype(float),
    })
    output.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()