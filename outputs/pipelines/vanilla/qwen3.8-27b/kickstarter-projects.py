import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/kickstarter-projects/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/kickstarter-projects/data/test.parquet"
TARGET = "status"
RANDOM_STATE = 42


def positive_label(y: pd.Series) -> object:
    vals = y.dropna().unique()
    if "successful" in vals:
        return "successful"
    if len(vals) == 2:
        return sorted(map(str, vals))[-1]
    return vals[0] if len(vals) else 1


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = [c for c in ["id", "name", TARGET] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    launched_dt = None
    deadline_dt = None

    if "launched_at" in df.columns:
        launched_dt = pd.to_datetime(df["launched_at"], errors="coerce")
    if "deadline" in df.columns:
        deadline_dt = pd.to_datetime(df["deadline"], errors="coerce")

    if launched_dt is not None:
        df["launched_year"] = launched_dt.dt.year
        df["launched_month"] = launched_dt.dt.month
        df["launched_day"] = launched_dt.dt.day
        df["launched_dayofweek"] = launched_dt.dt.dayofweek
        df["launched_is_weekend"] = (launched_dt.dt.dayofweek >= 5).astype(int)

    if deadline_dt is not None:
        df["deadline_year"] = deadline_dt.dt.year
        df["deadline_month"] = deadline_dt.dt.month
        df["deadline_day"] = deadline_dt.dt.day
        df["deadline_dayofweek"] = deadline_dt.dt.dayofweek
        df["deadline_is_weekend"] = (deadline_dt.dt.dayofweek >= 5).astype(int)

    if launched_dt is not None and deadline_dt is not None:
        computed_duration = (deadline_dt - launched_dt).dt.total_seconds() / 86400.0
        if "duration" in df.columns:
            df["duration"] = df["duration"].fillna(computed_duration)
        else:
            df["duration"] = computed_duration

    for col in ["launched_at", "deadline"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "goal_usd" in df.columns:
        df["log_goal_usd"] = np.log1p(df["goal_usd"].clip(lower=0))

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    for col in categorical_cols:
        df[col] = df[col].astype("string").fillna("missing")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fit_transformer(train_df: pd.DataFrame):
    train_df = prepare_features(train_df)
    numeric_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in train_df.columns if c not in numeric_cols]

    numeric_imputer = SimpleImputer(strategy="median")
    numeric_imputer.fit(train_df[numeric_cols])

    categorical_mappings = {}
    for col in categorical_cols:
        categories = sorted(train_df[col].dropna().unique().tolist())
        categorical_mappings[col] = {value: idx for idx, value in enumerate(categories)}

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df = prepare_features(df)
        out = pd.DataFrame(index=df.index)

        if numeric_cols:
            out[numeric_cols] = numeric_imputer.transform(df[numeric_cols])

        for col in categorical_cols:
            out[col] = df[col].map(categorical_mappings[col]).fillna(-1).astype(int)

        return out[numeric_cols + categorical_cols]

    return transform


def make_models():
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
        "catboost": CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
    }


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_raw = train_df[TARGET]
    pos_label = positive_label(y_raw)
    y = (y_raw == pos_label).astype(int)

    X_raw = train_df.drop(columns=[TARGET])

    if y.nunique() < 2:
        predictions = pd.DataFrame({"id": np.arange(len(test_df)), "status": float(y.mode().iloc[0])})
        predictions.to_csv("predictions.csv", index=False)
        return

    train_idx, val_idx = train_test_split(
        np.arange(len(X_raw)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_raw = X_raw.iloc[train_idx]
    X_val_raw = X_raw.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    transform = fit_transformer(X_train_raw)
    X_train = transform(X_train_raw)
    X_val = transform(X_val_raw)
    X_test = transform(test_df)

    models = make_models()
    val_probs = {}
    val_scores = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        val_probs[name] = prob
        val_scores[name] = roc_auc_score(y_val, prob)

    ensemble_prob = np.mean(list(val_probs.values()), axis=0)
    val_scores["ensemble"] = roc_auc_score(y_val, ensemble_prob)

    best_name = max(val_scores, key=val_scores.get)

    transform_full = fit_transformer(X_raw)
    X_train_full = transform_full(X_raw)
    X_test_full = transform_full(test_df)

    if best_name == "ensemble":
        final_models = make_models()
        final_probs = []
        for model in final_models.values():
            model.fit(X_train_full, y)
            final_probs.append(model.predict_proba(X_test_full)[:, 1])
        test_prob = np.mean(final_probs, axis=0)
    else:
        final_model = make_models()[best_name]
        final_model.fit(X_train_full, y)
        test_prob = final_model.predict_proba(X_test_full)[:, 1]

    predictions = pd.DataFrame({"id": np.arange(len(test_df)), "status": test_prob})
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()