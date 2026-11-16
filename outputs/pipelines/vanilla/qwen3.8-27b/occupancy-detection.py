import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/occupancy-detection/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/occupancy-detection/data/test.parquet"
TARGET = "Occupancy"


def find_column(df, names):
    for name in names:
        for col in df.columns:
            if str(col).lower() == str(name).lower():
                return col
    return None


def prepare_features(df, fit=False, encoders=None):
    df = df.copy()

    id_col = find_column(df, ["id", "index", "row_id"])
    if id_col is not None:
        df = df.drop(columns=[id_col])

    target_col = find_column(df, [TARGET])
    if target_col is not None:
        df = df.drop(columns=[target_col])

    date_col = find_column(df, ["date", "datetime", "timestamp"])
    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        df["date_year"] = dt.dt.year
        df["date_month"] = dt.dt.month
        df["date_day"] = dt.dt.day
        df["date_dayofweek"] = dt.dt.dayofweek
        df["date_hour"] = dt.dt.hour
        df["date_minute"] = dt.dt.minute
        df["date_second"] = dt.dt.second
        df["date_dayofyear"] = dt.dt.dayofyear
        df["date_is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
        df["date_hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        df["date_hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        df["date_dayofweek_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        df["date_dayofweek_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
        df = df.drop(columns=[date_col])

    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            s = df[col]
            non_null = int(s.notna().sum())
            if non_null == 0:
                df[col] = np.nan
                continue

            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().sum() / non_null >= 0.8:
                df[col] = numeric
            else:
                categorical_cols.append(col)

    if fit:
        encoders = {}
        for col in categorical_cols:
            s = df[col].astype("string")
            s = s.where(s.notna(), "missing")
            cats = sorted(s.unique().tolist(), key=lambda x: str(x))
            mapping = {str(c): i for i, c in enumerate(cats)}
            encoders[col] = mapping
            df[col] = s.map(mapping).astype(float)
    else:
        encoders = encoders or {}
        for col in categorical_cols:
            s = df[col].astype("string")
            s = s.where(s.notna(), "missing")
            mapping = encoders.get(col, {})
            df[col] = s.map(mapping).astype(float)

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df, encoders


def make_models():
    return {
        "lgbm": LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
            tree_method="hist",
        ),
        "cat": CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
    }


def select_features(X_train, X_val=None, X_test=None):
    cols = [c for c in X_train.columns if X_train[c].notna().any()]
    X_train = X_train.reindex(columns=cols)
    if X_val is not None:
        X_val = X_val.reindex(columns=cols)
    if X_test is not None:
        X_test = X_test.reindex(columns=cols)
    return X_train, X_val, X_test


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    target_col = find_column(train_df, [TARGET])
    if target_col is None:
        raise ValueError(f"Target column {TARGET!r} not found in train data.")

    train_df = train_df.dropna(subset=[target_col])
    y_full = train_df[target_col].astype(int)

    if y_full.nunique() > 1:
        train_part, val_part = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
            stratify=y_full,
        )
    else:
        train_part, val_part = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
        )

    y_train_part = train_part[target_col].astype(int)
    y_val_part = val_part[target_col].astype(int)

    X_train_part, encoders_part = prepare_features(train_part, fit=True)
    X_val_part, _ = prepare_features(val_part, fit=False, encoders=encoders_part)
    X_train_part, X_val_part, _ = select_features(X_train_part, X_val_part)

    weights = {}
    if X_train_part.shape[1] > 0 and y_val_part.nunique() > 1:
        val_models = make_models()
        val_preds = {}
        for name, model in val_models.items():
            model.fit(X_train_part, y_train_part)
            val_preds[name] = model.predict_proba(X_val_part)[:, 1]

        aucs = {
            name: roc_auc_score(y_val_part, pred)
            for name, pred in val_preds.items()
        }

        raw_weights = {name: max(auc - 0.5, 0.0) for name, auc in aucs.items()}
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            weights = {name: w / total_weight for name, w in raw_weights.items()}
        else:
            weights = {name: 1.0 / len(aucs) for name in aucs}
    else:
        weights = {"lgbm": 1.0 / 3, "xgb": 1.0 / 3, "cat": 1.0 / 3}

    X_full, encoders_full = prepare_features(train_df, fit=True)
    X_test, _ = prepare_features(test_df, fit=False, encoders=encoders_full)
    X_full, _, X_test = select_features(X_full, None, X_test)

    if X_full.shape[1] == 0:
        final_proba = np.full(len(test_df), float(y_full.mean()))
    else:
        final_models = make_models()
        weighted_preds = []
        all_preds = []

        for name, model in final_models.items():
            model.fit(X_full, y_full)
            proba = model.predict_proba(X_test)[:, 1]
            all_preds.append(proba)
            weighted_preds.append(weights.get(name, 0.0) * proba)

        final_proba = np.sum(weighted_preds, axis=0)
        if not np.isfinite(final_proba).all() or np.allclose(final_proba, 0.0):
            final_proba = np.mean(all_preds, axis=0)

    final_proba = np.nan_to_num(final_proba, nan=0.5, posinf=1.0, neginf=0.0)
    final_proba = np.clip(final_proba, 0.0, 1.0)

    submission = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "Occupancy": final_proba,
    })
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()