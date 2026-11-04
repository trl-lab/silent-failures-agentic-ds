import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/osic-pulmonary-fibrosis-progression/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/osic-pulmonary-fibrosis-progression/data/test.parquet"

TARGET = "FVC"
NUMERIC_COLS = ["Weeks", "Percent", "Age"]
CAT_COLS = ["Sex", "SmokingStatus"]
RANDOM_STATE = 42


class Preprocessor:
    def __init__(self, numeric_cols, cat_cols):
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.numeric_medians = {}
        self.cat_categories = {}

    def fit(self, df):
        for col in self.numeric_cols:
            if col in df.columns:
                median = df[col].median()
                self.numeric_medians[col] = 0.0 if pd.isna(median) else float(median)
            else:
                self.numeric_medians[col] = 0.0

        for col in self.cat_cols:
            if col in df.columns:
                categories = df[col].dropna().astype(str).unique()
                self.cat_categories[col] = sorted(set(categories) | {"Unknown"})
            else:
                self.cat_categories[col] = ["Unknown"]

        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)

        for col in self.numeric_cols:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_medians[col])
            else:
                out[col] = self.numeric_medians[col]

        for col in self.cat_cols:
            if col in df.columns:
                values = df[col].fillna("Unknown").astype(str)
            else:
                values = pd.Series("Unknown", index=df.index)

            for category in self.cat_categories[col]:
                out[f"{col}_{category}"] = (values == category).astype(float)

        return out


def make_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "catboost": CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


def clean_predictions(pred):
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pred, 0.0, None)


def evaluate_models(models, X_train, y_train, X_val, y_val, percent_val=None):
    val_preds = []
    weights = []

    for model in models.values():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        if percent_val is not None:
            pred = pred * percent_val

        pred = clean_predictions(pred)
        mae = mean_absolute_error(y_val, pred)
        val_preds.append(pred)
        weights.append(1.0 / max(mae, 1e-8))

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    val_pred = np.average(val_preds, axis=0, weights=weights)

    return clean_predictions(val_pred), weights


def predict_models(models, X_train, y_train, X_test, weights, percent_test=None):
    test_preds = []

    for model, weight in zip(models.values(), weights):
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        if percent_test is not None:
            pred = pred * percent_test

        pred = clean_predictions(pred)
        test_preds.append(pred)

    test_pred = np.average(test_preds, axis=0, weights=weights)
    return clean_predictions(test_pred)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError(f"Target column {TARGET!r} not found in train data.")

    if "Patient" in train_df.columns:
        patients = train_df["Patient"].unique()
        train_patients, val_patients = train_test_split(
            patients,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        train_part = train_df[train_df["Patient"].isin(train_patients)].copy()
        val_part = train_df[train_df["Patient"].isin(val_patients)].copy()
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        train_part = train_df.iloc[train_idx].copy()
        val_part = train_df.iloc[val_idx].copy()

    val_preprocessor = Preprocessor(NUMERIC_COLS, CAT_COLS).fit(train_part)
    X_train_val = val_preprocessor.transform(train_part)
    X_val = val_preprocessor.transform(val_part)

    y_train_val = train_part[TARGET].to_numpy(dtype=float)
    y_val = val_part[TARGET].to_numpy(dtype=float)
    percent_train_val = X_train_val["Percent"].to_numpy(dtype=float)
    percent_val = X_val["Percent"].to_numpy(dtype=float)

    direct_val_pred, direct_weights = evaluate_models(
        make_models(),
        X_train_val,
        y_train_val,
        X_val,
        y_val,
        percent_val=None,
    )

    valid_ratio = (y_train_val > 0) & (percent_train_val > 0)
    if valid_ratio.sum() > 0:
        X_train_val_ratio = X_train_val[valid_ratio]
        y_train_val_ratio = y_train_val[valid_ratio] / percent_train_val[valid_ratio]

        ratio_val_pred, ratio_weights = evaluate_models(
            make_models(),
            X_train_val_ratio,
            y_train_val_ratio,
            X_val,
            y_val,
            percent_val=percent_val,
        )
    else:
        ratio_val_pred = direct_val_pred
        ratio_weights = direct_weights

    direct_mae = mean_absolute_error(y_val, direct_val_pred)
    ratio_mae = mean_absolute_error(y_val, ratio_val_pred)

    direct_weight = 1.0 / max(direct_mae, 1e-8)
    ratio_weight = 1.0 / max(ratio_mae, 1e-8)
    total_weight = direct_weight + ratio_weight
    direct_weight /= total_weight
    ratio_weight /= total_weight

    final_preprocessor = Preprocessor(NUMERIC_COLS, CAT_COLS).fit(train_df)
    X_all = final_preprocessor.transform(train_df)
    X_test = final_preprocessor.transform(test_df)

    y_all = train_df[TARGET].to_numpy(dtype=float)
    percent_all = X_all["Percent"].to_numpy(dtype=float)
    percent_test = X_test["Percent"].to_numpy(dtype=float)

    final_direct = predict_models(
        make_models(),
        X_all,
        y_all,
        X_test,
        weights=direct_weights,
        percent_test=None,
    )

    valid_ratio_all = (y_all > 0) & (percent_all > 0)
    if valid_ratio_all.sum() > 0:
        X_all_ratio = X_all[valid_ratio_all]
        y_all_ratio = y_all[valid_ratio_all] / percent_all[valid_ratio_all]

        final_ratio = predict_models(
            make_models(),
            X_all_ratio,
            y_all_ratio,
            X_test,
            weights=ratio_weights,
            percent_test=percent_test,
        )
    else:
        final_ratio = final_direct

    final_pred = direct_weight * final_direct + ratio_weight * final_ratio
    final_pred = clean_predictions(final_pred)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_df), dtype=int),
            "FVC": final_pred,
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()