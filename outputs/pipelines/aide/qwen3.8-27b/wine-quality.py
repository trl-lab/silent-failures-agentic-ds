import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error


def encode_color(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "color" not in df.columns:
        return df

    color = df["color"]
    if not pd.api.types.is_numeric_dtype(color):
        mapped = color.astype(str).str.strip().str.lower().map({"red": 1, "white": 0})
        numeric = pd.to_numeric(color, errors="coerce")
        if mapped.notna().sum() >= numeric.notna().sum():
            df["color"] = mapped
        else:
            df["color"] = numeric
    else:
        df["color"] = pd.to_numeric(color, errors="coerce")

    return df


def prepare_features(df: pd.DataFrame, target: str = "quality") -> pd.DataFrame:
    df = df.copy()

    for col in [target, "id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = encode_color(df)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        if "free_sulfur_dioxide" in df.columns and "total_sulfur_dioxide" in df.columns:
            df["so2_ratio"] = df["free_sulfur_dioxide"] / df[
                "total_sulfur_dioxide"
            ].replace(0, np.nan)
            df["bound_so2"] = df["total_sulfur_dioxide"] - df["free_sulfur_dioxide"]

        if "alcohol" in df.columns and "density" in df.columns:
            df["alcohol_density"] = df["alcohol"] * df["density"]

        if "alcohol" in df.columns and "volatile_acidity" in df.columns:
            df["alcohol_volatile"] = df["alcohol"] * df["volatile_acidity"]

        if "citric_acid" in df.columns and "fixed_acidity" in df.columns:
            df["citric_fixed_ratio"] = df["citric_acid"] / df["fixed_acidity"].replace(
                0, np.nan
            )

        if "residual_sugar" in df.columns and "density" in df.columns:
            df["sugar_density"] = df["residual_sugar"] * df["density"]

        if "sulphates" in df.columns and "alcohol" in df.columns:
            df["sulphates_alcohol"] = df["sulphates"] * df["alcohol"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols]


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "quality"
    train = train.dropna(subset=[target])
    y = train[target].astype(int).to_numpy()

    X = prepare_features(train, target)
    X_test = prepare_features(test, target)

    common_cols = [c for c in X.columns if c in X_test.columns]
    X = X[common_cols].replace([np.inf, -np.inf], np.nan)
    X_test = X_test[common_cols].replace([np.inf, -np.inf], np.nan)

    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians).astype(float)
    X_test = X_test.fillna(medians).astype(float)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X), dtype=float)
    best_iterations = []
    num_boost_round = 2000

    for train_idx, val_idx in kf.split(X):
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y[train_idx], free_raw_data=False)
        dval = lgb.Dataset(
            X.iloc[val_idx], label=y[val_idx], reference=dtrain, free_raw_data=False
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iter = getattr(model, "best_iteration", 0)
        if best_iter is None or best_iter <= 0:
            best_iter = num_boost_round
        best_iter = int(best_iter)
        best_iterations.append(best_iter)

        val_pred = model.predict(X.iloc[val_idx], num_iteration=best_iter)
        val_pred = np.clip(val_pred, 0.0, 10.0)
        oof_preds[val_idx] = val_pred

    oof_rmse = root_mean_squared_error(y, oof_preds)
    print(f"OOF RMSE: {oof_rmse:.6f}")

    final_rounds = int(np.clip(np.mean(best_iterations), 100, num_boost_round))
    dtrain_full = lgb.Dataset(X, label=y, free_raw_data=False)
    final_model = lgb.train(params, dtrain_full, num_boost_round=final_rounds)

    test_pred = final_model.predict(X_test, num_iteration=final_rounds)
    test_pred = np.clip(test_pred, 0.0, 10.0)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "quality": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
