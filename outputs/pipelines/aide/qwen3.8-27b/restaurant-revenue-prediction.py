import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

os.makedirs("./working", exist_ok=True)

TRAIN_PATH = "./input/train.parquet"
TEST_PATH = "./input/test.parquet"
SUBMISSION_PATH = "./working/submission.csv"

TARGET = "revenue"
CAT_COLS = ["City", "City Group", "Type"]
SEED = 42
TARGET_ENC_M = 20

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

train = train.replace([np.inf, -np.inf], np.nan)
train = train.dropna(subset=[TARGET])
y = train[TARGET].astype(float).to_numpy()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    if "Open Date" in df.columns:
        dates = pd.to_datetime(df["Open Date"], errors="coerce")
        df["Open_Year"] = dates.dt.year
        df["Open_Month"] = dates.dt.month
        df = df.drop(columns=["Open Date"])

    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].astype(str).replace("nan", "Unknown")

    non_cat_cols = [c for c in df.columns if c not in CAT_COLS]
    for col in non_cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if non_cat_cols:
        df[non_cat_cols] = (
            df[non_cat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    return df


def fit_target_encoders(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list,
    m: int = TARGET_ENC_M,
):
    encoders = {}
    if len(y) == 0:
        return encoders

    global_mean = float(np.mean(y))

    for col in cat_cols:
        if col not in X_raw.columns:
            continue

        cats = X_raw[col].astype(str)
        tmp = pd.DataFrame({"cat": cats, "y": y})
        agg = tmp.groupby("cat")["y"].agg(["sum", "count"])
        smoothed = (agg["sum"] + m * global_mean) / (agg["count"] + m)

        encoders[col] = {
            "smoothed": smoothed,
            "global_mean": global_mean,
        }

    return encoders


def apply_target_encoders(X_raw: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    out = X_raw.copy()

    for col, info in encoders.items():
        if col in out.columns:
            out[f"{col}_target_enc"] = (
                out[col]
                .astype(str)
                .map(info["smoothed"])
                .fillna(info["global_mean"])
                .astype(float)
            )
        else:
            out[f"{col}_target_enc"] = info["global_mean"]

    return out


def encode_features(raw_df: pd.DataFrame, encoder=None, fit: bool = False):
    cat_cols = [c for c in CAT_COLS if c in raw_df.columns]
    non_cat_cols = [c for c in raw_df.columns if c not in cat_cols]

    out = raw_df[non_cat_cols].copy()

    if cat_cols:
        if fit:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoded = encoder.fit_transform(raw_df[cat_cols]).astype(int) + 1
        else:
            encoded = encoder.transform(raw_df[cat_cols]).astype(int) + 1

        out[cat_cols] = encoded

    return out, encoder


train_raw = preprocess(train)
test_raw = preprocess(test)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbose": -1,
    "num_threads": os.cpu_count() or 1,
}

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rmses = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_raw), start=1):
    X_train_raw = train_raw.iloc[train_idx]
    y_train = y[train_idx]
    X_val_raw = train_raw.iloc[val_idx]
    y_val = y[val_idx]

    target_encoders = fit_target_encoders(X_train_raw, y_train, CAT_COLS)
    X_train_raw_enc = apply_target_encoders(X_train_raw, target_encoders)
    X_val_raw_enc = apply_target_encoders(X_val_raw, target_encoders)

    X_train, encoder = encode_features(X_train_raw_enc, fit=True)
    X_val, _ = encode_features(X_val_raw_enc, encoder=encoder, fit=False)

    common_cols = [c for c in X_train.columns if c in X_val.columns]
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]

    categorical_features = [c for c in CAT_COLS if c in common_cols]

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(-1)],
    )

    best_iter = max(1, int(model.best_iteration))
    best_iterations.append(best_iter)

    val_pred = model.predict(X_val, num_iteration=best_iter)
    fold_rmse = root_mean_squared_error(y_val, val_pred)
    cv_rmses.append(fold_rmse)

    print(f"Fold {fold}: RMSE={fold_rmse:.6f}, best_iteration={best_iter}")

cv_rmse = float(np.mean(cv_rmses))
cv_rmse_std = float(np.std(cv_rmses))
print(f"5-Fold CV RMSE: {cv_rmse:.6f} +/- {cv_rmse_std:.6f}")
print(f"Validation Metric: {cv_rmse:.6f}")

final_boost_rounds = int(np.clip(np.mean(best_iterations), 1, 3000))

final_target_encoders = fit_target_encoders(train_raw, y, CAT_COLS)
X_train_final_raw = apply_target_encoders(train_raw, final_target_encoders)
X_test_final_raw = apply_target_encoders(test_raw, final_target_encoders)

X_train_final, final_encoder = encode_features(X_train_final_raw, fit=True)
X_test_final, _ = encode_features(X_test_final_raw, encoder=final_encoder, fit=False)

common_cols = [c for c in X_train_final.columns if c in X_test_final.columns]
X_train_final = X_train_final[common_cols]
X_test_final = X_test_final[common_cols]

categorical_features = [c for c in CAT_COLS if c in common_cols]

dtrain_final = lgb.Dataset(
    X_train_final,
    label=y,
    categorical_feature=categorical_features,
    free_raw_data=False,
)

final_model = lgb.train(
    params,
    dtrain_final,
    num_boost_round=final_boost_rounds,
)

test_pred = final_model.predict(X_test_final)
test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)
test_pred = np.clip(test_pred, 0.0, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_raw), dtype=int),
        "revenue": test_pred,
    }
)
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Saved submission to {SUBMISSION_PATH}")
