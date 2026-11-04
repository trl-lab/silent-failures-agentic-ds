import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/champs-scalar-coupling/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/champs-scalar-coupling/data/test.parquet"
TARGET = "scalar_coupling_constant"
ID_COL = "id"
RANDOM_STATE = 42
ALPHA = 20.0
N_ESTIMATORS = 800
LEARNING_RATE = 0.05


def clean_numeric(df: pd.DataFrame, cols: list, fill_values: dict) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(fill_values.get(col, 0.0))
    return df


def clean_categorical(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = "missing"
        df[col] = df[col].fillna("missing").astype(str)
    return df


def fit_fill_values(train_df: pd.DataFrame, numeric_cols: list) -> dict:
    fills = {}
    for col in numeric_cols:
        if col in train_df.columns:
            s = pd.to_numeric(train_df[col], errors="coerce")
            med = s.median()
            fills[col] = 0.0 if pd.isna(med) else float(med)
        else:
            fills[col] = 0.0
    return fills


def target_encode(
    train_code: np.ndarray,
    train_y: np.ndarray,
    other_code: np.ndarray,
    alpha: float,
    global_mean: float,
):
    train_code_s = pd.Series(train_code, dtype=np.int64)
    other_code_s = pd.Series(other_code, dtype=np.int64)

    tmp = pd.DataFrame({"code": train_code_s, "y": train_y})
    grouped = tmp.groupby("code", observed=True)["y"]
    sums = grouped.sum()
    counts = grouped.count()

    train_sum = train_code_s.map(sums).fillna(0.0).to_numpy(dtype=float)
    train_count = train_code_s.map(counts).fillna(0.0).to_numpy(dtype=float)
    train_enc = (train_sum - train_y + alpha * global_mean) / (train_count - 1.0 + alpha)

    other_sum = other_code_s.map(sums).fillna(0.0).to_numpy(dtype=float)
    other_count = other_code_s.map(counts).fillna(0.0).to_numpy(dtype=float)
    other_enc = (other_sum + alpha * global_mean) / (other_count + alpha)

    return train_enc, train_count, other_enc, other_count


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    a0 = df["atom_index_0"].astype(float)
    a1 = df["atom_index_1"].astype(float)
    df["atom_min"] = np.minimum(a0, a1)
    df["atom_max"] = np.maximum(a0, a1)
    df["atom_diff"] = np.abs(a0 - a1)
    df["atom_sum"] = a0 + a1
    return df


def build_features(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    other = other_df.copy()

    fill_values = fit_fill_values(train, numeric_cols)
    train = clean_numeric(train, numeric_cols, fill_values)
    other = clean_numeric(other, numeric_cols, fill_values)

    train = clean_categorical(train, categorical_cols)
    other = clean_categorical(other, categorical_cols)

    for col in categorical_cols:
        categories = list(train[col].unique())
        train[col + "_code"] = pd.Categorical(train[col], categories=categories).codes.astype(np.int32)
        other[col + "_code"] = pd.Categorical(other[col], categories=categories).codes.astype(np.int32)

    train = train.drop(columns=categorical_cols)
    other = other.drop(columns=categorical_cols)

    train = add_derived_features(train)
    other = add_derived_features(other)

    y = train[target].to_numpy(dtype=float)
    global_mean = float(np.nanmean(y)) if len(y) else 0.0

    if "type_code" in train.columns:
        tr_enc, tr_cnt, ot_enc, ot_cnt = target_encode(
            train["type_code"].to_numpy(), y, other["type_code"].to_numpy(), ALPHA, global_mean
        )
        train["type_target"] = tr_enc
        train["type_count"] = tr_cnt
        other["type_target"] = ot_enc
        other["type_count"] = ot_cnt

    if "molecule_name_code" in train.columns:
        tr_enc, tr_cnt, ot_enc, ot_cnt = target_encode(
            train["molecule_name_code"].to_numpy(), y, other["molecule_name_code"].to_numpy(), ALPHA, global_mean
        )
        train["molecule_target"] = tr_enc
        train["molecule_count"] = tr_cnt
        other["molecule_target"] = ot_enc
        other["molecule_count"] = ot_cnt

    if "molecule_name_code" in train.columns and "type_code" in train.columns:
        multiplier = int(max(train["type_code"].max(), other["type_code"].max())) + 1
        train_combined = train["molecule_name_code"].to_numpy(dtype=np.int64) * multiplier + train["type_code"].to_numpy(dtype=np.int64)
        other_combined = other["molecule_name_code"].to_numpy(dtype=np.int64) * multiplier + other["type_code"].to_numpy(dtype=np.int64)

        tr_enc, tr_cnt, ot_enc, ot_cnt = target_encode(
            train_combined, y, other_combined, ALPHA, global_mean
        )
        train["molecule_type_target"] = tr_enc
        train["molecule_type_count"] = tr_cnt
        other["molecule_type_target"] = ot_enc
        other["molecule_type_count"] = ot_cnt

    feature_cols = [c for c in train.columns if c != target]

    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce").fillna(0.0).astype(float)
        other[col] = pd.to_numeric(other[col], errors="coerce").fillna(0.0).astype(float)

    return train[feature_cols], other[feature_cols]


def main() -> None:
    train_raw = pd.read_parquet(TRAIN_PATH)
    test_raw = pd.read_parquet(TEST_PATH)

    train_raw = train_raw.drop(columns=[ID_COL], errors="ignore")
    test_raw = test_raw.drop(columns=[ID_COL], errors="ignore")

    train_raw = train_raw.dropna(subset=[TARGET])

    feature_cols = [c for c in train_raw.columns if c != TARGET]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_raw[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    train_part, val_part = train_test_split(
        train_raw,
        test_size=0.15,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    X_train_val, X_val = build_features(
        train_part,
        val_part,
        numeric_cols,
        categorical_cols,
        TARGET,
    )
    y_train_val = train_part[TARGET].to_numpy(dtype=float)
    y_val = val_part[TARGET].to_numpy(dtype=float)

    val_model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=400,
        learning_rate=LEARNING_RATE,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    val_model.fit(X_train_val, y_train_val)
    val_pred = val_model.predict(X_val)
    val_mae = float(np.mean(np.abs(y_val - val_pred)))

    del X_train_val, X_val, y_train_val, y_val, val_model, val_pred

    X_train, X_test = build_features(
        train_raw,
        test_raw,
        numeric_cols,
        categorical_cols,
        TARGET,
    )
    y_train = train_raw[TARGET].to_numpy(dtype=float)

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "scalar_coupling_constant": test_pred.astype(float),
        }
    )
    submission.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()