import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

NUMERIC_COLS = ["carat", "x", "y", "z", "depth", "table"]
DIM_COLS = ["carat", "x", "y", "z"]
CAT_COLS = ["cut", "color", "clarity"]

CUT_MAP = {
    "Fair": 1,
    "Good": 2,
    "Very Good": 3,
    "Premium": 4,
    "Ideal": 5,
}

COLOR_MAP = {
    "J": 1,
    "I": 2,
    "H": 3,
    "G": 4,
    "F": 5,
    "E": 6,
    "D": 7,
}

CLARITY_MAP = {
    "I1": 1,
    "SI2": 2,
    "SI1": 3,
    "VS2": 4,
    "VS1": 5,
    "VVS2": 6,
    "VVS1": 7,
    "IF": 8,
}

for col in NUMERIC_COLS:
    train[col] = pd.to_numeric(train[col], errors="coerce")
train["price"] = pd.to_numeric(train["price"], errors="coerce")

train = train.dropna(subset=["price"]).copy()
valid = train["price"] > 0
for col in DIM_COLS:
    valid &= train[col] > 0
train = train[valid].copy()

for col in ["depth", "table"]:
    train[col] = train[col].where((train[col] >= 0) & (train[col] <= 200), np.nan)

numeric_medians = train[NUMERIC_COLS].median()

cat_modes = {}
for col in CAT_COLS:
    s = train[col].astype("string").str.strip()
    s = s.where(s != "", pd.NA)
    mode = s.mode()
    cat_modes[col] = mode.iloc[0] if len(mode) else "Unknown"
    train[col] = s.fillna(cat_modes[col])

code_medians = {}
for col, mapping in [("cut", CUT_MAP), ("color", COLOR_MAP), ("clarity", CLARITY_MAP)]:
    train[col + "_code"] = train[col].map(mapping)
    code_medians[col + "_code"] = train[col + "_code"].median()
    train[col + "_code"] = train[col + "_code"].fillna(code_medians[col + "_code"])


def build_features(df, is_train=False):
    out = pd.DataFrame(index=df.index)

    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    for col in DIM_COLS:
        out[col] = out[col].where(out[col] > 0, np.nan)

    for col in ["depth", "table"]:
        out[col] = out[col].where((out[col] >= 0) & (out[col] <= 200), np.nan)

    out = out.fillna(numeric_medians)

    for col, mapping in [
        ("cut", CUT_MAP),
        ("color", COLOR_MAP),
        ("clarity", CLARITY_MAP),
    ]:
        s = df[col].astype("string").str.strip()
        s = s.where(s != "", pd.NA)
        s = s.fillna(cat_modes[col])
        out[col + "_code"] = s.map(mapping).fillna(code_medians[col + "_code"])

    out["carat2"] = out["carat"] ** 2
    out["carat_sqrt"] = np.sqrt(out["carat"])
    out["volume"] = out["x"] * out["y"] * out["z"]
    out["xy"] = out["x"] * out["y"]
    out["depth_carat"] = out["depth"] * out["carat"]

    return out


X = build_features(train, is_train=True)
y = np.log1p(train["price"])
X_test = build_features(test, is_train=False)

feature_cols = list(X.columns)
X = X[feature_cols]
X_test = X_test[feature_cols]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
best_iterations = []

for train_idx, val_idx in kf.split(X):
    model = xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        verbose=False,
    )

    val_pred = np.expm1(model.predict(X.iloc[val_idx]))
    val_true = np.expm1(y.iloc[val_idx])
    rmse_scores.append(root_mean_squared_error(val_true, val_pred))
    best_iterations.append(model.best_iteration)

cv_rmse = float(np.mean(rmse_scores))
print(f"CV RMSE: {cv_rmse:.4f}")

final_n_estimators = int(max(100, min(5000, np.mean(best_iterations))))
final_model = xgb.XGBRegressor(
    n_estimators=final_n_estimators,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

final_model.fit(X, y)
test_pred = np.expm1(final_model.predict(X_test))
test_pred = np.maximum(test_pred, 0.0)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "price": test_pred,
    }
)
submission.to_csv("./working/submission.csv", index=False)
