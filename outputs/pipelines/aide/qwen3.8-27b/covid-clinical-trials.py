import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

TARGET = "days_to_complete"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Convert target to numeric and remove rows with missing target.
y = pd.to_numeric(train[TARGET], errors="coerce").astype(float)
train = train.loc[y.notna()].reset_index(drop=True)
y = y.loc[y.notna()].values

X = train.drop(columns=[TARGET])

# Remove obvious identifiers and target-related columns to avoid leakage.
drop_names = {
    "id",
    "index",
    "row_id",
    "trial_id",
    "nct_id",
    "study_id",
    "participant_id",
    "target",
    "label",
}
leak_substrings = (
    "complete",
    "completion",
    "days_to",
    "duration",
    "finish",
    "target",
    "label",
)
drop_cols = []
n = len(X)
for col in X.columns:
    cl = col.lower()
    if cl in drop_names or any(s in cl for s in leak_substrings):
        drop_cols.append(col)
        continue
    s = X[col]
    if s.isna().all() or s.nunique(dropna=True) <= 1 or s.isna().mean() > 0.95:
        drop_cols.append(col)
        continue
    if (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s)
    ) and s.nunique(dropna=True) > 0.99 * n:
        drop_cols.append(col)
X = X.drop(columns=drop_cols)

# If no features remain, use mean prediction fallback.
if X.shape[1] == 0:
    pred_test = np.full(len(test), float(np.mean(y)))
    pd.DataFrame({"id": np.arange(len(test)), "days_to_complete": pred_test}).to_csv(
        os.path.join(WORK_DIR, "submission.csv"), index=False
    )
    print("CV RMSE: nan")
    raise SystemExit

cat_cols = [
    c
    for c in X.columns
    if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c])
]


def prepare(X_in, cat_cols, encoder=None, fit=False):
    X_out = X_in.copy()
    for col in X_out.columns:
        if pd.api.types.is_datetime64_any_dtype(X_out[col]):
            s = X_out[col]
            try:
                if getattr(s.dt, "tz", None) is not None:
                    s = s.dt.tz_localize(None)
            except Exception:
                pass
            X_out[col] = s.astype("int64") / 1e9
            X_out[col] = X_out[col].replace({-9223372036854775808: np.nan})
        elif pd.api.types.is_bool_dtype(X_out[col]):
            X_out[col] = X_out[col].astype("int8")
    if len(cat_cols) > 0:
        cat_df = X_out[cat_cols].astype(object).fillna("__MISSING__")
        if fit:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoder.fit(cat_df)
        elif encoder is None:
            raise ValueError("encoder must be provided when fit=False")
        enc_arr = encoder.transform(cat_df)
        enc_arr = np.where(enc_arr == -1, np.nan, enc_arr)
        for i, col in enumerate(cat_cols):
            X_out[col] = enc_arr[:, i]
    for col in X_out.columns:
        if not pd.api.types.is_numeric_dtype(X_out[col]):
            X_out[col] = pd.to_numeric(X_out[col], errors="coerce")
    return X_out, encoder


params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
best_iterations = []
clip_pred = float(np.min(y)) >= 0

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_tr_raw = X.iloc[tr_idx]
    X_val_raw = X.iloc[val_idx]
    X_tr, enc = prepare(X_tr_raw, cat_cols, fit=True)
    X_val, _ = prepare(X_val_raw, cat_cols, encoder=enc, fit=False)

    dtrain = lgb.Dataset(
        X_tr, label=y[tr_idx], categorical_feature=cat_cols, free_raw_data=False
    )
    dval = lgb.Dataset(
        X_val,
        label=y[val_idx],
        categorical_feature=cat_cols,
        reference=dtrain,
        free_raw_data=False,
    )
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
    )
    pred = model.predict(X_val, num_iteration=model.best_iteration)
    if clip_pred:
        pred = np.clip(pred, 0.0, None)
    rmse = float(np.sqrt(mean_squared_error(y[val_idx], pred)))
    rmse_scores.append(rmse)
    best_iterations.append(int(model.best_iteration))
    print(f"Fold {fold} RMSE: {rmse:.6f}, best_iteration: {model.best_iteration}")

cv_rmse = float(np.mean(rmse_scores))
print(f"CV RMSE: {cv_rmse:.6f}")

# Train final model on all training data using the average best iteration from CV.
X_all, enc_all = prepare(X, cat_cols, fit=True)
dtrain_all = lgb.Dataset(
    X_all, label=y, categorical_feature=cat_cols, free_raw_data=False
)
final_rounds = max(1, int(np.mean(best_iterations))) if best_iterations else 1000
final_model = lgb.train(
    params,
    dtrain_all,
    num_boost_round=final_rounds,
    callbacks=[lgb.log_evaluation(-1)],
)

test_clean = test.copy()
if TARGET in test_clean.columns:
    test_clean = test_clean.drop(columns=[TARGET])
test_clean = test_clean.reindex(columns=X.columns)
X_test, _ = prepare(test_clean, cat_cols, encoder=enc_all, fit=False)
pred_test = final_model.predict(X_test, num_iteration=final_rounds)
if clip_pred:
    pred_test = np.clip(pred_test, 0.0, None)

submission = pd.DataFrame(
    {
        "id": np.arange(len(pred_test), dtype=int),
        "days_to_complete": pred_test.astype(float),
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows.")
