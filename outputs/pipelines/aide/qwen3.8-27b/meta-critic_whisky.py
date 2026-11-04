import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

os.makedirs("./working", exist_ok=True)

TARGET = "Meta Critic"
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

y = pd.to_numeric(train[TARGET], errors="coerce").replace([np.inf, -np.inf], np.nan)
X = train.drop(columns=[TARGET], errors="ignore").copy()

id_names = {"id", "index", "idx", "row_id", "rowid", "sample_id", "sampleid"}
drop_cols = [
    c
    for c in X.columns
    if str(c).lower() in id_names or str(c).lower().startswith("unnamed")
]
X = X.drop(columns=drop_cols, errors="ignore")
test_X = test.drop(columns=drop_cols, errors="ignore").copy()

test_X = test_X.reindex(columns=X.columns, fill_value=np.nan)

valid = y.notna()
X = X.loc[valid].copy()
y = y.loc[valid].copy()

if X.empty:
    print("CV RMSE: 0.0000")
    pd.DataFrame({"id": np.arange(len(test)), "Meta Critic": 0.0}).to_csv(
        "./working/submission.csv", index=False
    )
    raise SystemExit

keep = []
cat_features = []
for c in X.columns:
    s = X[c]
    ts = test_X[c]

    if pd.api.types.is_bool_dtype(s):
        X[c] = s.astype(float)
        test_X[c] = pd.to_numeric(ts, errors="coerce")
        if X[c].nunique(dropna=True) > 1:
            keep.append(c)
        continue

    if pd.api.types.is_numeric_dtype(s):
        X[c] = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        test_X[c] = pd.to_numeric(ts, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if X[c].notna().sum() > 0 and X[c].nunique(dropna=True) > 1:
            keep.append(c)
        continue

    num = pd.to_numeric(s, errors="coerce")
    non_null = s.notna().sum()
    if (
        non_null > 0
        and num.notna().sum() >= 0.8 * non_null
        and num.nunique(dropna=True) > 1
    ):
        X[c] = num.replace([np.inf, -np.inf], np.nan)
        test_X[c] = pd.to_numeric(ts, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if X[c].notna().sum() > 0 and X[c].nunique(dropna=True) > 1:
            keep.append(c)
        continue

    if s.dtype.name == "category":
        cats = [c for c in s.cat.categories if pd.notna(c)]
    else:
        cats = list(s.dropna().unique())

    if len(cats) > 1 and s.notna().sum() > 0:
        X[c] = pd.Categorical(s, categories=cats)
        test_X[c] = pd.Categorical(ts, categories=cats)
        keep.append(c)
        cat_features.append(c)

X = X[keep].copy()
test_X = test_X[keep].copy()

if len(keep) == 0:
    cv_rmse = float(np.std(y)) if len(y) > 1 else 0.0
    print(f"CV RMSE: {cv_rmse:.4f}")
    pred = float(y.mean()) if len(y) > 0 else 0.0
    pd.DataFrame({"id": np.arange(len(test)), "Meta Critic": pred}).to_csv(
        "./working/submission.csv", index=False
    )
    raise SystemExit

for c in keep:
    if c not in cat_features:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        test_X[c] = pd.to_numeric(test_X[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 5,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

cv_scores = []
best_iters = []
if len(y) >= 2:
    n_splits = min(5, len(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(X):
        dtrain = lgb.Dataset(
            X.iloc[tr_idx], label=y.iloc[tr_idx], categorical_feature=cat_features
        )
        dvalid = lgb.Dataset(
            X.iloc[va_idx],
            label=y.iloc[va_idx],
            categorical_feature=cat_features,
            reference=dtrain,
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        best_iter = model.best_iteration if model.best_iteration > 0 else 100
        best_iters.append(best_iter)
        preds = model.predict(X.iloc[va_idx], num_iteration=best_iter)
        cv_scores.append(root_mean_squared_error(y.iloc[va_idx], preds))
else:
    cv_scores = [float(np.std(y)) if len(y) > 1 else 0.0]
    best_iters = [100]

cv_rmse = float(np.mean(cv_scores))
print(f"CV RMSE: {cv_rmse:.4f}")

final_rounds = int(np.clip(np.mean(best_iters), 1, 2000))
dtrain_full = lgb.Dataset(X, label=y, categorical_feature=cat_features)
final_model = lgb.train(params, dtrain_full, num_boost_round=final_rounds)

if len(test_X) == 0:
    test_pred = np.array([])
else:
    test_pred = final_model.predict(test_X)
    if y.notna().any():
        test_pred = np.clip(test_pred, float(y.min()), float(y.max()))

submission = pd.DataFrame({"id": np.arange(len(test)), "Meta Critic": test_pred})
submission.to_csv("./working/submission.csv", index=False)


def main():
    ...
    if X.empty:
        ...
        return
    ...


if __name__ == "__main__":
    main()


def main():
    ...
    if X.empty:
        ...
        return
    ...
    if len(keep) == 0:
        ...
        return
    ...


if __name__ == "__main__":
    main()
