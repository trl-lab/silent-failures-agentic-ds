import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target = "scalar_coupling_constant"
train = train.dropna(subset=[target]).reset_index(drop=True)

exclude = {"id", target, "full_mol_idx", "atom_idx1", "atom_idx2"}
feature_cols = [c for c in train.columns if c not in exclude and c in test.columns]


def sanitize(name):
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


numeric_cols = []
cat_cols = []

for c in feature_cols:
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)
    else:
        converted_tr = pd.to_numeric(train[c], errors="coerce")
        if converted_tr.notna().mean() >= 0.9:
            train[c] = converted_tr
            test[c] = pd.to_numeric(test[c], errors="coerce")
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

parts_train = []
parts_test = []

if numeric_cols:
    parts_train.append(train[numeric_cols].astype(np.float32))
    parts_test.append(test[numeric_cols].astype(np.float32))
else:
    parts_train.append(pd.DataFrame(index=train.index))
    parts_test.append(pd.DataFrame(index=test.index))

for c in cat_cols:
    if train[c].isna().all() and test[c].isna().all():
        continue

    nunique = train[c].nunique(dropna=True)
    if nunique == 0:
        continue

    prefix = sanitize(c)

    if nunique <= 500:
        dtr = pd.get_dummies(train[c], prefix=prefix, dtype=np.float32)
        dte = pd.get_dummies(test[c], prefix=prefix, dtype=np.float32)
        dte = dte.reindex(columns=dtr.columns, fill_value=0)
        parts_train.append(dtr)
        parts_test.append(dte)
    else:
        codes_tr, uniques = pd.factorize(train[c], sort=True)
        mapping = {v: i for i, v in enumerate(uniques)}
        codes_te = test[c].map(mapping)

        codes_tr = pd.Series(codes_tr, index=train.index).astype(np.float32)
        codes_te = codes_te.astype(np.float32)

        codes_tr[codes_tr < 0] = np.nan
        codes_te[codes_te < 0] = np.nan

        parts_train.append(pd.DataFrame({f"{prefix}_code": codes_tr}))
        parts_test.append(pd.DataFrame({f"{prefix}_code": codes_te}))

X_train = pd.concat(parts_train, axis=1)
X_test = pd.concat(parts_test, axis=1)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train.columns = [f"{i}_{c}" for i, c in enumerate(X_train.columns)]
X_test.columns = X_train.columns

X_train = X_train.replace([np.inf, -np.inf], np.nan).astype(np.float32)
X_test = X_test.replace([np.inf, -np.inf], np.nan).astype(np.float32)

valid_cols = [c for c in X_train.columns if X_train[c].notna().any()]
X_train = X_train[valid_cols]
X_test = X_test[valid_cols]

y = train[target].astype(np.float32).values
X_np = X_train.to_numpy(dtype=np.float32)
X_test_np = X_test.to_numpy(dtype=np.float32)
del X_train, X_test

if "full_mol_idx" in train.columns:
    groups, _ = pd.factorize(train["full_mol_idx"].fillna(-1), sort=False)
else:
    groups = np.arange(len(train))


def make_splits(X, y, groups):
    n_groups = len(np.unique(groups))
    if n_groups >= 5 and n_groups * 2 < len(y):
        rng = np.random.default_rng(42)
        order = rng.permutation(len(y))
        X = X[order]
        y = y[order]
        groups = groups[order]
        return GroupKFold(n_splits=5).split(X, y, groups=groups), X, y
    else:
        return KFold(n_splits=5, shuffle=True, random_state=42).split(X, y), X, y


splits, X_np, y = make_splits(X_np, y, groups)

params = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

if X_np.shape[1] == 0:
    val_maes = []
    for tr_idx, val_idx in splits:
        pred = np.full(len(val_idx), y[tr_idx].mean(), dtype=np.float32)
        val_maes.append(float(np.mean(np.abs(pred - y[val_idx]))))

    cv_mae = float(np.mean(val_maes))
    print(f"Validation MAE: {cv_mae:.6f}")

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "scalar_coupling_constant": float(y.mean()),
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("submission.csv saved.")
else:
    test_preds = np.zeros(len(X_test_np), dtype=np.float32)
    val_maes = []

    for fold, (tr_idx, val_idx) in enumerate(splits):
        dtrain = lgb.Dataset(X_np[tr_idx], label=y[tr_idx])
        dval = lgb.Dataset(X_np[val_idx], label=y[val_idx], reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(500)],
        )

        best_it = model.best_iteration
        if best_it <= 0:
            best_it = model.current_iteration()

        val_pred = model.predict(X_np[val_idx], num_iteration=best_it)
        mae = float(np.mean(np.abs(val_pred - y[val_idx])))
        val_maes.append(mae)
        print(f"Fold {fold} validation MAE: {mae:.6f}")

        test_preds += model.predict(X_test_np, num_iteration=best_it)

    test_preds /= len(val_maes)
    cv_mae = float(np.mean(val_maes))
    print(f"Validation MAE: {cv_mae:.6f}")

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "scalar_coupling_constant": test_preds,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("submission.csv saved.")
