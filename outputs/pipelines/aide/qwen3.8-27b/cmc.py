import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

TARGET = "Contraceptive_method_used"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUND = 2000

os.makedirs("./working", exist_ok=True)


def load_data():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    feature_cols = [
        c for c in train.columns if c not in (TARGET, ID_COL) and c in test.columns
    ]

    X = train[feature_cols].copy()
    y = train[TARGET].copy()
    X_test = test[feature_cols].copy()

    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    return X, y, X_test


def get_categorical_cols(X):
    known_categorical = {
        "Wife's education",
        "Husband's education",
        "Wife's religion",
        "Wife's now working?",
        "Husband's occupation",
        "Standard-of-living index",
        "Media exposure",
    }
    known_numeric = {
        "Wife's age",
        "Number of children ever born",
    }

    cols = []
    for c in X.columns:
        if c in known_numeric:
            continue
        if (
            c in known_categorical
            or not pd.api.types.is_numeric_dtype(X[c])
            or X[c].nunique() <= 10
        ):
            cols.append(c)
    return cols


def encode_categoricals(df, categorical_cols, reference_df=None):
    out = df.copy()
    for col in categorical_cols:
        if col not in out.columns:
            continue

        ref = reference_df if reference_df is not None else out
        cats = np.unique(ref[col].dropna().to_numpy())

        if len(cats) == 0:
            out[col] = out[col].astype("category")
        else:
            dtype = pd.CategoricalDtype(categories=cats, ordered=False)
            out[col] = out[col].astype(dtype)

    return out


def main():
    X, y_orig, X_test = load_data()
    categorical_cols = get_categorical_cols(X)

    classes = np.unique(y_orig.to_numpy())
    y = np.searchsorted(classes, y_orig.to_numpy()).astype(int)

    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": RANDOM_STATE,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_accs = []
    best_iters = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr_raw = X.iloc[tr_idx]
        X_val_raw = X.iloc[val_idx]

        X_tr = encode_categoricals(X_tr_raw, categorical_cols, reference_df=X_tr_raw)
        X_val = encode_categoricals(X_val_raw, categorical_cols, reference_df=X_tr_raw)

        y_tr = y[tr_idx]
        y_val = y[val_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr, categorical_features=categorical_cols)
        valid_set = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_set,
            categorical_features=categorical_cols,
        )

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(-1),
        ]

        model = lgb.train(
            params,
            train_set,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        best_iter = max(1, model.best_iteration)
        best_iters.append(best_iter)

        val_pred = model.predict(X_val, num_iteration=best_iter).argmax(axis=1)
        fold_accs.append(accuracy_score(y_val, val_pred))

    cv_accuracy = float(np.mean(fold_accs))
    print(f"CV Accuracy: {cv_accuracy:.4f}")

    X_train_final = encode_categoricals(X, categorical_cols, reference_df=X)
    X_test_final = encode_categoricals(X_test, categorical_cols, reference_df=X)

    final_rounds = max(1, int(np.median(best_iters))) if best_iters else 100

    final_train_set = lgb.Dataset(
        X_train_final,
        label=y,
        categorical_features=categorical_cols,
    )

    final_model = lgb.train(
        params,
        final_train_set,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_proba = final_model.predict(X_test_final, num_iteration=final_rounds)
    test_pred = classes[test_proba.argmax(axis=1)]

    submission = pd.DataFrame(
        {
            "id": np.arange(len(X_test)),
            "Contraceptive_method_used": test_pred.astype(int),
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved ./working/submission.csv")


if __name__ == "__main__":
    main()
