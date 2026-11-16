import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def get_params(n_classes, seed):
    return {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_error",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "verbosity": -1,
    }


def main():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    out_dir = "./working"
    os.makedirs(out_dir, exist_ok=True)

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target = "class"
    train = train.dropna(subset=[target]).reset_index(drop=True)

    if train[target].dtype == object or str(train[target].dtype).startswith("category"):
        train[target] = train[target].astype(str).str.strip()

    exclude_cols = {target, "id"}
    feature_cols = [
        col for col in train.columns if col not in exclude_cols and col in test.columns
    ]

    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    X_train = train[feature_cols].copy()
    y_train = train[target].copy()
    X_test = test[feature_cols].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_train)
    classes = label_encoder.classes_
    n_classes = len(classes)

    seeds = [42, 7, 123, 456, 789]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    best_iterations = {seed: [] for seed in seeds}
    max_rounds = 1000

    for train_idx, val_idx in skf.split(X_train, y):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        medians = X_tr.median(numeric_only=True)
        X_tr_imp = X_tr.fillna(medians).fillna(0.0)
        X_val_imp = X_val.fillna(medians).fillna(0.0)

        val_proba_sum = np.zeros((len(X_val_imp), n_classes), dtype=np.float64)

        for seed in seeds:
            params = get_params(n_classes, seed)

            train_set = lgb.Dataset(X_tr_imp, label=y_tr)
            valid_set = lgb.Dataset(X_val_imp, label=y_val, reference=train_set)

            booster = lgb.train(
                params,
                train_set,
                num_boost_round=max_rounds,
                valid_sets=[valid_set],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
            )

            best_iteration = getattr(booster, "best_iteration", 0)
            if not best_iteration or best_iteration <= 0:
                best_iteration = max_rounds

            best_iterations[seed].append(int(best_iteration))
            val_proba_sum += booster.predict(X_val_imp, num_iteration=best_iteration)

        val_pred = np.argmax(val_proba_sum, axis=1)
        cv_accuracies.append(accuracy_score(y_val, val_pred))

    print(
        f"5-fold CV accuracy: {np.mean(cv_accuracies):.4f} "
        f"(+/- {np.std(cv_accuracies):.4f})"
    )

    medians_all = X_train.median(numeric_only=True)
    X_train_imp = X_train.fillna(medians_all).fillna(0.0)
    X_test_imp = X_test.fillna(medians_all).fillna(0.0)

    test_proba_sum = np.zeros((len(X_test_imp), n_classes), dtype=np.float64)

    for seed in seeds:
        seed_rounds = int(np.clip(np.median(best_iterations[seed]), 50, max_rounds))
        params = get_params(n_classes, seed)

        final_set = lgb.Dataset(X_train_imp, label=y)
        booster = lgb.train(params, final_set, num_boost_round=seed_rounds)

        test_proba_sum += booster.predict(X_test_imp, num_iteration=seed_rounds)

    test_pred = label_encoder.inverse_transform(np.argmax(test_proba_sum, axis=1))

    submission = pd.DataFrame(
        {
            "id": np.arange(len(X_test)),
            "class": test_pred,
        }
    )
    submission.to_csv(os.path.join(out_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
