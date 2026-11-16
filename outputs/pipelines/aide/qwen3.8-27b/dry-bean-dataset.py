import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]
TARGET = "Class"


def get_features(train: pd.DataFrame, test: pd.DataFrame):
    available = [f for f in FEATURES if f in train.columns and f in test.columns]
    if len(available) == len(FEATURES):
        return available
    return [c for c in train.columns if c != TARGET and c in test.columns]


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = train.dropna(subset=[TARGET]).reset_index(drop=True)

    features = get_features(train, test)

    X = train[features].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    y_raw = train[TARGET]
    X_test = test[features].astype(np.float64).replace([np.inf, -np.inf], np.nan)

    label_encoder = LabelEncoder().fit(y_raw)
    y = label_encoder.transform(y_raw)
    num_class = len(label_encoder.classes_)

    params = {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    best_iterations = []
    max_rounds = 5000

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        train_data = lgb.Dataset(
            X.iloc[train_idx], label=y[train_idx], free_raw_data=True
        )
        valid_data = lgb.Dataset(
            X.iloc[val_idx],
            label=y[val_idx],
            reference=train_data,
            free_raw_data=True,
        )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=max_rounds,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iter = model.best_iteration if model.best_iteration > 0 else max_rounds
        best_iterations.append(best_iter)

        pred_iter = model.best_iteration if model.best_iteration > 0 else None
        val_proba = model.predict(X.iloc[val_idx], num_iteration=pred_iter)
        val_pred = np.argmax(val_proba, axis=1)

        fold_acc = accuracy_score(y[val_idx], val_pred)
        fold_scores.append(fold_acc)
        print(f"Fold {fold} accuracy: {fold_acc:.6f}")

    cv_accuracy = float(np.mean(fold_scores))
    print(f"5-fold CV accuracy: {cv_accuracy:.6f}")

    final_rounds = max(1, int(np.mean(best_iterations)))

    full_data = lgb.Dataset(X, label=y, free_raw_data=True)
    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_proba = final_model.predict(X_test, num_iteration=final_rounds)
    test_pred = label_encoder.inverse_transform(np.argmax(test_proba, axis=1))

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "Class": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
