import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def build_model():
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        min_samples_leaf=1,
        random_state=42,
    )


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "car"
    feature_cols = [col for col in train.columns if col != target]

    X = train[feature_cols].astype(str).copy()
    y = train[target].astype(str).copy()
    X_test = test[feature_cols].astype(str).copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_enc = encoder.fit_transform(X_train_fold)
        X_val_enc = encoder.transform(X_val_fold)

        model = build_model()
        model.fit(X_train_enc, y_train_fold)

        val_pred = model.predict(X_val_enc)
        fold_acc = accuracy_score(y_val_fold, val_pred)
        fold_scores.append(fold_acc)

        print(f"Fold {fold} accuracy: {fold_acc:.4f}")

    cv_accuracy = float(np.mean(fold_scores))
    print(f"Validation accuracy (5-fold CV): {cv_accuracy:.4f}")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_train_full_enc = encoder.fit_transform(X)

    final_model = build_model()
    final_model.fit(X_train_full_enc, y)

    X_test_enc = encoder.transform(X_test)
    test_pred = final_model.predict(X_test_enc)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test), dtype=int),
            "car": test_pred,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
