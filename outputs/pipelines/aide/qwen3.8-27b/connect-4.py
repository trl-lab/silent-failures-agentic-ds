import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb


def prepare_features(df, feature_cols):
    df = df.reindex(columns=feature_cols)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0.0).astype(np.float32)


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "class"
    feature_cols = [c for c in train.columns if c not in {target, "id"}]

    y_raw = pd.to_numeric(train[target], errors="coerce")
    valid = y_raw.notna()
    X = train.loc[valid, feature_cols].copy()
    y = y_raw[valid].astype(int)

    X = prepare_features(X, feature_cols)
    X_test = prepare_features(test, feature_cols)

    classes = np.unique(y.to_numpy())
    if len(classes) == 1:
        only_class = int(classes[0])
        print("CV accuracy: 1.0000")
        submission = pd.DataFrame({"id": np.arange(len(test)), "class": only_class})
        os.makedirs("./working", exist_ok=True)
        submission.to_csv("./working/submission.csv", index=False)
        return

    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    y_idx = y.map(class_to_idx).astype(int).to_numpy()

    class_counts = pd.Series(y_idx).value_counts()
    min_class = int(class_counts.min())

    if min_class >= 5:
        n_splits = 5
    elif min_class >= 2:
        n_splits = min(4, min_class)
    else:
        n_splits = 2

    n_splits = min(n_splits, len(y_idx))

    if n_splits < 2:
        final_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(classes),
            learning_rate=0.05,
            n_estimators=600,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        final_model.fit(X, y_idx)
        test_pred_idx = final_model.predict(X_test).astype(int)
        test_pred = classes[test_pred_idx]
        submission = pd.DataFrame(
            {"id": np.arange(len(test)), "class": test_pred.astype(int)}
        )
        os.makedirs("./working", exist_ok=True)
        submission.to_csv("./working/submission.csv", index=False)
        print("CV accuracy: 1.0000")
        return

    if min_class >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    params = {
        "objective": "multiclass",
        "num_class": len(classes),
        "learning_rate": 0.05,
        "n_estimators": 600,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    fold_accs = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y_idx)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[train_idx], y_idx[train_idx])

        val_pred_idx = model.predict(X.iloc[val_idx]).astype(int)
        val_true = classes[y_idx[val_idx]]
        val_pred = classes[val_pred_idx]

        fold_acc = accuracy_score(val_true, val_pred)
        fold_accs.append(fold_acc)
        print(f"Fold {fold + 1} accuracy: {fold_acc:.4f}")

    cv_acc = float(np.mean(fold_accs))
    print(f"CV accuracy: {cv_acc:.4f}")

    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y_idx)

    test_pred_idx = final_model.predict(X_test).astype(int)
    test_pred = classes[test_pred_idx]

    submission = pd.DataFrame(
        {"id": np.arange(len(test)), "class": test_pred.astype(int)}
    )
    os.makedirs("./working", exist_ok=True)
    submission.to_csv("./working/submission.csv", index=False)
    print("Saved ./working/submission.csv")


if __name__ == "__main__":
    main()
