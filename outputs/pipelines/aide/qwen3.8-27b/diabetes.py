import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


def main() -> None:
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    output_dir = "./working"
    os.makedirs(output_dir, exist_ok=True)

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target = "class"
    feature_cols = [
        col for col in train.columns if col != target and col.lower() != "id"
    ]

    zero_keywords = [
        "glucose",
        "blood",
        "pressure",
        "skin",
        "triceps",
        "thickness",
        "insulin",
        "bmi",
        "mass",
        "age",
    ]

    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in df.columns:
            if col == target:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

            col_lower = col.lower()
            if any(keyword in col_lower for keyword in zero_keywords):
                df[col] = df[col].replace(0, np.nan)

        return df

    train_clean = clean_dataframe(train)
    test_clean = clean_dataframe(test)

    y = train_clean[target]
    if not pd.api.types.is_numeric_dtype(y):
        label_map = {
            "tested_positive": 1,
            "tested_negative": 0,
            "positive": 1,
            "negative": 0,
            "1": 1,
            "0": 0,
        }
        y = y.map(label_map)

    y = y.astype(int)

    X = train_clean[feature_cols].astype(float)

    for col in feature_cols:
        if col not in test_clean.columns:
            test_clean[col] = np.nan

    X_test = test_clean[feature_cols].astype(float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_predictions = np.zeros(X.shape[0], dtype=float)
    test_predictions = np.zeros(X_test.shape[0], dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train_fold)
        X_val_imputed = imputer.transform(X_val_fold)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)

        model = SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            C=1.0,
            gamma="scale",
        )
        model.fit(X_train_scaled, y_train_fold)

        oof_predictions[val_idx] = model.predict_proba(X_val_scaled)[:, 1]

        if X_test.shape[0] > 0:
            X_test_imputed = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imputed)
            test_predictions += model.predict_proba(X_test_scaled)[:, 1]

        fold_auc = roc_auc_score(y_val_fold, oof_predictions[val_idx])
        print(f"Fold {fold} ROC AUC: {fold_auc:.5f}")

    test_predictions /= 5.0
    overall_auc = roc_auc_score(y, oof_predictions)
    print(f"OOF ROC AUC: {overall_auc:.5f}")

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_predictions), dtype=int),
            "class": np.clip(test_predictions, 0.0, 1.0),
        }
    )
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
