import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def bmi_rule_one_hot(bmi: np.ndarray) -> np.ndarray:
    """
    Return one-hot probabilities for the standard BMI obesity categories.
    Class order follows the LabelEncoder order used below, which is lexicographic:
    0: Insufficient Weight
    1: Normal Weight
    2: Obesity level I
    3: Obesity level II
    4: Obesity level III
    5: Overweight level I
    6: Overweight level II
    """
    probs = np.zeros((len(bmi), 7), dtype=float)

    insufficient = bmi < 18.5
    normal = (bmi >= 18.5) & (bmi < 25.0)
    overweight_i = (bmi >= 25.0) & (bmi < 30.0)
    overweight_ii = (bmi >= 30.0) & (bmi < 35.0)
    obesity_i = (bmi >= 35.0) & (bmi < 40.0)
    obesity_ii = (bmi >= 40.0) & (bmi < 50.0)
    obesity_iii = bmi >= 50.0

    probs[insufficient, 0] = 1.0
    probs[normal, 1] = 1.0
    probs[obesity_i, 2] = 1.0
    probs[obesity_ii, 3] = 1.0
    probs[obesity_iii, 4] = 1.0
    probs[overweight_i, 5] = 1.0
    probs[overweight_ii, 6] = 1.0

    # Fallback for any NaN/edge cases: use Normal Weight as the most common central class.
    nan_mask = np.isnan(bmi)
    if np.any(nan_mask):
        probs[nan_mask] = 0.0
        probs[nan_mask, 1] = 1.0

    return probs


def main():
    os.makedirs("./working", exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target = "NObeyesdad"

    train = train.dropna(subset=[target])
    y_raw = train[target].astype(str).str.strip().copy()
    train = train.drop(columns=[target])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    if n_classes == 1:
        print("CV accuracy: 1.00000")
        submission = pd.DataFrame(
            {
                "id": np.arange(len(test)),
                "NObeyesdad": label_encoder.classes_[0],
            }
        )
        submission.to_csv("./working/submission.csv", index=False)
        return

    explicit_categoricals = ["Gender", "CAEC", "CALC", "MTRANS"]
    categorical_columns = [
        c
        for c in train.columns
        if train[c].dtype == object or train[c].dtype.name in ("category", "string")
    ]

    for col in explicit_categoricals:
        if col in train.columns and col not in categorical_columns:
            categorical_columns.append(col)

    for col in categorical_columns:
        train[col] = train[col].fillna("Unknown").astype(str).str.strip()

        if col in test.columns:
            test[col] = test[col].fillna("Unknown").astype(str).str.strip()
        else:
            test[col] = pd.Series(["Unknown"] * len(test), index=test.index)

        train_categories = sorted(train[col].unique())
        if "Unknown" not in train_categories:
            train_categories.append("Unknown")

        train[col] = train[col].astype("category", categories=train_categories)
        test[col] = test[col].where(test[col].isin(train_categories), "Unknown")
        test[col] = test[col].astype("category", categories=train_categories)

    numeric_columns = [c for c in train.columns if c not in categorical_columns]

    for col in numeric_columns:
        if col not in test.columns:
            test[col] = np.nan

        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

        train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)

        median_value = train[col].median()
        if pd.isna(median_value):
            median_value = 0.0

        train[col] = train[col].fillna(median_value)
        test[col] = test[col].fillna(median_value)

    if all(
        col in train.columns and col in test.columns for col in ["Height", "Weight"]
    ):

        def compute_bmi(frame):
            height = frame["Height"].astype(float)
            weight = frame["Weight"].astype(float)
            bmi = weight / (height / 100.0) ** 2
            return bmi.replace([np.inf, -np.inf], np.nan)

        train["BMI"] = compute_bmi(train)
        bmi_median = train["BMI"].median()
        if pd.isna(bmi_median):
            bmi_median = 0.0

        train["BMI"] = train["BMI"].fillna(bmi_median)
        test["BMI"] = compute_bmi(test).fillna(bmi_median)
    else:
        train["BMI"] = np.nan
        test["BMI"] = np.nan
        bmi_median = 0.0
        train["BMI"] = train["BMI"].fillna(bmi_median)
        test["BMI"] = test["BMI"].fillna(bmi_median)

    feature_columns = [c for c in train.columns if c in test.columns]
    categorical_columns = [c for c in categorical_columns if c in feature_columns]

    X = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    # The rule-based predictor uses BMI in the exact LabelEncoder class order.
    # If the target labels are not the expected seven classes, fall back to LGB-only.
    expected_classes = [
        "Insufficient Weight",
        "Normal Weight",
        "Obesity level I",
        "Obesity level II",
        "Obesity level III",
        "Overweight level I",
        "Overweight level II",
    ]
    use_bmi_rule = n_classes == 7 and list(label_encoder.classes_) == expected_classes

    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_error",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    best_iterations = []
    num_boost_round = 2000

    # Weight the BMI rule slightly more heavily because the target is primarily BMI-driven.
    rule_weight = 0.6
    lgb_weight = 1.0 - rule_weight

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        dtrain = lgb.Dataset(
            X.iloc[train_idx],
            label=y[train_idx],
            categorical_feature=categorical_columns,
        )
        dvalid = lgb.Dataset(
            X.iloc[valid_idx],
            label=y[valid_idx],
            categorical_feature=categorical_columns,
            reference=dtrain,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        best_iteration = max(1, int(model.best_iteration or num_boost_round))
        best_iterations.append(best_iteration)

        valid_lgb_probs = model.predict(X.iloc[valid_idx], num_iteration=best_iteration)

        if use_bmi_rule:
            valid_rule_probs = bmi_rule_one_hot(
                X.iloc[valid_idx]["BMI"].to_numpy(dtype=float)
            )
            valid_probs = rule_weight * valid_rule_probs + lgb_weight * valid_lgb_probs
        else:
            valid_probs = valid_lgb_probs

        valid_pred_labels = np.argmax(valid_probs, axis=1)
        fold_score = accuracy_score(y[valid_idx], valid_pred_labels)
        fold_scores.append(fold_score)
        print(f"Fold {fold} accuracy: {fold_score:.5f}")

    cv_accuracy = float(np.mean(fold_scores))
    print(f"CV accuracy: {cv_accuracy:.5f}")

    final_rounds = max(1, int(np.median(best_iterations)))

    final_model = lgb.train(
        params,
        lgb.Dataset(X, label=y, categorical_feature=categorical_columns),
        num_boost_round=final_rounds,
    )

    test_lgb_probs = final_model.predict(X_test, num_iteration=final_rounds)

    if use_bmi_rule:
        test_rule_probs = bmi_rule_one_hot(X_test["BMI"].to_numpy(dtype=float))
        test_probs = rule_weight * test_rule_probs + lgb_weight * test_lgb_probs
    else:
        test_probs = test_lgb_probs

    test_pred_labels = label_encoder.inverse_transform(np.argmax(test_probs, axis=1))

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "NObeyesdad": test_pred_labels,
        }
    )

    submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
