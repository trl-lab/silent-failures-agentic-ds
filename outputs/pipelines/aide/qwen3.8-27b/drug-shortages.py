import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

TARGET = "availability"
ID_COL = "id"
RANDOM_STATE = 42
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


def drop_id_and_target(df, keep_target=False):
    df = df.copy()
    drop_cols = []
    if ID_COL in df.columns:
        drop_cols.append(ID_COL)
    if TARGET in df.columns and not keep_target:
        drop_cols.append(TARGET)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def identify_feature_types(X, features):
    categorical_cols = []
    numeric_cols = []

    for col in features:
        s = X[col]

        if (
            s.dtype == object
            or s.dtype.name == "category"
            or pd.api.types.is_bool_dtype(s)
        ):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        else:
            converted = pd.to_numeric(s, errors="coerce")
            non_missing = s.notna().sum()
            if non_missing == 0 or converted.notna().sum() >= 0.5 * non_missing:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

    return categorical_cols, numeric_cols


def preprocess(X, categorical_cols, numeric_cols, encoder=None, fit=False):
    X = X.copy()

    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    if categorical_cols:
        cat_X = X[categorical_cols].astype("object")
        cat_X = cat_X.fillna("__missing__")

        if fit:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoder.fit(cat_X)
            X[categorical_cols] = encoder.transform(cat_X).astype(int)
        else:
            X[categorical_cols] = encoder.transform(cat_X).astype(int)

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, encoder


def make_params(n_classes):
    base = {
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
        "nthread": -1,
        "verbosity": -1,
    }

    if n_classes == 2:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
        }
    else:
        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
        }

    params.update(base)
    return params


def train_model(X_train, y_train, X_valid, y_valid, params, categorical_feature):
    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=dtrain,
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(-1),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    best_iteration = getattr(model, "best_iteration", 0)
    if best_iteration is None or best_iteration <= 0:
        best_iteration = NUM_BOOST_ROUND

    return model, best_iteration


def predict_model(model, X, n_classes, categorical_feature):
    dtest = lgb.Dataset(
        X,
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )
    proba = model.predict(dtest)

    if n_classes == 2:
        return proba[:, 1]
    return proba


def probabilities_to_labels(proba, n_classes):
    if n_classes == 2:
        return (proba > 0.5).astype(int)
    return np.argmax(proba, axis=1)


def save_submission(predictions, n_rows):
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(n_rows, dtype=int),
            "availability": predictions,
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    return submission


def main():
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    train = train_raw.dropna(subset=[TARGET]).reset_index(drop=True)
    test = test_raw.copy()

    y_raw = train[TARGET]
    X_train_full = drop_id_and_target(train, keep_target=False)
    X_test = drop_id_and_target(test, keep_target=False)

    features = [c for c in X_train_full.columns if c in X_test.columns]
    X_train_full = X_train_full[features]
    X_test = X_test[features]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    if n_classes == 1 or len(features) == 0:
        fallback_label = (
            label_encoder.classes_[0] if n_classes == 1 else y_raw.mode().iloc[0]
        )
        cv_accuracy = 1.0 if n_classes == 1 else 0.0
        print(f"CV Accuracy: {cv_accuracy:.6f}")
        save_submission(np.full(len(test), fallback_label), len(test))
        return

    categorical_cols, numeric_cols = identify_feature_types(X_train_full, features)
    params = make_params(n_classes)

    class_counts = pd.Series(y_encoded).value_counts()
    min_class_count = int(class_counts.min())

    if min_class_count >= 5:
        n_splits = 5
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    elif min_class_count >= 2:
        n_splits = min_class_count
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        n_splits = 2
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = []
    best_iterations = []

    for train_idx, valid_idx in cv.split(X_train_full, y_encoded):
        X_train_fold_raw = X_train_full.iloc[train_idx]
        X_valid_fold_raw = X_train_full.iloc[valid_idx]
        y_train_fold = y_encoded[train_idx]
        y_valid_fold = y_encoded[valid_idx]

        X_train_fold, encoder = preprocess(
            X_train_fold_raw,
            categorical_cols,
            numeric_cols,
            fit=True,
        )
        X_valid_fold, _ = preprocess(
            X_valid_fold_raw,
            categorical_cols,
            numeric_cols,
            encoder=encoder,
            fit=False,
        )

        model, best_iteration = train_model(
            X_train_fold,
            y_train_fold,
            X_valid_fold,
            y_valid_fold,
            params,
            categorical_cols,
        )
        best_iterations.append(best_iteration)

        valid_proba = predict_model(model, X_valid_fold, n_classes, categorical_cols)
        valid_pred = probabilities_to_labels(valid_proba, n_classes)
        fold_accuracy = accuracy_score(y_valid_fold, valid_pred)
        cv_scores.append(fold_accuracy)

    cv_accuracy = float(np.mean(cv_scores))
    print(f"CV Accuracy: {cv_accuracy:.6f}")

    X_train_full_processed, full_encoder = preprocess(
        X_train_full,
        categorical_cols,
        numeric_cols,
        fit=True,
    )
    X_test_processed, _ = preprocess(
        X_test,
        categorical_cols,
        numeric_cols,
        encoder=full_encoder,
        fit=False,
    )

    final_boost_rounds = (
        int(np.clip(np.mean(best_iterations), 100, NUM_BOOST_ROUND))
        if best_iterations
        else 500
    )

    dtrain_full = lgb.Dataset(
        X_train_full_processed,
        label=y_encoded,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=final_boost_rounds,
        callbacks=[lgb.log_evaluation(-1)],
    )

    test_proba = predict_model(
        final_model, X_test_processed, n_classes, categorical_cols
    )
    test_pred_encoded = probabilities_to_labels(test_proba, n_classes)
    test_pred = label_encoder.inverse_transform(test_pred_encoded)

    save_submission(test_pred, len(test))


if __name__ == "__main__":
    main()
