import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/madelon/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/madelon/data/test.parquet"
TARGET = "Class"
RANDOM_STATE = 42


def encode_target(y: pd.Series):
    codes, uniques = pd.factorize(y, sort=True)
    label_map = {i: uniques[i] for i in range(len(uniques))}
    return pd.Series(codes, index=y.index, dtype=int), label_map


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    sample = series.dropna()
    if len(sample) == 0:
        return True
    converted = pd.to_numeric(sample, errors="coerce")
    return converted.notna().mean() >= 0.9


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    feature_cols = [c for c in train_df.columns if c != target]
    X = train_df[feature_cols].copy()
    X_test = test_df.reindex(columns=feature_cols).copy()

    for col in X.columns:
        if is_numeric_like(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        else:
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                cats = X[col].cat.categories.tolist()
            else:
                cats = X[col].dropna().unique().tolist()
            cats = sorted(cats, key=str)
            mapping = {v: i for i, v in enumerate(cats)}
            X[col] = X[col].map(mapping).fillna(-1).astype(int)
            X_test[col] = X_test[col].map(mapping).fillna(-1).astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    return X, X_test


def impute_and_clean(X: pd.DataFrame, imputer: SimpleImputer, fit: bool = True) -> pd.DataFrame:
    if fit:
        X_imp = imputer.fit_transform(X)
    else:
        X_imp = imputer.transform(X)
    X_imp = pd.DataFrame(X_imp, columns=X.columns, index=X.index)
    return X_imp.fillna(0.0)


def make_selection_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def make_final_models():
    lgbm = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    xgb = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    catboost = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_state=RANDOM_STATE,
        verbose=False,
    )

    return [lgbm, xgb, catboost]


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    y_raw = train_df[TARGET]
    y, label_map = encode_target(y_raw)

    X, X_test = prepare_features(train_df, test_df, TARGET)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = impute_and_clean(X_train, imputer, fit=True)
    X_val_imp = impute_and_clean(X_val, imputer, fit=False)

    train_var = X_train_imp.var(axis=0)
    usable_cols = train_var[train_var > 0].index
    X_train_imp = X_train_imp[usable_cols]
    X_val_imp = X_val_imp[usable_cols]

    best_k = 50
    best_acc = -1.0

    if len(usable_cols) > 0:
        mi_scores = mutual_info_classif(X_train_imp, y_train, random_state=RANDOM_STATE)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
        ranked = np.argsort(mi_scores)[::-1]

        for k in [20, 30, 50, 80, 120]:
            k = min(k, len(usable_cols))
            idx = ranked[:k]
            model = make_selection_model()
            model.fit(X_train_imp.iloc[:, idx], y_train)
            acc = accuracy_score(y_val, model.predict(X_val_imp.iloc[:, idx]))
            if acc > best_acc:
                best_acc = acc
                best_k = k

    imputer_final = SimpleImputer(strategy="median")
    X_train_final = impute_and_clean(X, imputer_final, fit=True)
    X_test_final = impute_and_clean(X_test, imputer_final, fit=False)

    final_var = X_train_final.var(axis=0)
    final_cols = final_var[final_var > 0].index
    X_train_final = X_train_final[final_cols]
    X_test_final = X_test_final[final_cols]

    if len(final_cols) == 0:
        majority_class = int(y.mode().iloc[0])
        pred_codes = np.full(len(test_df), majority_class, dtype=int)
    else:
        mi_final = mutual_info_classif(X_train_final, y, random_state=RANDOM_STATE)
        mi_final = np.nan_to_num(mi_final, nan=0.0, posinf=0.0, neginf=0.0)
        ranked_final = np.argsort(mi_final)[::-1]
        k_final = min(max(best_k, 1), len(final_cols))
        idx_final = ranked_final[:k_final]

        X_train_sel = X_train_final.iloc[:, idx_final]
        X_test_sel = X_test_final.iloc[:, idx_final]

        probas = []
        for model in make_final_models():
            model.fit(X_train_sel, y)
            probas.append(model.predict_proba(X_test_sel))

        avg_proba = np.mean(probas, axis=0)
        pred_codes = np.argmax(avg_proba, axis=1)

    pred_labels = np.array([label_map[int(c)] for c in pred_codes])
    if np.issubdtype(pred_labels.dtype, np.number):
        pred_labels = pred_labels.astype(int)

    predictions = pd.DataFrame({
        "id": np.arange(len(test_df)),
        "Class": pred_labels,
    })
    predictions.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()