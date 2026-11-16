import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder

WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    area_cols = [
        c
        for c in ["GrLivArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "GarageArea"]
        if c in df.columns
    ]
    if area_cols:
        df["TotalSF"] = (
            df[area_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        )

    if "OverallQual" in df.columns and "OverallCond" in df.columns:
        df["OverallScore"] = pd.to_numeric(df["OverallQual"], errors="coerce").fillna(
            0
        ) + pd.to_numeric(df["OverallCond"], errors="coerce").fillna(0)

    bath_cols = [
        c
        for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
        if c in df.columns
    ]
    if bath_cols:
        df["TotalBath"] = (
            df[bath_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        )

    for col in ["GarageArea", "GarageCars", "GarageYrBlt"]:
        if col in df.columns:
            df[f"Has{col}"] = (pd.to_numeric(df[col], errors="coerce") > 0).astype(int)

    for col in ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"]:
        if col in df.columns:
            df[f"Has{col}"] = df[col].notna().astype(int)

    if "LotFrontage" in df.columns:
        df["LotFrontageMissing"] = df["LotFrontage"].isna().astype(int)

    return df


def preprocess(
    X: pd.DataFrame,
    cat_cols: list,
    num_cols: list,
    encoder: OrdinalEncoder | None = None,
    fit: bool = False,
):
    X = X.copy()

    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in num_cols if c in X.columns]

    X_num = X[num_cols].copy()
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
    X_num = X_num.replace([np.inf, -np.inf], np.nan)

    X_cat = X[cat_cols].astype(object).copy()
    X_cat = X_cat.fillna("missing")

    if cat_cols:
        if fit:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoded = encoder.fit_transform(X_cat)
        else:
            if encoder is None:
                raise ValueError("encoder must be provided when fit=False")
            encoded = encoder.transform(X_cat)

        X_cat_enc = pd.DataFrame(
            encoded.astype(int) + 1,
            index=X.index,
            columns=cat_cols,
        )
    else:
        X_cat_enc = pd.DataFrame(index=X.index)

    X_out = pd.concat([X_num, X_cat_enc], axis=1)
    cols = [c for c in num_cols + cat_cols if c in X_out.columns]
    X_out = X_out[cols]

    return X_out, cat_cols, encoder


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    train = train.dropna(subset=["SalePrice"])
    y = train["SalePrice"].astype(float)

    X = train.drop(columns=["SalePrice"])
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])
    if "Id" in test.columns:
        test = test.drop(columns=["Id"])

    X = add_features(X)
    test = add_features(test)

    X = X.reset_index(drop=True)
    test = test.reset_index(drop=True)

    test = test.reindex(columns=X.columns)

    all_nan_cols = X.columns[X.isna().all()]
    X = X.drop(columns=all_nan_cols)
    test = test.drop(columns=all_nan_cols)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()

    num_cols += [c for c in bool_cols if c not in num_cols]
    other_cols = [c for c in X.columns if c not in num_cols and c not in cat_cols]
    cat_cols += other_cols

    num_cols = list(dict.fromkeys(num_cols))
    cat_cols = list(dict.fromkeys(cat_cols))

    X = X[num_cols + cat_cols]
    test = test[num_cols + cat_cols]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        X_tr_p, cat_feats, enc = preprocess(X_tr, cat_cols, num_cols, fit=True)
        X_va_p, _, _ = preprocess(X_va, cat_cols, num_cols, encoder=enc)

        train_set = lgb.Dataset(
            X_tr_p,
            label=np.log1p(y_tr),
            categorical_feature=cat_feats,
        )
        valid_set = lgb.Dataset(
            X_va_p,
            label=np.log1p(y_va),
            categorical_feature=cat_feats,
            reference=train_set,
        )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=2000,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)],
        )

        pred_log = model.predict(X_va_p, num_iteration=model.best_iteration)
        pred = np.expm1(pred_log).clip(0.0, None)
        oof[va_idx] = pred

        best_iters.append(max(1, int(model.best_iteration)))
        print(f"Fold {fold} best_iteration: {model.best_iteration}")

    cv_rmsle = rmsle(y, oof)
    print(f"CV RMSLE: {cv_rmsle:.6f}")

    X_all_p, cat_feats_all, enc_all = preprocess(X, cat_cols, num_cols, fit=True)
    X_test_p, _, _ = preprocess(test, cat_cols, num_cols, encoder=enc_all)

    final_rounds = int(np.mean(best_iters)) if best_iters else 500

    train_set = lgb.Dataset(
        X_all_p,
        label=np.log1p(y),
        categorical_feature=cat_feats_all,
    )
    final_model = lgb.train(params, train_set, num_boost_round=final_rounds)

    test_pred_log = final_model.predict(X_test_p)
    test_pred = np.expm1(test_pred_log).clip(0.0, None)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            "SalePrice": test_pred,
        }
    )
    submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)


if __name__ == "__main__":
    main()
