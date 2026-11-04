import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def main():
    # 1. Load Data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "LOANTOASSET"

    # Determine ID column for test set
    id_col = None
    for candidate in ["id", "ID", "cert", "CERT", "id_code"]:
        if candidate in test_df.columns:
            id_col = candidate
            break

    if id_col is None:
        test_ids = np.arange(len(test_df))
    else:
        test_ids = test_df[id_col].values
        train_df = train_df.drop(columns=[id_col])
        test_df = test_df.drop(columns=[id_col])

    # Separate raw features and target
    X_raw = train_df.drop(columns=[target_col])
    y_raw = train_df[target_col].values

    X_test_raw = test_df.copy()
    if target_col in X_test_raw.columns:
        X_test_raw = X_test_raw.drop(columns=[target_col])

    # 2. Validation Split (Strict Isolation)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42
    )

    # Re-index to keep standard DataFrame indexing
    X_tr = X_tr.reset_index(drop=True)
    X_va = X_va.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)

    # 3. Categorical Processing & Frequency Encoding (Fit strictly on Train)
    categorical_cols = X_tr.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in categorical_cols:
        X_tr[col] = X_tr[col].astype(str).fillna("missing")
        X_va[col] = X_va[col].astype(str).fillna("missing")
        X_test_raw[col] = X_test_raw[col].astype(str).fillna("missing")

        freq_map = X_tr[col].value_counts(normalize=True).to_dict()
        X_tr[f"{col}_freq"] = X_tr[col].map(freq_map).fillna(0.0)
        X_va[f"{col}_freq"] = X_va[col].map(freq_map).fillna(0.0)
        X_test_raw[f"{col}_freq"] = X_test_raw[col].map(freq_map).fillna(0.0)

        X_tr = X_tr.drop(columns=[col])
        X_va = X_va.drop(columns=[col])
        X_test_raw = X_test_raw.drop(columns=[col])

    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()

    # 4. Imputation (Fit on Train Split ONLY)
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr[num_cols]), columns=num_cols)
    X_va_imp = pd.DataFrame(imputer.transform(X_va[num_cols]), columns=num_cols)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_raw[num_cols]), columns=num_cols)

    # 5. Skewness Transformation & Domain Feature Engineering
    skews = X_tr_imp.skew()
    skewed_cols = skews[skews > 1.5].index.tolist()

    for col in skewed_cols:
        if (X_tr_imp[col] >= 0).all():
            X_tr_imp[f"{col}_log1p"] = np.log1p(X_tr_imp[col])
            X_va_imp[f"{col}_log1p"] = np.log1p(np.maximum(0, X_va_imp[col]))
            X_test_imp[f"{col}_log1p"] = np.log1p(np.maximum(0, X_test_imp[col]))

    base_num_features = [c for c in num_cols if not c.endswith("_freq")]
    if len(base_num_features) >= 2:
        for df in [X_tr_imp, X_va_imp, X_test_imp]:
            df["row_mean"] = df[base_num_features].mean(axis=1)
            df["row_std"] = df[base_num_features].std(axis=1).fillna(0.0)
            df["row_sum"] = df[base_num_features].sum(axis=1)
            df["row_max"] = df[base_num_features].max(axis=1)
            df["row_min"] = df[base_num_features].min(axis=1)
            df["row_zeros"] = (df[base_num_features] == 0).sum(axis=1)

    if len(base_num_features) >= 2:
        top_cols = base_num_features[: min(5, len(base_num_features))]
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                c1, c2 = top_cols[i], top_cols[j]

                X_tr_imp[f"{c1}_minus_{c2}"] = X_tr_imp[c1] - X_tr_imp[c2]
                X_va_imp[f"{c1}_minus_{c2}"] = X_va_imp[c1] - X_va_imp[c2]
                X_test_imp[f"{c1}_minus_{c2}"] = X_test_imp[c1] - X_test_imp[c2]

                X_tr_imp[f"{c1}_ratio_{c2}"] = X_tr_imp[c1] / (
                    X_tr_imp[c2].abs() + 1e-5
                )
                X_va_imp[f"{c1}_ratio_{c2}"] = X_va_imp[c1] / (
                    X_va_imp[c2].abs() + 1e-5
                )
                X_test_imp[f"{c1}_ratio_{c2}"] = X_test_imp[c1] / (
                    X_test_imp[c2].abs() + 1e-5
                )

    # 6. Scaling (Fit on Train Split ONLY)
    scaler = RobustScaler()
    final_cols = X_tr_imp.columns.tolist()

    X_tr_scaled = pd.DataFrame(
        scaler.fit_transform(X_tr_imp), columns=final_cols
    ).fillna(0)
    X_va_scaled = pd.DataFrame(scaler.transform(X_va_imp), columns=final_cols).fillna(0)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imp), columns=final_cols
    ).fillna(0)

    # 7. Model Building & Training
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=3000,
        learning_rate=0.015,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_tr_scaled,
        y_tr,
        eval_set=[(X_va_scaled, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    # 8. Evaluation & Inference
    val_preds = model.predict(X_va_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_va, val_preds))

    test_preds = model.predict(X_test_scaled)
    test_preds = np.clip(test_preds, a_min=0.0, a_max=None)

    # 9. Output Generation
    os.makedirs("./submission", exist_ok=True)
    submission_df = pd.DataFrame({"id": test_ids, "LOANTOASSET": test_preds})
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {val_rmse}")


if __name__ == "__main__":
    main()
