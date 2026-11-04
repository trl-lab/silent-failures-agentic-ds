import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/prudential-life-insurance-assessment/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/prudential-life-insurance-assessment/data/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    y_train = train_df["Response"].values.astype(int)
    
    df_combined = pd.concat([train_df.drop(columns=["Response"]), test_df], ignore_index=True)

    if "Id" in df_combined.columns:
        df_combined = df_combined.drop(columns=["Id"])
    if "id" in df_combined.columns:
        df_combined = df_combined.drop(columns=["id"])

    med_kw_cols = [c for c in df_combined.columns if c.startswith("Medical_Keyword_")]
    if med_kw_cols:
        df_combined["Medical_Keyword_count"] = df_combined[med_kw_cols].sum(axis=1)

    if "Product_Info_2" in df_combined.columns:
        df_combined["Product_Info_2_char"] = df_combined["Product_Info_2"].astype(str).str[0]
        df_combined["Product_Info_2_num"] = pd.to_numeric(df_combined["Product_Info_2"].astype(str).str[1:], errors="coerce").fillna(-1).astype(int)

    if "BMI" in df_combined.columns and "Ins_Age" in df_combined.columns:
        df_combined["BMI_Age"] = df_combined["BMI"] * df_combined["Ins_Age"]
    if "BMI" in df_combined.columns and "Weight" in df_combined.columns:
        df_combined["BMI_Weight"] = df_combined["BMI"] * df_combined["Weight"]
    if "Height" in df_combined.columns and "Weight" in df_combined.columns:
        df_combined["Height_Weight"] = df_combined["Height"] * df_combined["Weight"]
        df_combined["BMI_Calc"] = df_combined["Weight"] / (df_combined["Height"] ** 2 + 1e-5)

    df_combined["null_count"] = df_combined.isnull().sum(axis=1)

    cat_cols = df_combined.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        df_combined[c] = pd.factorize(df_combined[c].astype(str))[0]

    n_train = len(train_df)
    X_train = df_combined.iloc[:n_train].copy()
    X_test = df_combined.iloc[n_train:].copy()

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(n_train)
    oof_preds_xgb = np.zeros(n_train)
    oof_preds_cb = np.zeros(n_train)

    test_preds_lgb = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_cb = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

        # LightGBM
        model_lgb = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        oof_preds_lgb[val_idx] = model_lgb.predict(X_va)
        test_preds_lgb += model_lgb.predict(X_test) / n_splits

        # XGBoost
        model_xgb = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42 + fold,
            n_jobs=-1,
            early_stopping_rounds=50,
            tree_method='hist'
        )
        model_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        oof_preds_xgb[val_idx] = model_xgb.predict(X_va)
        test_preds_xgb += model_xgb.predict(X_test) / n_splits

        # CatBoost
        model_cb = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=50,
            thread_count=-1
        )
        model_cb.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            use_best_model=True
        )
        oof_preds_cb[val_idx] = model_cb.predict(X_va)
        test_preds_cb += model_cb.predict(X_test) / n_splits

    oof_preds = (oof_preds_lgb + oof_preds_xgb + oof_preds_cb) / 3.0
    test_preds = (test_preds_lgb + test_preds_xgb + test_preds_cb) / 3.0

    class ThresholdOptimizer:
        def __init__(self):
            self.coef_ = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

        def _loss(self, coef, X, y):
            coef_sorted = np.sort(coef)
            preds = np.digitize(X, coef_sorted) + 1
            return -cohen_kappa_score(y, preds, weights='quadratic')

        def fit(self, X, y):
            res_nm = minimize(self._loss, self.coef_, args=(X, y), method='Nelder-Mead', options={'maxiter': 1000})
            self.coef_ = np.sort(res_nm.x)
            res_powell = minimize(self._loss, self.coef_, args=(X, y), method='Powell', options={'maxiter': 1000})
            self.coef_ = np.sort(res_powell.x)

        def predict(self, X):
            return np.digitize(X, self.coef_) + 1

    optimizer = ThresholdOptimizer()
    optimizer.fit(oof_preds, y_train)

    final_test_preds = optimizer.predict(test_preds)

    sub = pd.DataFrame({
        "id": np.arange(len(final_test_preds)),
        "Response": final_test_preds
    })
    sub.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()