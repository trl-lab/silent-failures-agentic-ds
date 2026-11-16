import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/openml/eucalyptus/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/openml/eucalyptus/data/test.parquet"

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    target_col = "Utility"

    y_raw = train[target_col].copy()
    X = train.drop(columns=[target_col]).copy()
    X_test = test.copy()

    if target_col in X_test.columns:
        X_test = X_test.drop(columns=[target_col])

    if "id" in X.columns:
        X = X.drop(columns=["id"])
    if "id" in X_test.columns:
        X_test = X_test.drop(columns=["id"])

    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)
    num_classes = len(le_target.classes_)

    cat_cols = []
    num_cols = []

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            cat_cols.append(col)
        elif col in ['Abbrev', 'Locality', 'Map_Ref', 'Latitude', 'Sp', 'PMCno', 'Rep', 'Year']:
            cat_cols.append(col)
        else:
            num_cols.append(col)

    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("missing")
        X_test[col] = X_test[col].astype(str).fillna("missing")

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32)
    X_cat_oe = oe.fit_transform(X[cat_cols])
    X_test_cat_oe = oe.transform(X_test[cat_cols])

    X_encoded = X.copy()
    X_test_encoded = X_test.copy()
    for i, col in enumerate(cat_cols):
        X_encoded[col] = X_cat_oe[:, i]
        X_test_encoded[col] = X_test_cat_oe[:, i]

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_encoded)
    X_test_imp = imputer.transform(X_test_encoded)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    test_preds = np.zeros((len(test), num_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        X_tr_enc, X_va_enc = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
        X_tr_imp, X_va_imp = X_imp[train_idx], X_imp[val_idx]

        cb = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            cat_features=cat_cols,
            verbose=0,
            random_seed=42 + fold
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        preds_cb = cb.predict_proba(X_test)

        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42 + fold,
            verbose=-1
        )
        lgb_model.fit(
            X_tr_enc, y_tr,
            eval_set=[(X_va_enc, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        preds_lgb = lgb_model.predict_proba(X_test_encoded)

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42 + fold,
            eval_metric='mlogloss',
            early_stopping_rounds=50
        )
        xgb_model.fit(
            X_tr_enc, y_tr,
            eval_set=[(X_va_enc, y_va)],
            verbose=False
        )
        preds_xgb = xgb_model.predict_proba(X_test_encoded)

        et_model = ExtraTreesClassifier(
            n_estimators=300,
            random_state=42 + fold,
            n_jobs=-1
        )
        et_model.fit(X_tr_imp, y_tr)
        preds_et = et_model.predict_proba(X_test_imp)

        rf_model = RandomForestClassifier(
            n_estimators=300,
            random_state=42 + fold,
            n_jobs=-1
        )
        rf_model.fit(X_tr_imp, y_tr)
        preds_rf = rf_model.predict_proba(X_test_imp)

        fold_preds = (
            0.25 * preds_cb +
            0.25 * preds_lgb +
            0.20 * preds_xgb +
            0.15 * preds_et +
            0.15 * preds_rf
        )
        test_preds += fold_preds / skf.n_splits

    final_pred_idx = np.argmax(test_preds, axis=1)
    final_preds = le_target.inverse_transform(final_pred_idx)

    out_df = pd.DataFrame({
        "id": np.arange(len(test)),
        "Utility": final_preds
    })
    out_df.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()