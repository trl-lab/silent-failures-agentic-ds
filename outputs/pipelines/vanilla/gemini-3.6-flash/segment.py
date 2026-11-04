import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

def main():
    train_path = "/path/to/user/workspace/ds_failures/datasets/openml/segment/data/train.parquet"
    test_path = "/path/to/user/workspace/ds_failures/datasets/openml/segment/data/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "class"
    
    X_train = train_df.drop(columns=[target_col])
    y_train_raw = train_df[target_col]
    X_test = test_df.copy()

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    num_classes = len(le.classes_)
    test_probs = np.zeros((len(X_test_imp), num_classes))

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X_train_imp, y_train):
        X_tr, y_tr = X_train_imp[train_idx], y_train[train_idx]
        
        lgb = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        lgb.fit(X_tr, y_tr)

        cb = CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            random_seed=42,
            verbose=0,
            thread_count=-1
        )
        cb.fit(X_tr, y_tr)

        et = ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        et.fit(X_tr, y_tr)

        rf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)

        p_lgb = lgb.predict_proba(X_test_imp)
        p_cb = cb.predict_proba(X_test_imp)
        p_et = et.predict_proba(X_test_imp)
        p_rf = rf.predict_proba(X_test_imp)

        fold_probs = (p_lgb + p_cb + p_et + p_rf) / 4.0
        test_probs += fold_probs / n_splits

    final_preds_idx = np.argmax(test_probs, axis=1)
    final_preds = le.inverse_transform(final_preds_idx)

    sub = pd.DataFrame({
        "id": np.arange(len(final_preds)),
        "class": final_preds
    })
    sub.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()